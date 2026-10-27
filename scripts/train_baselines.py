"""Train and compare baselines, log runs to MLflow (if installed), save the model.

The threshold is chosen on out-of-fold predictions from the training split and
then evaluated once on the held-out test split. Selecting it on the test set
would make the reported value optimistic.

Usage:  python scripts/train_baselines.py
View:   mlflow ui --backend-store-uri sqlite:///mlflow.db
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from telco_churn.data import load_clean
from telco_churn.economics import Economics
from telco_churn.evaluate import (
    best_threshold,
    bootstrap_capture_at_top,
    bootstrap_ci,
    breakeven_probability,
    calibration_slope_intercept,
    expected_net_value_curve,
    oracle_net_value_curve,
    paired_bootstrap_delta,
    summary_metrics,
    threshold_band,
)
from telco_churn.features import FEATURES, train_test
from telco_churn.model import MODEL_DIR, build_pipeline, save_bundle

SEED = 42
ECON = Economics()
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "reports" / "results.json"

try:  # MLflow is optional so the pipeline runs in a minimal environment.
    import mlflow
except ImportError:  # pragma: no cover - depends on environment
    mlflow = None


def candidates() -> dict[str, Pipeline]:
    return {
        "logistic_regression": build_pipeline("logistic", seed=SEED),
        "hist_gradient_boosting": build_pipeline("boosting", seed=SEED),
    }


def main() -> None:
    df = load_clean()
    X_train, X_test, y_train, y_test = train_test(df, seed=SEED)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    if mlflow is not None:
        mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
        mlflow.set_experiment("telco-churn-baselines")

    results: dict = {
        "dataset": {
            "customers": int(len(df)),
            "churn_rate": float(df.Churn.mean()),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
        },
        "economics": ECON.as_dict(),
        "breakeven_probability": breakeven_probability(
            ECON.contact_cost, ECON.value_saved, ECON.save_rate
        ),
        "models": {},
    }
    oof: dict[str, np.ndarray] = {}
    test_probs: dict[str, np.ndarray] = {}

    for name, pipe in candidates().items():
        oof[name] = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        pipe.fit(X_train, y_train)
        test_probs[name] = pipe.predict_proba(X_test)[:, 1]

        cv_metrics = summary_metrics(y_train, oof[name])
        test_metrics = summary_metrics(y_test, test_probs[name])
        intervals = {
            metric: bootstrap_ci(y_test, test_probs[name], metric=metric, seed=SEED)
            for metric in ("roc_auc", "brier")
        }
        capture = bootstrap_capture_at_top(y_test, test_probs[name], 0.2, seed=SEED)
        calibration = calibration_slope_intercept(y_test, test_probs[name])

        results["models"][name] = {
            "cv": cv_metrics,
            "test": test_metrics,
            "test_ci": intervals,
            "capture_top20": capture,
            "calibration": calibration,
        }

        if mlflow is not None:
            with mlflow.start_run(run_name=name):
                mlflow.log_param("model", name)
                mlflow.log_param("features", "signup-time only (no tenure/TotalCharges)")
                mlflow.log_param("seed", SEED)
                mlflow.log_metrics({f"cv_{k}": v for k, v in cv_metrics.items()
                                    if np.isfinite(v)})
                mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()
                                    if np.isfinite(v)})
                mlflow.log_metric("test_capture_top20", capture["captured_share"])

        print(f"{name:24s} CV AUC {cv_metrics['roc_auc']:.3f} | "
              f"test AUC {test_metrics['roc_auc']:.3f} "
              f"[{intervals['roc_auc']['lo']:.3f}, {intervals['roc_auc']['hi']:.3f}] | "
              f"Brier {test_metrics['brier']:.3f} | "
              f"top-20% captures {capture['captured_share']:.0%}")

    delta = paired_bootstrap_delta(
        y_test, test_probs["logistic_regression"], test_probs["hist_gradient_boosting"],
        metric="roc_auc", seed=SEED,
    )
    results["paired_delta_auc"] = delta
    print(f"Delta AUC (logistic - boosting): {delta['delta']:+.3f} "
          f"[{delta['lo']:+.3f}, {delta['hi']:+.3f}], includes zero: {delta['includes_zero']}")

    # --- decision layer -----------------------------------------------------
    chosen = "logistic_regression"
    oof_curve = expected_net_value_curve(
        oof[chosen], ECON.value_saved, ECON.contact_cost, ECON.save_rate
    )
    best = best_threshold(oof_curve)
    threshold = float(best.threshold)

    test_curve = expected_net_value_curve(
        test_probs[chosen], ECON.value_saved, ECON.contact_cost, ECON.save_rate
    )
    at_threshold = test_curve.iloc[(test_curve["threshold"] - threshold).abs().argmin()]
    oracle_curve = oracle_net_value_curve(
        y_test, test_probs[chosen], ECON.value_saved, ECON.contact_cost, ECON.save_rate
    )
    oracle_best = best_threshold(oracle_curve)
    realised_at_threshold = oracle_curve.iloc[
        (oracle_curve["threshold"] - threshold).abs().argmin()
    ]

    lo, hi = threshold_band(test_curve, tolerance=0.02)
    results["decision"] = {
        "selected_model": chosen,
        "threshold_from_oof": threshold,
        "contacted": int(at_threshold["contacted"]),
        "expected_net_value": float(at_threshold["net_value"]),
        "expected_saves": float(at_threshold["expected_saves"]),
        "realised_net_value_on_test": float(realised_at_threshold["net_value"]),
        "oracle_best_threshold": float(oracle_best["threshold"]),
        "oracle_best_net_value": float(oracle_best["net_value"]),
        "treat_all_expected_net_value": float(
            expected_net_value_curve(
                test_probs[chosen], ECON.value_saved, ECON.contact_cost, ECON.save_rate,
                thresholds=[0.0],
            )["net_value"].iloc[0]
        ),
        "threshold_band_2pct": [None if np.isnan(lo) else float(lo),
                                None if np.isnan(hi) else float(hi)],
        "capture_top20": results["models"][chosen]["capture_top20"],
    }

    # --- one-at-a-time sensitivity on the three assumptions -----------------
    # Each scenario is run under its own analytic break-even rule, which is the
    # policy a business would actually operate. This is what makes the memo's
    # "what would change this recommendation" section checkable.
    scenarios = {
        "base": ECON.as_dict(),
        "contact_cost_doubled": {**ECON.as_dict(), "contact_cost": 2 * ECON.contact_cost},
        "save_rate_halved": {**ECON.as_dict(), "save_rate": ECON.save_rate / 2},
        "value_saved_halved": {**ECON.as_dict(), "value_saved": ECON.value_saved / 2},
    }
    results["economics_sensitivity"] = {}
    for label, econ in scenarios.items():
        curve = expected_net_value_curve(
            test_probs[chosen], econ["value_saved"], econ["contact_cost"], econ["save_rate"]
        )
        rule = breakeven_probability(
            econ["contact_cost"], econ["value_saved"], econ["save_rate"]
        )
        at_rule = curve.iloc[(curve["threshold"] - rule).abs().argmin()]
        results["economics_sensitivity"][label] = {
            "breakeven_probability": rule,
            "contacted": int(at_rule["contacted"]),
            "share_contacted": float(at_rule["contacted"]) / len(test_probs[chosen]),
            "expected_net_value": float(at_rule["net_value"]),
        }
        print(f"  {label:22s} rule p >= {rule:.2f} -> "
              f"{int(at_rule['contacted']):4d} contacts, "
              f"expected {at_rule['net_value']:>8,.0f}")

    # --- persist the artefact ----------------------------------------------
    final = build_pipeline("logistic", seed=SEED).fit(df[FEATURES], df.Churn)
    path = save_bundle(
        final,
        MODEL_DIR / "churn_logistic.joblib",
        {
            "model": "logistic_regression",
            "features": FEATURES,
            "excluded_leaky": ["tenure", "TotalCharges"],
            "threshold": threshold,
            "economics": ECON.as_dict(),
            "seed": SEED,
            "rows_trained_on": int(len(df)),
            "test_metrics": results["models"][chosen]["test"],
            "test_metrics_ci": results["models"][chosen]["test_ci"],
        },
    )
    results["artefact"] = str(path.relative_to(ROOT))

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved model to {path.relative_to(ROOT)}")
    print(f"Threshold from OOF {threshold:.3f} (break-even {ECON.breakeven_probability:.3f}); "
          f"expected value {at_threshold['net_value']:,.0f}, realised on holdout "
          f"{realised_at_threshold['net_value']:,.0f}, retrospective best "
          f"{oracle_best['net_value']:,.0f}")
    print(f"Wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
