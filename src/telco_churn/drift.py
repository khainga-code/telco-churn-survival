"""Simple data-drift monitoring with an explicit retraining trigger."""
from __future__ import annotations

import numpy as np
import pandas as pd

# PSI rule of thumb: <0.1 stable, 0.1-0.2 watch, >0.2 significant shift.
PSI_WATCH = 0.1
PSI_ALERT = 0.2


def psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between a reference and a new sample.

    Numeric inputs use reference quantile bins; categorical inputs use category
    shares. Unseen categories in ``actual`` are counted rather than dropped.
    """
    if len(expected) == 0 or len(actual) == 0:
        raise ValueError("psi requires two non-empty samples")
    eps = 1e-6
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.quantile(expected.dropna(), np.linspace(0, 1, bins + 1)))
        if len(edges) < 2:
            return 0.0  # constant reference: no quantile structure to compare
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(expected.dropna(), edges)[0] / len(expected)
        a = np.histogram(actual.dropna(), edges)[0] / len(actual)
    else:
        cats = sorted(set(expected.unique()) | set(actual.unique()))
        e = expected.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        a = actual.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def drift_report(reference: pd.DataFrame, current: pd.DataFrame,
                 psi_threshold: float = PSI_ALERT) -> pd.DataFrame:
    """PSI per shared feature, worst first, with a boolean flag."""
    features = [c for c in reference.columns if c in current.columns]
    rows = [{"feature": c, "psi": psi(reference[c], current[c])} for c in features]
    out = pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
    out["flag"] = out["psi"] > psi_threshold
    return out


def drift_trajectory(reference: pd.DataFrame, snapshots: dict[str, pd.DataFrame],
                     psi_threshold: float = PSI_ALERT) -> pd.DataFrame:
    """Summarise drift period by period.

    ``snapshots`` maps a period label to the feature frame observed then. The
    output is what a monitoring dashboard would show: how many features moved,
    the worst offender, and whether the alert line was crossed.
    """
    rows = []
    for period, frame in snapshots.items():
        report = drift_report(reference, frame, psi_threshold=psi_threshold)
        worst = report.iloc[0]
        rows.append({
            "period": period,
            "features_flagged": int(report["flag"].sum()),
            "max_psi": float(worst["psi"]),
            "worst_feature": worst["feature"],
            "alert": bool(report["flag"].any()),
            "watch": bool((report["psi"] > PSI_WATCH).any()),
        })
    return pd.DataFrame(rows)


def should_retrain(report: pd.DataFrame, brier_now: float, brier_ref: float,
                   max_flagged: int = 0, max_brier_increase: float = 0.02) -> tuple[bool, str]:
    """Retraining trigger: owner is the model's data scientist; review within one week.

    Trigger a retrain review if (a) more than ``max_flagged`` features have PSI
    above threshold (default: any single feature), or (b) Brier score on recent
    labelled data worsened by more than ``max_brier_increase``. Thresholds are
    starting points to agree with the business, not universal constants.
    """
    n_flagged = int(report["flag"].sum())
    if n_flagged > max_flagged:
        return True, f"{n_flagged} features drifted (PSI above threshold)"
    if brier_now - brier_ref > max_brier_increase:
        return True, f"Brier worsened by {brier_now - brier_ref:.3f}"
    return False, "no trigger met"
