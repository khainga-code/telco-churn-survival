import numpy as np
import pandas as pd
import pytest

from telco_churn.evaluate import (
    best_threshold,
    bootstrap_ci,
    breakeven_probability,
    calibration_slope_intercept,
    capture_at_top,
    decision_curve,
    expected_net_value_curve,
    oracle_net_value_curve,
    paired_bootstrap_delta,
    summary_metrics,
    threshold_band,
)


def test_summary_metrics_has_all_keys():
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.1, 0.8, 0.7, 0.2, 0.9, 0.3])
    assert set(summary_metrics(y, p)) == {"roc_auc", "pr_auc", "brier", "log_loss"}


def test_summary_metrics_single_class_returns_nan_not_crash():
    out = summary_metrics(np.zeros(10), np.linspace(0, 1, 10))
    assert np.isnan(out["roc_auc"])


def test_bootstrap_interval_brackets_estimate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 500)
    p = np.clip(0.3 + 0.2 * (y - 0.3) + rng.normal(0, 0.1, 500), 0, 1)
    ci = bootstrap_ci(y, p, "roc_auc", n_boot=200)
    assert ci["lo"] <= ci["estimate"] <= ci["hi"]


def test_paired_bootstrap_identical_models_is_zero():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.4, 300)
    p = rng.random(300)
    out = paired_bootstrap_delta(y, p, p, "roc_auc", n_boot=100)
    assert out["delta"] == pytest.approx(0.0)
    assert out["includes_zero"]


def test_capture_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert capture_at_top(y, p, 0.2)["captured_share"] == 1.0


def test_oracle_curve_uses_labels_but_expected_curve_does_not():
    p = np.array([0.5, 0.5, 0.5, 0.5])
    y_a = np.array([1, 1, 0, 0])
    y_b = np.array([0, 0, 0, 1])
    kwargs = dict(value_saved=100, contact_cost=10, save_rate=0.5, thresholds=[0.4])
    oracle_a = oracle_net_value_curve(y_a, p, **kwargs)["net_value"].iloc[0]
    oracle_b = oracle_net_value_curve(y_b, p, **kwargs)["net_value"].iloc[0]
    expected_a = expected_net_value_curve(p, 100, 10, 0.5, [0.4])["net_value"].iloc[0]
    expected_b = expected_net_value_curve(p, 100, 10, 0.5, [0.4])["net_value"].iloc[0]
    assert oracle_a != oracle_b           # labels matter for the oracle
    assert expected_a == expected_b       # only probabilities matter for the forecast
    assert expected_a == pytest.approx(4 * (0.5 * 0.5 * 100 - 10))


def test_best_threshold_is_break_even_for_expected_curve():
    econ = dict(value_saved=300, contact_cost=10, save_rate=0.3)
    p = np.linspace(0.01, 0.99, 500)
    curve = expected_net_value_curve(p, **econ)
    assert best_threshold(curve)["threshold"] == pytest.approx(
        breakeven_probability(10, 300, 0.3), abs=0.02
    )


def test_threshold_band_contains_the_optimum():
    curve = pd.DataFrame({"threshold": np.linspace(0, 1, 101),
                          "net_value": 100 - (np.linspace(0, 1, 101) - 0.5) ** 2 * 100})
    lo, hi = threshold_band(curve, tolerance=0.05)
    assert lo <= 0.5 <= hi


def test_decision_curve_no_treatment_is_zero():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.8, 0.3, 0.7])
    dc = decision_curve(y, p, thresholds=[0.5])
    assert (dc["treat_none"] == 0).all()


def test_calibration_slope_keys():
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.3, 200)
    p = np.clip(0.3 + 0.1 * (y - 0.3) + rng.normal(0, 0.05, 200), 0.01, 0.99)
    out = calibration_slope_intercept(y, p)
    assert set(out) == {"slope", "intercept"}
    assert np.isfinite(out["slope"])
