import numpy as np
import pandas as pd

from telco_churn.drift import drift_report, drift_trajectory, psi, should_retrain


def test_psi_identical_is_near_zero():
    s = pd.Series(np.random.default_rng(0).normal(size=2000))
    assert psi(s, s) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi(pd.Series(rng.normal(0, 1, 2000)), pd.Series(rng.normal(2, 1, 2000))) > 0.2


def test_psi_constant_reference_is_zero():
    assert psi(pd.Series(np.ones(100)), pd.Series(np.ones(100))) == 0.0


def test_psi_unseen_category_moves_the_index():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    new = pd.Series(["a"] * 50 + ["c"] * 50)
    assert psi(ref, new) > 0.1


def test_drift_report_columns():
    a = pd.DataFrame({"x": np.arange(100), "c": ["u", "v"] * 50})
    out = drift_report(a, a)
    assert set(out.columns) == {"feature", "psi", "flag"}
    assert not out["flag"].any()


def test_drift_trajectory_summarises_periods():
    ref = pd.DataFrame({"x": np.arange(100), "c": ["u", "v"] * 50})
    snapshots = {"month_1": ref.copy(), "month_2": ref.assign(x=ref.x + 500)}
    out = drift_trajectory(ref, snapshots)
    assert list(out["period"]) == ["month_1", "month_2"]
    assert out.loc[0, "alert"] is False or not out.loc[0, "alert"]
    assert bool(out.loc[1, "alert"]) is True


def test_retrain_trigger_on_drift():
    rep = pd.DataFrame({"feature": list("abc"), "psi": [0.5, 0.4, 0.3], "flag": [True, True, True]})
    assert should_retrain(rep, 0.15, 0.15)[0] is True
    assert should_retrain(rep.assign(flag=False), 0.15, 0.15)[0] is False


def test_retrain_trigger_on_brier_regression():
    rep = pd.DataFrame({"feature": ["a"], "psi": [0.01], "flag": [False]})
    fired, reason = should_retrain(rep, brier_now=0.18, brier_ref=0.15)
    assert fired and "Brier" in reason
