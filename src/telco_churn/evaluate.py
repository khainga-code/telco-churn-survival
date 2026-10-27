"""Business-aligned evaluation: calibration, uncertainty, capture rates and value.

Two different value curves live here and the distinction matters:

* ``oracle_net_value_curve`` uses the *true* churn label of every contacted
  customer. It answers "what would we have earned if we knew who would churn?"
  and is therefore an upper bound, not a forecast.
* ``expected_net_value_curve`` uses only the model's predicted probability and
  the stated economics. It is what a campaign can actually expect.

Reporting only the first is a common and flattering mistake; both are returned
so the gap is visible.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

_METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "roc_auc": lambda y, p: roc_auc_score(y, p),
    "pr_auc": lambda y, p: average_precision_score(y, p),
    "brier": lambda y, p: brier_score_loss(y, p),
    "log_loss": lambda y, p: log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1]),
}


def _safe_metric(name: str, y: np.ndarray, p: np.ndarray) -> float:
    """Return NaN instead of raising when a resample has one class."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(_METRIC_FUNCS[name](y, p))
    except ValueError:
        return float("nan")


def summary_metrics(y_true, p) -> dict:
    """ROC-AUC, PR-AUC, Brier and log loss.

    A single-class slice makes ROC-AUC/PR-AUC undefined; those are reported as
    NaN rather than crashing a report.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    return {name: _safe_metric(name, y, p) for name in _METRIC_FUNCS}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------


def bootstrap_ci(y_true, p, metric: str = "roc_auc", n_boot: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Percentile bootstrap interval for one metric.

    Resamples rows with replacement. Resamples that contain a single outcome
    class are skipped, because the metric is undefined there.
    """
    if metric not in _METRIC_FUNCS:
        raise ValueError(f"unknown metric {metric!r}; choose from {sorted(_METRIC_FUNCS)}")
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        value = _safe_metric(metric, y[idx], p[idx])
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return {"estimate": _safe_metric(metric, y, p), "lo": np.nan, "hi": np.nan, "n_boot": 0}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": _safe_metric(metric, y, p), "lo": float(lo), "hi": float(hi),
            "n_boot": len(draws)}


def paired_bootstrap_delta(y_true, p_a, p_b, metric: str = "roc_auc",
                           n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Paired bootstrap for ``metric(p_a) - metric(p_b)`` on the same rows.

    Pairing matters: the two models are scored on identical customers, so the
    uncertainty in the *difference* is much smaller than either model's own
    interval. If the interval contains zero, the models are not distinguishable.
    """
    y = np.asarray(y_true).astype(int)
    a = np.asarray(p_a, dtype=float)
    b = np.asarray(p_b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        ma = _safe_metric(metric, y[idx], a[idx])
        mb = _safe_metric(metric, y[idx], b[idx])
        if np.isfinite(ma) and np.isfinite(mb):
            deltas.append(ma - mb)
    observed = _safe_metric(metric, y, a) - _safe_metric(metric, y, b)
    if not deltas:
        return {"delta": observed, "lo": np.nan, "hi": np.nan,
                "includes_zero": True, "n_boot": 0}
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"delta": float(observed), "lo": float(lo), "hi": float(hi),
            "includes_zero": bool(lo <= 0 <= hi), "n_boot": len(deltas)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_table(y_true, p, bins: int = 10) -> pd.DataFrame:
    """Mean predicted vs observed churn rate per equal-count probability bin."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], q=bins, duplicates="drop")
    out = d.groupby("bin", observed=True).agg(
        n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean")
    )
    return out.reset_index(drop=True)


def calibration_slope_intercept(y_true, p) -> dict:
    """Logistic recalibration of the predictions.

    Perfect calibration gives slope 1 and intercept 0. Slope < 1 means the
    model is over-confident (predictions too extreme); slope > 1 means
    under-confident.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if y.min() == y.max():
        return {"slope": np.nan, "intercept": np.nan}
    lr = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, y)
    return {"slope": float(lr.coef_[0, 0]), "intercept": float(lr.intercept_[0])}


# ---------------------------------------------------------------------------
# Capture and value
# ---------------------------------------------------------------------------


def capture_at_top(y_true, p, top_frac: float = 0.2) -> dict:
    """Share of all churners found in the top ``top_frac`` of customers by risk."""
    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d = d.sort_values("p", ascending=False)
    k = max(1, int(round(len(d) * top_frac)))
    captured = d["y"].iloc[:k].sum() / d["y"].sum()
    precision = d["y"].iloc[:k].mean()
    return {"top_frac": top_frac, "captured_share": float(captured),
            "precision": float(precision), "base_rate": float(d["y"].mean())}


def bootstrap_capture_at_top(y_true, p, top_frac: float = 0.2, n_boot: int = 500,
                             alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap interval for capture-at-top, which is a ratio of counts."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        draws.append(capture_at_top(y[idx], p[idx], top_frac)["captured_share"])
    base = capture_at_top(y, p, top_frac)
    if not draws:
        return {**base, "lo": np.nan, "hi": np.nan}
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {**base, "lo": float(lo), "hi": float(hi)}


def oracle_net_value_curve(y_true, p, value_saved: float, contact_cost: float,
                           save_rate: float,
                           thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Net value using the true churn label of each contacted customer.

    This is a ceiling: only the true churners can be "saved". It is useful for
    comparing ranking quality, not for forecasting campaign returns.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        contacted = p >= t
        saves = y[contacted].sum()
        net = saves * save_rate * value_saved - contacted.sum() * contact_cost
        rows.append({"threshold": float(t), "contacted": int(contacted.sum()),
                     "expected_saves": float(saves * save_rate), "net_value": float(net)})
    return pd.DataFrame(rows)


def expected_net_value_curve(p, value_saved: float, contact_cost: float, save_rate: float,
                             thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Net value implied by the model's own probabilities.

    Per contacted customer the expected margin is
    ``p * save_rate * value_saved - contact_cost``. This is what a campaign
    run on the model can expect, conditional on the model being calibrated and
    the economics being right.
    """
    p = np.asarray(p, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)
    rows = []
    for t in thresholds:
        contacted = p >= t
        ev = contacted * (p * save_rate * value_saved - contact_cost)
        rows.append({"threshold": float(t), "contacted": int(contacted.sum()),
                     "expected_saves": float((p[contacted] * save_rate).sum()),
                     "net_value": float(ev.sum())})
    return pd.DataFrame(rows)


def breakeven_probability(contact_cost: float, value_saved: float, save_rate: float) -> float:
    """Probability above which contacting a customer has positive expected value."""
    return float(contact_cost / (save_rate * value_saved))


def best_threshold(curve: pd.DataFrame) -> pd.Series:
    return curve.loc[curve["net_value"].idxmax()]


def threshold_band(curve: pd.DataFrame, tolerance: float = 0.02) -> tuple[float, float]:
    """Range of thresholds whose value is within ``tolerance`` of the maximum.

    A wide band means the exact cut-off is not a decision that deserves
    precision; a narrow one means it does.
    """
    peak = curve["net_value"].max()
    if peak <= 0:
        return (np.nan, np.nan)
    ok = curve.loc[curve["net_value"] >= peak * (1 - tolerance), "threshold"]
    return (float(ok.min()), float(ok.max()))


# ---------------------------------------------------------------------------
# Decision-curve analysis
# ---------------------------------------------------------------------------


def decision_curve(y_true, p, thresholds: Iterable[float] | None = None) -> pd.DataFrame:
    """Net benefit of using the model, treat-all and treat-none at each threshold.

    ``net_benefit = TP/n - FP/n * pt/(1-pt)``. This is the standard way to show
    that a model adds value over a blanket policy without committing to one
    cost ratio.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y)
    prevalence = y.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.75, 75)
    rows = []
    for pt in thresholds:
        predicted = p >= pt
        tp = int((predicted & (y == 1)).sum())
        fp = int((predicted & (y == 0)).sum())
        weight = pt / (1 - pt)
        model_nb = tp / n - (fp / n) * weight
        all_nb = prevalence - (1 - prevalence) * weight
        rows.append({"threshold": float(pt), "model": float(model_nb),
                     "treat_all": float(all_nb), "treat_none": 0.0})
    return pd.DataFrame(rows)


# Backwards-compatible alias: earlier versions called the oracle curve
# ``net_value_curve``. New code should use the explicit name.
def net_value_curve(*args, **kwargs) -> pd.DataFrame:  # pragma: no cover - alias
    return oracle_net_value_curve(*args, **kwargs)
