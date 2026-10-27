"""Turn a churn score into a capacity-constrained contact list.

A risk ranking answers "who is most likely to leave?". A retention team needs
"whom do we call first, and what is that worth?". Those are different
questions: a high-risk customer on a cheap plan may be worth less than a
slightly lower-risk customer on an expensive one. Everything here is expressed
in expected margin so the list can be compared with its own cost.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_value(p, margin, save_rate: float, contact_cost: float):
    """Expected margin from contacting one customer.

    ``margin`` is the value at risk if the customer leaves (revenue or gross
    margin over the relevant horizon), not the monthly charge itself unless
    that is the business's definition.
    """
    p = np.asarray(p, dtype=float)
    margin = np.asarray(margin, dtype=float)
    return p * margin * save_rate - contact_cost


def rank_by_value(df: pd.DataFrame, p, margin, save_rate: float = 0.3,
                  contact_cost: float = 10.0) -> pd.DataFrame:
    """Return a copy of ``df`` ranked by expected value (descending).

    ``margin`` is the value at risk for each customer (array-like, same order
    as ``df``). Columns added: ``p_churn``, ``expected_value``, ``rank``.
    Rows are sorted and the index is reset so downstream selection is positional.
    """
    out = df.reset_index(drop=True).copy()
    p = np.asarray(p, dtype=float)
    margin = np.asarray(margin, dtype=float)
    if len(p) != len(out):
        raise ValueError(f"got {len(p)} scores for {len(out)} rows")
    if len(margin) != len(out):
        raise ValueError(f"got {len(margin)} margins for {len(out)} rows")
    out["p_churn"] = p
    out["expected_value"] = expected_value(p, margin, save_rate, contact_cost)
    out = out.sort_values(["expected_value", "p_churn"], ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out


def select_with_capacity(ranked: pd.DataFrame, capacity: int,
                         min_expected_value: float = 0.0) -> pd.DataFrame:
    """Top ``capacity`` customers, dropping anyone with non-positive EV.

    ``min_expected_value=0`` is the break-even rule: never contact a customer
    whose expected margin does not cover the cost of the contact.
    """
    if capacity < 0:
        raise ValueError("capacity must be non-negative")
    eligible = ranked.loc[ranked["expected_value"] > min_expected_value]
    return eligible.head(int(capacity)).copy()


def segment_report(queue: pd.DataFrame,
                   by: tuple[str, ...] = ("Contract", "InternetService")) -> pd.DataFrame:
    """Composition of a contact list, for capacity and fairness reviews."""
    by = [c for c in by if c in queue.columns]
    if not by:
        raise ValueError("none of the grouping columns are present")
    return (
        queue.groupby(by, observed=True)
        .agg(customers=("p_churn", "size"),
             mean_risk=("p_churn", "mean"),
             expected_value=("expected_value", "sum"))
        .sort_values("expected_value", ascending=False)
        .reset_index()
    )


def compare_risk_vs_value(ranked: pd.DataFrame, k: int) -> dict:
    """Compare a risk-ranked top-k with a value-ranked top-k.

    ``ranked`` must come from :func:`rank_by_value` (already value-sorted).
    The risk ranking is recomputed from ``p_churn``.
    """
    k = min(int(k), len(ranked))
    value_top = ranked.head(k)
    risk_top = ranked.sort_values("p_churn", ascending=False).head(k)
    overlap = len(set(value_top.index) & set(risk_top.index))
    return {
        "k": k,
        "overlap": overlap,
        "overlap_share": overlap / k if k else float("nan"),
        "value_selected": float(value_top["expected_value"].sum()),
        "value_risk_ranked": float(risk_top["expected_value"].sum()),
        "value_gap": float(value_top["expected_value"].sum() - risk_top["expected_value"].sum()),
        "mean_risk_value_ranked": float(value_top["p_churn"].mean()),
        "mean_risk_risk_ranked": float(risk_top["p_churn"].mean()),
    }
