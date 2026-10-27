"""Scoring pipeline: raw customer frame in, ranked contact queue out."""
from __future__ import annotations

import pandas as pd

from telco_churn.economics import Economics
from telco_churn.features import FEATURES
from telco_churn.schema import validate_customers
from telco_churn.targeting import rank_by_value, select_with_capacity


def build_queue(df: pd.DataFrame, pipe, econ: Economics | None = None,
                capacity: int = 500, margin_col: str = "MonthlyCharges") -> pd.DataFrame:
    """Validate, score and rank customers, returning at most ``capacity`` rows.

    Raises ``ValueError`` on a malformed input frame rather than scoring it.
    """
    econ = econ or Economics()
    validate_customers(df, required=tuple(FEATURES))
    p = pipe.predict_proba(df[list(FEATURES)])[:, 1]
    margin = econ.relative_value(df[margin_col])
    ranked = rank_by_value(df, p, margin,
                           save_rate=econ.save_rate, contact_cost=econ.contact_cost)
    return select_with_capacity(ranked, capacity)
