"""Feature definitions and preprocessing.

Leakage note
------------
`tenure` and `TotalCharges` are only known *after* a customer has stayed for a
while, and for churned customers `tenure` is simply how long they lasted. A model
that uses them answers "who has already behaved like a churner?", not "who
should we act on early?". The default feature set therefore uses only
signup-time attributes (plan, contract, price, payment method, demographics).
Time-to-event information is used properly in the survival analysis instead.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
LEAKY = ["tenure", "TotalCharges"]
NUMERIC = ["MonthlyCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def assert_no_leakage(columns) -> None:
    """Raise if any post-outcome column slipped into a feature matrix."""
    leaked = [c for c in LEAKY if c in set(columns)]
    if leaked:
        raise ValueError(
            f"leaky feature(s) {leaked} are only known after a customer has stayed; "
            "use them for survival analysis, not churn classification"
        )


def split_xy(df: pd.DataFrame, include_leaky: bool = False):
    cols = NUMERIC + CATEGORICAL + (LEAKY if include_leaky else [])
    if not include_leaky:
        assert_no_leakage(cols)
    return df[cols].copy(), df[TARGET].copy()


def make_preprocessor(include_leaky: bool = False) -> ColumnTransformer:
    numeric = NUMERIC + (LEAKY if include_leaky else [])
    categorical = [c for c in CATEGORICAL]
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), categorical),
        ]
    )


def train_test(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42,
               include_leaky: bool = False):
    X, y = split_xy(df, include_leaky)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
