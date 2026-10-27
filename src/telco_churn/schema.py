"""Input validation for raw or scoring data.

The point is to fail loudly *before* a model silently scores a customer whose
contract type is misspelled, or a negative monthly charge. This is deliberately
small and dependency-free rather than a full data-contract library.
"""
from __future__ import annotations

import pandas as pd

ALLOWED_VALUES: dict[str, set[str]] = {
    "gender": {"Female", "Male"},
    "Partner": {"No", "Yes"},
    "Dependents": {"No", "Yes"},
    "PhoneService": {"No", "Yes"},
    "MultipleLines": {"No", "No phone service", "Yes"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"No", "No internet service", "Yes"},
    "OnlineBackup": {"No", "No internet service", "Yes"},
    "DeviceProtection": {"No", "No internet service", "Yes"},
    "TechSupport": {"No", "No internet service", "Yes"},
    "StreamingTV": {"No", "No internet service", "Yes"},
    "StreamingMovies": {"No", "No internet service", "Yes"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"No", "Yes"},
    "PaymentMethod": {
        "Bank transfer (automatic)",
        "Credit card (automatic)",
        "Electronic check",
        "Mailed check",
    },
}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
)

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 1000),
    "TotalCharges": (0, 1_000_000),
    "SeniorCitizen": (0, 1),
}

# TotalCharges is blank for brand-new customers in the raw file; that is a
# documented, expected missing value rather than a schema violation.
ALLOW_MISSING: set[str] = {"TotalCharges"}


def validate_customers(df: pd.DataFrame, strict: bool = True,
                       required: tuple[str, ...] | None = None) -> list[str]:
    """Check columns, category values and numeric ranges.

    ``required`` overrides the default column list, which is useful when
    scoring new customers: the model only needs signup-time features, not
    ``tenure`` or the churn label.

    Returns a list of human-readable problems. When ``strict`` is True and the
    list is non-empty, raises ``ValueError`` with every problem joined together.
    """
    problems: list[str] = []
    required = REQUIRED_COLUMNS if required is None else required

    missing = [c for c in required if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")

    for col, allowed in ALLOWED_VALUES.items():
        if col not in df.columns:
            continue
        unexpected = sorted(set(df[col].dropna().unique()) - allowed)
        if unexpected:
            problems.append(f"{col}: unexpected values {unexpected}")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        allow_missing = col in ALLOW_MISSING
        if values.isna().any() and not allow_missing:
            problems.append(f"{col}: {int(values.isna().sum())} non-numeric or missing values")
        bad = values.dropna().loc[(values.dropna() < lo) | (values.dropna() > hi)]
        if len(bad):
            problems.append(f"{col}: {len(bad)} values outside [{lo}, {hi}]")

    if "Churn" in df.columns:
        churn = set(df["Churn"].dropna().unique())
        if not churn <= {"Yes", "No", 0, 1}:
            problems.append(f"Churn: unexpected values {sorted(churn)}")

    if strict and problems:
        raise ValueError("input validation failed:\n- " + "\n- ".join(problems))
    return problems
