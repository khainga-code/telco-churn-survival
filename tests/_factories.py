"""Small synthetic frames shared by the tests."""
from __future__ import annotations

import numpy as np
import pandas as pd


def tiny_frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "gender": rng.choice(["Female", "Male"], n),
        "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["No", "Yes"], n),
        "Dependents": rng.choice(["No", "Yes"], n),
        "tenure": rng.integers(1, 60, n),
        "PhoneService": rng.choice(["No", "Yes"], n),
        "MultipleLines": rng.choice(["No", "Yes"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["No", "Yes"], n),
        "OnlineBackup": rng.choice(["No", "Yes"], n),
        "DeviceProtection": rng.choice(["No", "Yes"], n),
        "TechSupport": rng.choice(["No", "Yes"], n),
        "StreamingTV": rng.choice(["No", "Yes"], n),
        "StreamingMovies": rng.choice(["No", "Yes"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["No", "Yes"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": rng.uniform(20, 100, n),
        "TotalCharges": rng.uniform(20, 5000, n),
        "Churn": rng.integers(0, 2, n),
    })
