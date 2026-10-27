"""Data loading and cleaning for the IBM Telco Customer Churn dataset."""
from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

import pandas as pd

from telco_churn.schema import validate_customers

URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "telco_churn.csv"

# Hash of the file this project was built against. If the upstream sample is
# ever replaced, the download fails loudly instead of silently changing results.
EXPECTED_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def download(path: str | Path = DEFAULT_PATH, url: str = URL,
             check_hash: bool = True, timeout: int = 60) -> Path:
    """Download the raw CSV if it is not already on disk, then verify its hash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(url, timeout=timeout) as response, open(path, "wb") as out:
            shutil.copyfileobj(response, out)
    if check_hash and EXPECTED_SHA256:
        actual = sha256(path)
        if actual != EXPECTED_SHA256:
            raise ValueError(
                f"downloaded file hash {actual} does not match the pinned "
                f"{EXPECTED_SHA256}; the upstream sample may have changed"
            )
    return path


def load_clean(path: str | Path = DEFAULT_PATH, validate: bool = True) -> pd.DataFrame:
    """Load the CSV and apply documented cleaning steps.

    Cleaning decisions (kept explicit so they can be reviewed):
    - TotalCharges is blank for customers with tenure 0 (new, not yet billed);
      these are set to 0 rather than dropped.
    - Churn is converted to a 0/1 integer target.
    - customerID is dropped (identifier, not a feature).

    ``validate=True`` checks every categorical value and numeric range against
    the documented schema before any modelling happens.
    """
    df = pd.read_csv(download(path))
    if validate:
        validate_customers(df)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_blank = int(df["TotalCharges"].isna().sum())
    df.loc[df["TotalCharges"].isna(), "TotalCharges"] = 0.0
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["customerID"], errors="ignore")
    df.attrs["blank_total_charges"] = n_blank
    return df
