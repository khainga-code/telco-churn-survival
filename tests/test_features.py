import numpy as np
import pytest
from _factories import tiny_frame

from telco_churn.features import (
    FEATURES,
    LEAKY,
    assert_no_leakage,
    make_preprocessor,
    split_xy,
)
from telco_churn.schema import validate_customers


def test_leaky_columns_are_never_features():
    X, _ = split_xy(tiny_frame())
    assert not set(LEAKY) & set(X.columns)
    assert list(X.columns) == FEATURES


def test_assert_no_leakage_raises():
    with pytest.raises(ValueError, match="tenure"):
        assert_no_leakage(["MonthlyCharges", "tenure"])


def test_leaky_features_are_opt_in():
    X, _ = split_xy(tiny_frame(), include_leaky=True)
    assert set(LEAKY) <= set(X.columns)


def test_preprocessor_handles_unseen_category():
    df = tiny_frame()
    pre = make_preprocessor().fit(df[FEATURES])
    new = df[FEATURES].head(3).copy()
    new.loc[new.index[0], "Contract"] = "Three year"  # not in training data
    out = pre.transform(new)
    assert out.shape[0] == 3
    assert np.isfinite(out).all()


def test_single_row_transform_is_2d():
    df = tiny_frame()
    pre = make_preprocessor().fit(df[FEATURES])
    assert pre.transform(df[FEATURES].head(1)).ndim == 2


def test_schema_accepts_tiny_frame():
    assert validate_customers(tiny_frame()) == []


def test_schema_rejects_missing_column():
    with pytest.raises(ValueError, match="missing columns"):
        validate_customers(tiny_frame().drop(columns=["Contract"]))


def test_schema_rejects_unknown_category():
    df = tiny_frame()
    df.loc[0, "Contract"] = "Three year"
    with pytest.raises(ValueError, match="Contract"):
        validate_customers(df)


def test_schema_rejects_negative_price():
    df = tiny_frame()
    df.loc[0, "MonthlyCharges"] = -5
    with pytest.raises(ValueError, match="MonthlyCharges"):
        validate_customers(df)


def test_schema_allows_blank_total_charges():
    df = tiny_frame()
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[0, "TotalCharges"] = " "
    assert validate_customers(df) == []


def test_schema_can_validate_scoring_subset():
    df = tiny_frame()
    assert validate_customers(df[FEATURES], required=tuple(FEATURES)) == []
