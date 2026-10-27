import pandas as pd
import pytest

from telco_churn.economics import Economics
from telco_churn.targeting import (
    compare_risk_vs_value,
    expected_value,
    rank_by_value,
    segment_report,
    select_with_capacity,
)


def frame():
    return pd.DataFrame({
        "MonthlyCharges": [100.0, 20.0, 80.0, 10.0],
        "Contract": ["Month-to-month", "Two year", "Month-to-month", "Two year"],
        "InternetService": ["Fiber optic", "DSL", "DSL", "No"],
    })


def margin(df):
    return df["MonthlyCharges"].to_numpy()


def test_expected_value_formula():
    assert expected_value(0.5, 100, 0.3, 10) == pytest.approx(0.5 * 100 * 0.3 - 10)


def test_rank_by_value_sorts_and_ranks():
    df = frame()
    ranked = rank_by_value(df, [0.9, 0.1, 0.5, 0.2], margin(df), save_rate=0.3, contact_cost=10)
    assert ranked["expected_value"].is_monotonic_decreasing
    assert ranked["rank"].tolist() == [1, 2, 3, 4]


def test_rank_by_value_rejects_length_mismatch():
    df = frame()
    with pytest.raises(ValueError):
        rank_by_value(df, [0.5, 0.5], margin(df))
    with pytest.raises(ValueError):
        rank_by_value(df, [0.5] * 4, margin(df)[:2])


def test_capacity_selection_drops_negative_ev():
    df = frame()
    ranked = rank_by_value(df, [0.1] * 4, margin(df), save_rate=0.3, contact_cost=10)
    queue = select_with_capacity(ranked, capacity=3)
    assert len(queue) <= 3
    assert (queue["expected_value"] > 0).all()


def test_value_ranking_beats_risk_ranking_on_value():
    # the second customer has lower risk but far more revenue at stake
    df = frame()
    p = [0.50, 0.45, 0.40, 0.35]
    ranked = rank_by_value(df, p, margin(df), save_rate=0.3, contact_cost=1)
    out = compare_risk_vs_value(ranked, k=2)
    assert out["value_selected"] >= out["value_risk_ranked"]
    assert 0 <= out["overlap"] <= 2


def test_segment_report_groups():
    df = frame()
    ranked = rank_by_value(df, [0.9, 0.2, 0.4, 0.1], margin(df),
                           save_rate=0.3, contact_cost=10)
    out = segment_report(ranked)
    assert out["customers"].sum() == 4
    assert set(out.columns) == {"Contract", "InternetService", "customers", "mean_risk",
                                "expected_value"}


def test_segment_report_requires_a_known_column():
    with pytest.raises(ValueError):
        segment_report(pd.DataFrame({"x": [1]}), by=("nope",))


def test_relative_value_preserves_average_scale():
    econ = Economics(value_saved=300)
    charges = pd.Series([40.0, 80.0])
    values = econ.relative_value(charges)
    assert values.mean() == pytest.approx(300)
    assert values[1] > values[0]
