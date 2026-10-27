import numpy as np
import pytest
from _factories import tiny_frame

from telco_churn.economics import Economics
from telco_churn.features import FEATURES
from telco_churn.model import build_pipeline, load_bundle, save_bundle
from telco_churn.scoring import build_queue


def test_pipeline_fits_and_predicts_probabilities():
    df = tiny_frame(60)
    pipe = build_pipeline("logistic", seed=1).fit(df[FEATURES], df["Churn"])
    p = pipe.predict_proba(df[FEATURES])[:, 1]
    assert len(p) == len(df)
    assert ((p >= 0) & (p <= 1)).all()


def test_unknown_model_kind_rejected():
    with pytest.raises(ValueError):
        build_pipeline("magic")


def test_save_and_load_roundtrip(tmp_path):
    df = tiny_frame(60)
    pipe = build_pipeline("logistic", seed=1).fit(df[FEATURES], df["Churn"])
    path = save_bundle(pipe, tmp_path / "m.joblib", {"threshold": 0.11, "owner": "test"})
    loaded, meta = load_bundle(path)
    assert meta["threshold"] == 0.11
    assert meta["owner"] == "test"
    assert "sklearn_version" in meta
    np.testing.assert_allclose(
        loaded.predict_proba(df[FEATURES])[:, 1],
        pipe.predict_proba(df[FEATURES])[:, 1],
    )


def test_load_missing_model_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="train_baselines"):
        load_bundle(tmp_path / "absent.joblib")


def test_build_queue_respects_capacity():
    df = tiny_frame(80)
    pipe = build_pipeline("logistic", seed=1).fit(df[FEATURES], df["Churn"])
    queue = build_queue(df, pipe, Economics(), capacity=10)
    assert len(queue) <= 10
    assert set(["p_churn", "expected_value", "rank"]).issubset(queue.columns)
    assert queue["rank"].tolist() == list(range(1, len(queue) + 1))

