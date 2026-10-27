"""Fit, persist and reload the churn model with its decision metadata.

A model file on its own is not deployable: you also need the feature list, the
contact threshold, the economics it was tuned under, and the library versions
used to produce it. Those travel next to the artefact as a JSON sidecar.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from telco_churn.features import make_preprocessor

MODEL_DIR = Path(__file__).resolve().parents[2] / "models"


def build_pipeline(kind: str = "logistic", seed: int = 42) -> Pipeline:
    """Return an unfitted preprocessing + classifier pipeline."""
    if kind == "logistic":
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    elif kind in {"boosting", "hist_gradient_boosting"}:
        model = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, random_state=seed
        )
    else:
        raise ValueError(f"unknown model kind {kind!r}")
    return Pipeline([("prep", make_preprocessor()), ("model", model)])


def save_bundle(pipe: Pipeline, path: str | Path, metadata: dict[str, Any]) -> Path:
    """Persist the fitted pipeline and a ``.json`` sidecar of decision metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        **metadata,
    }
    joblib.dump(pipe, path)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def load_bundle(path: str | Path) -> tuple[Pipeline, dict[str, Any]]:
    """Load a pipeline saved by :func:`save_bundle` and its metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run `python scripts/train_baselines.py` first"
        )
    pipe = joblib.load(path)
    meta_path = path.with_suffix(".json")
    metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return pipe, metadata
