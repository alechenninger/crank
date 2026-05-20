"""Model training tests."""

from pathlib import Path

import joblib
import pytest

from crank.config import ScoringConfig
from crank.model.scorer import MIN_TRAINING_SAMPLES, ClusterScorer
from crank.training import train_from_dataset


def test_train_writes_model(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    metrics = train_from_dataset(dataset, out)
    assert out.exists()
    payload = joblib.load(out)
    assert "regressor" in payload
    assert "anomaly" in payload
    assert payload["feature_names"] is not None
    assert "metrics" in payload
    assert "mae" in metrics


def test_trained_model_changes_scores(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    train_from_dataset(dataset, out)
    scorer = ClusterScorer(ScoringConfig(model_path=out))
    assert scorer.has_trained_model
    assert scorer.scoring_mode().value == "blended"


def test_train_requires_minimum_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"name": "a", "label": 10, "nodes": {"total": 1, "ready": 1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=str(MIN_TRAINING_SAMPLES)):
        train_from_dataset(dataset, tmp_path / "m.joblib")
