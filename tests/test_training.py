"""Model training tests."""

from pathlib import Path

import joblib

from crank.config import ScoringConfig
from crank.model.scorer import ClusterScorer
from crank.training import train_from_dataset


def test_train_writes_model(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    train_from_dataset(dataset, out)
    assert out.exists()
    payload = joblib.load(out)
    assert "regressor" in payload
    assert "anomaly" in payload


def test_trained_model_changes_scores(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    train_from_dataset(dataset, out)
    scorer = ClusterScorer(ScoringConfig(model_path=out))
    assert scorer.has_trained_model
