"""Model training tests."""

from pathlib import Path

import joblib
import numpy as np
import pytest

from crank.model.scorer import ClusterScorer
from crank.training import MIN_TRAINING_SAMPLES
from crank.training import train_from_dataset


def test_train_writes_model(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    metrics = train_from_dataset(dataset, out)
    assert out.exists()
    payload = joblib.load(out)
    assert "coef" in payload
    assert "regressor" not in payload
    assert "calibration_min" in payload
    assert "calibration_max" in payload
    assert isinstance(payload["coef"], np.ndarray)
    assert payload["feature_names"] is not None
    assert "metrics" in payload
    assert "pairwise_accuracy" in metrics
    assert "kendall_tau" in metrics
    assert "n_pairs" in metrics


def test_trained_model_uses_ml_mode(trained_model_path: Path) -> None:
    scorer = ClusterScorer(model_path=trained_model_path)
    assert scorer.has_trained_model
    assert scorer.scoring_mode.value == "ml"


def test_train_requires_minimum_samples(tmp_path: Path) -> None:
    dataset = tmp_path / "tiny.jsonl"
    dataset.write_text(
        '{"session": "s", "rank": 1, "name": "a", "nodes": {"total": 1, "ready": 1}}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=str(MIN_TRAINING_SAMPLES)):
        train_from_dataset(dataset, tmp_path / "m.joblib")


def test_trained_model_schema_version(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    train_from_dataset(dataset, out)
    payload = joblib.load(out)
    assert payload["schema_version"] == 3


def test_session_parsing_groups_correctly(tmp_path: Path) -> None:
    """Verify session/rank fields are parsed and result in valid pairwise metrics."""
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    out = tmp_path / "model.joblib"
    metrics = train_from_dataset(dataset, out)
    # 3 sessions: C(4,2)=6 + C(3,2)=3 + C(3,2)=3 = 12 ordered pairs, x2 for both directions
    assert metrics["n_pairs"] == 24
    assert 0.0 <= metrics["pairwise_accuracy"] <= 1.0
    assert -1.0 <= metrics["kendall_tau"] <= 1.0


def test_train_rejects_missing_session(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    lines = []
    for i in range(10):
        lines.append(f'{{"rank": {i + 1}, "name": "c{i}", "nodes": {{"total": 1}}}}\n')
    dataset.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="session"):
        train_from_dataset(dataset, tmp_path / "m.joblib")


def test_train_rejects_missing_rank(tmp_path: Path) -> None:
    dataset = tmp_path / "bad.jsonl"
    lines = []
    for i in range(10):
        lines.append(f'{{"session": "s", "name": "c{i}", "nodes": {{"total": 1}}}}\n')
    dataset.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="rank"):
        train_from_dataset(dataset, tmp_path / "m.joblib")
