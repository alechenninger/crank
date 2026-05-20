"""Scorer unit tests."""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import pytest

from crank.config import ScoringConfig
from crank.features.extractor import FEATURE_NAMES, FeatureExtractor
from crank.model.scorer import ClusterScorer, HeuristicScorer, ModelSchemaError
from crank.training import train_from_dataset
from crank.types import ClusterIdentity, ClusterSnapshot, FeatureVector, NodeState


def _features(**kwargs: float) -> FeatureVector:
    values = tuple(kwargs.get(n, 0.0) for n in FEATURE_NAMES)
    return FeatureVector(names=FEATURE_NAMES, values=values)


def test_heuristic_scores_higher_with_more_signals() -> None:
    scorer = HeuristicScorer()
    low = scorer.score(_features())
    high = scorer.score(_features(pod_crash_loop_ratio=1.0, node_not_ready_ratio=1.0))
    assert high > low


def test_heuristic_mode_without_model() -> None:
    cluster_scorer = ClusterScorer(ScoringConfig())
    assert not cluster_scorer.has_trained_model
    assert cluster_scorer.scoring_mode().value == "heuristic"


def test_blended_mode_with_trained_model(tmp_path: Path) -> None:
    dataset = Path(__file__).resolve().parents[1] / "examples" / "training_dataset.jsonl"
    model_path = tmp_path / "model.joblib"
    train_from_dataset(dataset, model_path)
    scorer = ClusterScorer(ScoringConfig(model_path=model_path))
    assert scorer.scoring_mode().value == "blended"
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=5),
    )
    fv = FeatureExtractor().extract(snap)
    score = scorer.score_vector(fv)
    assert 0.0 <= score <= 100.0


def test_model_rejects_mismatched_feature_names(tmp_path: Path) -> None:
    bad = tmp_path / "bad.joblib"
    joblib.dump(
        {"regressor": None, "anomaly": None, "feature_names": ("wrong",)},
        bad,
    )
    with pytest.raises(ModelSchemaError):
        ClusterScorer(ScoringConfig(model_path=bad))
