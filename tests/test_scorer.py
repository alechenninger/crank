"""Scorer unit tests."""

from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pytest

from crank.config import ScoringConfig
from crank.features.extractor import FEATURE_NAMES, FeatureExtractor, N_BASE_FEATURES
from crank.keywords.matcher import KeywordMatcher
from crank.model.scorer import (
    ClusterScorer,
    HeuristicScorer,
    ModelSchemaError,
)
from crank.training import _generate_pairs
from crank.types import ClusterIdentity, ClusterSnapshot, FeatureVector, NodeState


def _features(**kwargs: float) -> FeatureVector:
    values = tuple(kwargs.get(n, 0.0) for n in FEATURE_NAMES)
    return FeatureVector(names=FEATURE_NAMES, values=values)


def test_heuristic_all_zero_features_scores_zero() -> None:
    scorer = HeuristicScorer()
    assert scorer.score(_features()) == 0.0


def test_score_vector_returns_midpoint_when_calibration_span_is_zero(tmp_path: Path) -> None:
    """When cal_min == cal_max, score_vector should return 50.0."""
    coef = np.ones(len(FEATURE_NAMES))
    payload = {
        "coef": coef,
        "calibration_min": 5.0,
        "calibration_max": 5.0,
        "feature_names": FEATURE_NAMES,
        "schema_version": 3,
    }
    model_path = tmp_path / "flat.joblib"
    joblib.dump(payload, model_path)
    scorer = ClusterScorer(model_path=model_path)
    fv = _features(node_not_ready_ratio=0.5)
    assert scorer.score_vector(fv) == 50.0


def test_heuristic_scores_higher_with_more_signals() -> None:
    scorer = HeuristicScorer()
    low = scorer.score(_features())
    high = scorer.score(_features(pod_crash_loop_ratio=1.0, node_not_ready_ratio=1.0))
    assert high > low


def test_heuristic_includes_keyword_features() -> None:
    scorer = HeuristicScorer()
    without = scorer.score(_features())
    with_kw = scorer.score(_features(keyword_reliability=5.0, keyword_security=3.0))
    assert with_kw > without
    assert with_kw - without == pytest.approx(8.0, abs=0.01)


def test_heuristic_mode_without_model() -> None:
    cluster_scorer = ClusterScorer()
    assert not cluster_scorer.has_trained_model
    assert cluster_scorer.scoring_mode.value == "heuristic"


def test_ml_mode_with_trained_model(trained_model_path: Path) -> None:
    scorer = ClusterScorer(model_path=trained_model_path)
    assert scorer.scoring_mode.value == "ml"

    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=5),
    )
    config = ScoringConfig()
    extractor = FeatureExtractor()
    matcher = KeywordMatcher(config)
    areas = matcher.match(snap)
    fv = extractor.extract_full(snap, areas)
    score = scorer.score_vector(fv)
    assert 0.0 <= score <= 100.0


def test_ml_score_differs_from_heuristic(trained_model_path: Path) -> None:
    """With a trained model, score comes from learned coef, not heuristic weights."""
    ml_scorer = ClusterScorer(model_path=trained_model_path)
    heuristic_scorer = HeuristicScorer()

    fv = _features(
        node_not_ready_ratio=0.3,
        pod_crash_loop_ratio=0.1,
        event_oom_rate=0.5,
        keyword_reliability=4.0,
    )
    ml_score = ml_scorer.score_vector(fv)
    heuristic_score = heuristic_scorer.score(fv)
    assert ml_score != pytest.approx(heuristic_score, abs=0.01)


def test_heuristic_keyword_contribution() -> None:
    scorer = ClusterScorer()
    no_kw = _features(node_not_ready_ratio=0.5)
    with_kw = _features(node_not_ready_ratio=0.5, keyword_reliability=5.0)
    assert scorer.keyword_contribution(no_kw) == pytest.approx(0.0, abs=0.01)
    assert scorer.keyword_contribution(with_kw) == pytest.approx(5.0, abs=0.01)


def test_ml_keyword_contribution_is_non_negative(trained_model_path: Path) -> None:
    scorer = ClusterScorer(model_path=trained_model_path)
    fv = _features(node_not_ready_ratio=0.5, keyword_reliability=5.0)
    assert scorer.keyword_contribution(fv) >= 0.0


def test_model_rejects_mismatched_feature_names(tmp_path: Path) -> None:
    bad = tmp_path / "bad.joblib"
    joblib.dump(
        {"coef": None, "feature_names": ("wrong",)},
        bad,
    )
    with pytest.raises(ModelSchemaError):
        ClusterScorer(model_path=bad)


# --- pair generation ---


def test_pair_generation_within_sessions() -> None:
    X = np.array([
        [1.0, 0.0],
        [0.5, 0.0],
        [0.0, 0.0],
        [0.8, 0.0],
        [0.2, 0.0],
    ])
    ranks = [1, 2, 3, 1, 2]
    sessions = ["s1", "s1", "s1", "s2", "s2"]

    diffs, labels = _generate_pairs(X, ranks, sessions)

    # C(3,2)=3 from s1 + C(2,2)=1 from s2 = 4 ordered pairs, x2 for both directions
    assert len(diffs) == 8
    assert sum(labels) == 4  # half are label=1, half are label=0


def test_pair_generation_no_cross_session_pairs() -> None:
    """Snapshots from different sessions must not be paired."""
    X = np.array([[1.0], [0.0]])
    ranks = [1, 1]
    sessions = ["a", "b"]

    diffs, labels = _generate_pairs(X, ranks, sessions)
    assert len(diffs) == 0


def test_trained_model_orders_sick_above_healthy(trained_model_path: Path) -> None:
    """A model trained on the example dataset should score sicker clusters higher."""
    scorer = ClusterScorer(model_path=trained_model_path)

    sick = _features(
        node_not_ready_ratio=0.5,
        pod_crash_loop_ratio=0.2,
        event_oom_rate=1.0,
    )
    healthy = _features()
    assert scorer.score_vector(sick) > scorer.score_vector(healthy)


def test_calibrated_scores_in_range(trained_model_path: Path) -> None:
    """All scores from a trained model should fall in [0, 100]."""
    scorer = ClusterScorer(model_path=trained_model_path)

    for nr in (0.0, 0.1, 0.5, 1.0):
        for cl in (0.0, 0.05, 0.2, 0.5):
            fv = _features(node_not_ready_ratio=nr, pod_crash_loop_ratio=cl)
            score = scorer.score_vector(fv)
            assert 0.0 <= score <= 100.0, f"score {score} out of range for nr={nr}, cl={cl}"


def test_top_features_with_trained_coef(trained_model_path: Path) -> None:
    scorer = ClusterScorer(model_path=trained_model_path)

    fv = _features(node_not_ready_ratio=0.5, pod_crash_loop_ratio=0.3)
    top = scorer.top_features(fv, limit=3)
    assert len(top) <= 3
    for name, value in top:
        assert isinstance(name, str)
        assert value > 0
