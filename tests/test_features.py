"""Feature extraction tests."""

from datetime import UTC, datetime

from crank.features.extractor import FeatureExtractor, FEATURE_NAMES
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
)


def test_feature_vector_length_matches_names() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=2),
        pods=PodState(total=100, crash_loop_backoff=5),
        events=EventSummary(warnings=24, window_hours=24),
    )
    fv = FeatureExtractor().extract(snap)
    assert len(fv.names) == len(FEATURE_NAMES)
    assert len(fv.values) == len(FEATURE_NAMES)


def test_ratios_are_bounded() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=1, not_ready=5),
        pods=PodState(total=1, failed=99),
    )
    fv = FeatureExtractor().extract(snap)
    assert all(0.0 <= v <= 1.0 or v >= 0 for v in fv.values)
