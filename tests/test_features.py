"""Feature extraction tests."""

from datetime import UTC, datetime

from crank.features.extractor import (
    FEATURE_NAMES,
    RATIO_FEATURE_INDICES,
    FeatureExtractor,
)
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


def test_ratio_features_are_bounded() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=2),
        pods=PodState(total=100, failed=5),
        events=EventSummary(warnings=24, window_hours=24),
    )
    fv = FeatureExtractor().extract(snap)
    for i in RATIO_FEATURE_INDICES:
        assert 0.0 <= fv.values[i] <= 1.0, f"{fv.names[i]}={fv.values[i]}"


def test_event_rate_features_are_non_negative() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        events=EventSummary(warnings=100, errors=50, window_hours=24),
    )
    fv = FeatureExtractor().extract(snap)
    rate_indices = set(range(len(FEATURE_NAMES))) - RATIO_FEATURE_INDICES
    for i in rate_indices:
        assert fv.values[i] >= 0.0, f"{fv.names[i]}={fv.values[i]}"


def test_workload_ratios_use_totals_not_unavailable_only() -> None:
    """12/100 unavailable should score much lower than 12/12."""
    large = ClusterSnapshot(
        identity=ClusterIdentity(name="large"),
        collected_at=datetime.now(UTC),
        deployments_total=100,
        deployments_unavailable=12,
    )
    small = ClusterSnapshot(
        identity=ClusterIdentity(name="small"),
        collected_at=datetime.now(UTC),
        deployments_total=12,
        deployments_unavailable=12,
    )
    ext = FeatureExtractor()
    large_ratio = ext.extract(large).as_dict()["deployment_unavailable_ratio"]
    small_ratio = ext.extract(small).as_dict()["deployment_unavailable_ratio"]
    assert large_ratio < small_ratio
    assert abs(large_ratio - 0.12) < 0.01
    assert abs(small_ratio - 1.0) < 0.01
