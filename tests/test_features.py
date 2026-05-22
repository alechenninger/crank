"""Feature extraction tests."""

from datetime import UTC, datetime

from crank.features.extractor import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    KEYWORD_FEATURE_NAMES,
    N_BASE_FEATURES,
    RATIO_FEATURE_INDICES,
    FeatureExtractor,
    keyword_feature_values,
)
from crank.types import (
    AreaContribution,
    AttentionArea,
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
    WorkloadState,
)


def test_base_feature_vector_length_matches_names() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=2),
        pods=PodState(total=100, crash_loop_backoff=5),
        events=EventSummary(warnings=24, window_hours=24),
    )
    fv = FeatureExtractor().extract(snap)
    assert len(fv.names) == len(BASE_FEATURE_NAMES)
    assert len(fv.values) == len(BASE_FEATURE_NAMES)


def test_full_feature_vector_includes_keyword_features() -> None:
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="t"),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=2),
        pods=PodState(total=100, crash_loop_backoff=5),
        events=EventSummary(warnings=24, window_hours=24),
    )
    areas = (
        AreaContribution(
            area=AttentionArea.RELIABILITY,
            score=3.5,
            matched_keywords=("crashloop",),
        ),
    )
    fv = FeatureExtractor().extract_full(snap, areas)
    assert len(fv.names) == len(FEATURE_NAMES)
    assert len(fv.values) == len(FEATURE_NAMES)
    d = fv.as_dict()
    assert d["keyword_reliability"] == 3.5
    assert d["keyword_security"] == 0.0


def test_feature_names_are_base_plus_keyword() -> None:
    assert FEATURE_NAMES == BASE_FEATURE_NAMES + KEYWORD_FEATURE_NAMES
    assert N_BASE_FEATURES == len(BASE_FEATURE_NAMES)


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
    rate_indices = set(range(len(BASE_FEATURE_NAMES))) - RATIO_FEATURE_INDICES
    for i in rate_indices:
        assert fv.values[i] >= 0.0, f"{fv.names[i]}={fv.values[i]}"


def test_workload_ratios_use_totals_not_unavailable_only() -> None:
    """12/100 unavailable should score much lower than 12/12."""
    large = ClusterSnapshot(
        identity=ClusterIdentity(name="large"),
        collected_at=datetime.now(UTC),
        workloads=WorkloadState(deployments_total=100, deployments_unavailable=12),
    )
    small = ClusterSnapshot(
        identity=ClusterIdentity(name="small"),
        collected_at=datetime.now(UTC),
        workloads=WorkloadState(deployments_total=12, deployments_unavailable=12),
    )
    ext = FeatureExtractor()
    large_ratio = ext.extract(large).as_dict()["deployment_unavailable_ratio"]
    small_ratio = ext.extract(small).as_dict()["deployment_unavailable_ratio"]
    assert large_ratio < small_ratio
    assert abs(large_ratio - 0.12) < 0.01
    assert abs(small_ratio - 1.0) < 0.01


def test_rate_with_zero_window_returns_zero() -> None:
    from crank.features.extractor import _rate

    assert _rate(10, 0.0) == 0.0
    assert _rate(0, 0.0) == 0.0


def test_ratio_feature_indices_match_feature_names() -> None:
    """Guard against RATIO_FEATURE_INDICES drifting from BASE_FEATURE_NAMES."""
    expected = frozenset(
        i
        for i, name in enumerate(BASE_FEATURE_NAMES)
        if name.endswith("_ratio") or name == "pending_age_hours"
    )
    assert RATIO_FEATURE_INDICES == expected


def test_keyword_feature_values_from_area_contributions() -> None:
    areas = (
        AreaContribution(area=AttentionArea.SECURITY, score=3.0),
        AreaContribution(area=AttentionArea.CAPACITY, score=2.5),
    )
    values = keyword_feature_values(areas)
    assert len(values) == len(AttentionArea)
    assert values[0] == 0.0   # reliability
    assert values[1] == 3.0   # security
    assert values[2] == 2.5   # capacity
    assert values[3] == 0.0   # compliance
    assert values[4] == 0.0   # platform
