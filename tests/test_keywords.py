"""Keyword matcher tests."""

from datetime import UTC, datetime

from crank.config import KeywordRule, ScoringConfig
from crank.keywords.matcher import KeywordMatcher
from crank.types import AttentionArea, ClusterIdentity, ClusterSnapshot


def test_keyword_boosts_security_area() -> None:
    cfg = ScoringConfig(
        keyword_rules=[
            KeywordRule("privileged", AttentionArea.SECURITY, weight=3.0),
        ]
    )
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="c"),
        collected_at=datetime.now(UTC),
        searchable_text=("prod/api privileged container",),
    )
    areas = KeywordMatcher(cfg).match(snap)
    assert len(areas) == 1
    assert areas[0].area == AttentionArea.SECURITY
    assert areas[0].score == 3.0


def test_multiple_areas_sorted_by_score() -> None:
    cfg = ScoringConfig(
        keyword_rules=[
            KeywordRule("privileged", AttentionArea.SECURITY, weight=3.0),
            KeywordRule("oom", AttentionArea.CAPACITY, weight=5.0),
        ]
    )
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="c"),
        collected_at=datetime.now(UTC),
        searchable_text=("oom privileged",),
    )
    areas = KeywordMatcher(cfg).match(snap)
    assert len(areas) == 2
    assert areas[0].area == AttentionArea.CAPACITY
    assert areas[0].score == 5.0
    assert areas[1].area == AttentionArea.SECURITY


def test_no_match_returns_empty() -> None:
    cfg = ScoringConfig(
        keyword_rules=[
            KeywordRule("privileged", AttentionArea.SECURITY, weight=3.0),
        ]
    )
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="c"),
        collected_at=datetime.now(UTC),
        searchable_text=("nothing relevant",),
    )
    areas = KeywordMatcher(cfg).match(snap)
    assert areas == ()
