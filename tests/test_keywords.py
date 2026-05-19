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
    boost, areas = KeywordMatcher(cfg).match(snap)
    assert boost == 3.0
    assert areas[0].area == AttentionArea.SECURITY


def test_keyword_boost_is_capped() -> None:
    rules = [
        KeywordRule(f"kw{i}", AttentionArea.RELIABILITY, weight=10.0)
        for i in range(10)
    ]
    cfg = ScoringConfig(keyword_rules=rules, keyword_boost_cap=25.0)
    text = " ".join(f"kw{i}" for i in range(10))
    snap = ClusterSnapshot(
        identity=ClusterIdentity(name="c"),
        collected_at=datetime.now(UTC),
        searchable_text=(text,),
    )
    boost, _ = KeywordMatcher(cfg).match(snap)
    assert boost == 25.0
