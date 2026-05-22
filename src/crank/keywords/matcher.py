"""Match configurable keywords against cluster searchable text."""

from __future__ import annotations

from crank.config import ScoringConfig
from crank.types import AreaContribution, AttentionArea, ClusterSnapshot


class KeywordMatcher:
    """Boost scores when operator-interest keywords appear in cluster signals."""

    def __init__(self, config: ScoringConfig) -> None:
        self._rules = config.keyword_rules
        self._cap = config.keyword_boost_cap

    def match(self, snapshot: ClusterSnapshot) -> tuple[AreaContribution, ...]:
        corpus = "\n".join(snapshot.searchable_text)
        area_scores: dict[AttentionArea, float] = {a: 0.0 for a in AttentionArea}
        area_keywords: dict[AttentionArea, list[str]] = {a: [] for a in AttentionArea}

        for rule in self._rules:
            haystack = corpus if rule.case_sensitive else corpus.lower()
            needle = rule.pattern if rule.case_sensitive else rule.pattern.lower()
            if needle in haystack:
                area_scores[rule.area] += rule.weight
                area_keywords[rule.area].append(rule.pattern)

        contributions = tuple(
            sorted(
                (
                    AreaContribution(
                        area=area,
                        score=round(area_scores[area], 2),
                        matched_keywords=tuple(sorted(set(area_keywords[area]))),
                    )
                    for area in AttentionArea
                    if area_scores[area] > 0
                ),
                key=lambda c: c.score,
                reverse=True,
            )
        )
        return contributions
