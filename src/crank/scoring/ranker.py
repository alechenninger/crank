"""Rank clusters by combined ML + keyword attention score."""

from __future__ import annotations

from crank.collector.base import ClusterCollector
from crank.config import ScoringConfig
from crank.features.extractor import FeatureExtractor
from crank.keywords.matcher import KeywordMatcher
from crank.model.scorer import ClusterScorer
from crank.types import AttentionArea, ClusterScore, ClusterSnapshot


def _summarize(
    snapshot: ClusterSnapshot,
    ml_score: float,
    keyword_boost: float,
    areas: tuple,
) -> str:
    parts: list[str] = []
    if snapshot.nodes.not_ready:
        parts.append(f"{snapshot.nodes.not_ready} node(s) not Ready")
    if snapshot.pods.crash_loop_backoff:
        parts.append(f"{snapshot.pods.crash_loop_backoff} CrashLoopBackOff")
    if snapshot.events.oom_killed:
        parts.append(f"{snapshot.events.oom_killed} OOM event(s)")
    if snapshot.events.evicted:
        parts.append(f"{snapshot.events.evicted} eviction(s)")
    if snapshot.pods.privileged:
        parts.append(f"{snapshot.pods.privileged} privileged pod(s)")
    if areas:
        top = areas[0]
        parts.append(f"top area: {top.area.value}")
    if not parts:
        parts.append("no critical signals; baseline attention")
    return (
        f"ml={ml_score:.1f} keyword_boost={keyword_boost:.1f}; "
        + "; ".join(parts)
    )


class ClusterRanker:
    """Scores one or more clusters and returns attention ranking."""

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._extractor = FeatureExtractor()
        self._matcher = KeywordMatcher(config)
        self._scorer = ClusterScorer(config)

    def score_snapshot(self, snapshot: ClusterSnapshot) -> ClusterScore:
        features = self._extractor.extract(snapshot)
        ml_score = self._scorer.score_vector(features)
        keyword_boost, areas = self._matcher.match(snapshot)
        total = min(100.0, ml_score + keyword_boost)
        return ClusterScore(
            identity=snapshot.identity,
            rank=0,
            total_score=round(total, 2),
            ml_score=round(ml_score, 2),
            keyword_boost=round(keyword_boost, 2),
            area_contributions=areas,
            top_features=self._scorer.top_features(features),
            summary=_summarize(snapshot, ml_score, keyword_boost, areas),
        )

    def rank(self, collectors: list[ClusterCollector]) -> list[ClusterScore]:
        scores = [self.score_snapshot(c.collect()) for c in collectors]
        scores.sort(key=lambda s: s.total_score, reverse=True)
        for i, score in enumerate(scores, start=1):
            score.rank = i
        return scores

    def rank_snapshots(self, snapshots: list[ClusterSnapshot]) -> list[ClusterScore]:
        scores = [self.score_snapshot(s) for s in snapshots]
        scores.sort(key=lambda s: s.total_score, reverse=True)
        for i, score in enumerate(scores, start=1):
            score.rank = i
        return scores
