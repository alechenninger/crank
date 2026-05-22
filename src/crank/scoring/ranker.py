"""Rank clusters by combined ML + keyword attention score."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from crank.collector.base import ClusterCollector
from crank.config import ScoringConfig
from crank.features.extractor import FeatureExtractor
from crank.keywords.matcher import KeywordMatcher
from crank.model.scorer import ClusterScorer
from crank.types import AreaContribution, ClusterScore, ClusterSnapshot, ScoringMode

logger = logging.getLogger(__name__)


def _summarize(
    snapshot: ClusterSnapshot,
    total_score: float,
    scoring_mode: ScoringMode,
    keyword_boost: float,
    areas: tuple[AreaContribution, ...],
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
    mode_label = scoring_mode.value
    return (
        f"score={total_score:.1f} mode={mode_label} keyword_boost={keyword_boost:.1f}; "
        + "; ".join(parts)
    )


class ClusterRanker:
    """Scores one or more clusters and returns attention ranking."""

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._extractor = FeatureExtractor()
        self._matcher = KeywordMatcher(config)
        self._scorer = ClusterScorer(model_path=config.model_path)

    def score_snapshot(self, snapshot: ClusterSnapshot) -> ClusterScore:
        _, areas = self._matcher.match(snapshot)
        features = self._extractor.extract_full(snapshot, areas)
        total_score = self._scorer.score_vector(features)
        scoring_mode = self._scorer.scoring_mode()
        keyword_boost = self._scorer.keyword_contribution(features)

        base_score = total_score - keyword_boost

        return ClusterScore(
            identity=snapshot.identity,
            rank=0,
            total_score=round(total_score, 2),
            base_score=round(base_score, 2),
            scoring_mode=scoring_mode,
            keyword_boost=round(keyword_boost, 2),
            area_contributions=areas,
            top_features=self._scorer.top_features(features),
            summary=_summarize(snapshot, total_score, scoring_mode, keyword_boost, areas),
        )

    def rank(self, collectors: Sequence[ClusterCollector]) -> list[ClusterScore]:
        snapshots = self._collect_parallel(collectors)
        return self.rank_snapshots(snapshots)

    def _collect_parallel(self, collectors: Sequence[ClusterCollector]) -> list[ClusterSnapshot]:
        if not collectors:
            return []
        if len(collectors) == 1:
            return [collectors[0].collect()]

        snapshots: list[ClusterSnapshot] = []
        workers = min(len(collectors), 8)
        logger.info("Collecting %s clusters with %s workers", len(collectors), workers)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(c.collect) for c in collectors]
            for future in as_completed(futures):
                snapshots.append(future.result())
        return snapshots

    def rank_snapshots(self, snapshots: list[ClusterSnapshot]) -> list[ClusterScore]:
        scores = [self.score_snapshot(s) for s in snapshots]
        scores.sort(key=lambda s: s.total_score, reverse=True)
        for i, score in enumerate(scores, start=1):
            score.rank = i
        return scores
