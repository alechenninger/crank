"""End-to-end ranking tests."""

from datetime import UTC, datetime
from pathlib import Path

from crank.config import ScoringConfig
from crank.scoring.ranker import ClusterRanker
from crank.snapshots import load_snapshots_jsonl
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
    ScoringMode,
)


def _snap(
    name: str,
    *,
    not_ready: int = 0,
    crash: int = 0,
    text: str = "",
) -> ClusterSnapshot:
    return ClusterSnapshot(
        identity=ClusterIdentity(name=name),
        collected_at=datetime.now(UTC),
        nodes=NodeState(total=10, not_ready=not_ready, ready=10 - not_ready),
        pods=PodState(total=100, running=90, crash_loop_backoff=crash),
        events=EventSummary(warnings=crash * 5, oom_killed=crash),
        searchable_text=(text,) if text else (),
    )


def test_unhealthy_cluster_ranks_above_healthy() -> None:
    ranker = ClusterRanker(ScoringConfig())
    scores = ranker.rank_snapshots(
        [
            _snap("healthy"),
            _snap("sick", not_ready=5, crash=10, text="prod payment crashloop oom"),
        ]
    )
    assert scores[0].identity.name == "sick"
    assert scores[0].total_score > scores[1].total_score
    assert scores[0].scoring_mode == ScoringMode.HEURISTIC


def test_keyword_boost_integrated_into_total_score() -> None:
    """Keywords now flow through the feature vector, not as a separate additive term."""
    ranker = ClusterRanker(ScoringConfig())
    no_kw = ranker.score_snapshot(_snap("bare", not_ready=2, crash=3))
    with_kw = ranker.score_snapshot(
        _snap("flagged", not_ready=2, crash=3, text="prod payment privileged oom")
    )
    assert with_kw.total_score > no_kw.total_score
    assert with_kw.keyword_boost > 0.0
    assert no_kw.keyword_boost == 0.0


def test_pci_prod_cluster_scores_highest_in_demo_set() -> None:
    """Regression: prod-eu-pci demo should beat dev-eu."""
    path = Path(__file__).resolve().parents[1] / "examples" / "demo_clusters.jsonl"
    snapshots = load_snapshots_jsonl(path)
    scores = ClusterRanker(ScoringConfig()).rank_snapshots(snapshots)
    assert scores[0].identity.name == "prod-eu-pci"
    assert scores[-1].identity.name == "dev-eu"
