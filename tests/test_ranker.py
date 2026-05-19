"""End-to-end ranking tests."""

from datetime import UTC, datetime

from crank.config import ScoringConfig
from crank.scoring.ranker import ClusterRanker
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
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


def test_pci_prod_cluster_scores_highest_in_demo_set() -> None:
    """Regression: prod-eu-pci demo should beat dev-eu."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "demo_clusters.jsonl"
    snapshots: list[ClusterSnapshot] = []
    with path.open() as fh:
        for line in fh:
            data = json.loads(line)
            snapshots.append(
                ClusterSnapshot(
                    identity=ClusterIdentity(name=data["name"]),
                    collected_at=datetime.now(UTC),
                    nodes=NodeState(**data.get("nodes", {})),
                    pods=PodState(**data.get("pods", {})),
                    events=EventSummary(**data.get("events", {})),
                    namespaces=data.get("namespaces", 0),
                    deployments_unavailable=data.get("deployments_unavailable", 0),
                    statefulsets_not_ready=data.get("statefulsets_not_ready", 0),
                    daemonsets_misscheduled=data.get("daemonsets_misscheduled", 0),
                    searchable_text=tuple(data.get("searchable_text", [])),
                )
            )
    scores = ClusterRanker(ScoringConfig()).rank_snapshots(snapshots)
    assert scores[0].identity.name == "prod-eu-pci"
    assert scores[-1].identity.name == "dev-eu"
