"""In-memory collector for tests and offline demos."""

from __future__ import annotations

from datetime import UTC, datetime

from crank.types import ClusterSnapshot


class FakeCollector:
    """Returns a pre-built snapshot (no Kubernetes API)."""

    def __init__(self, snapshot: ClusterSnapshot) -> None:
        self._snapshot = snapshot

    def collect(self) -> ClusterSnapshot:
        return ClusterSnapshot(
            identity=self._snapshot.identity,
            collected_at=datetime.now(UTC),
            nodes=self._snapshot.nodes,
            pods=self._snapshot.pods,
            events=self._snapshot.events,
            namespaces=self._snapshot.namespaces,
            deployments_unavailable=self._snapshot.deployments_unavailable,
            statefulsets_not_ready=self._snapshot.statefulsets_not_ready,
            daemonsets_misscheduled=self._snapshot.daemonsets_misscheduled,
            searchable_text=self._snapshot.searchable_text,
        )
