"""In-memory collector for tests and offline demos."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from crank.types import ClusterSnapshot


class FakeCollector:
    """Returns a pre-built snapshot (no Kubernetes API)."""

    def __init__(self, snapshot: ClusterSnapshot) -> None:
        self._snapshot = snapshot

    def collect(self) -> ClusterSnapshot:
        return replace(self._snapshot, collected_at=datetime.now(UTC))
