"""Collector protocol."""

from __future__ import annotations

from typing import Protocol

from crank.types import ClusterSnapshot


class ClusterCollector(Protocol):
    """Collects cluster snapshots for scoring."""

    def collect(self) -> ClusterSnapshot:
        """Return a fresh snapshot of cluster state and recent events."""
        ...
