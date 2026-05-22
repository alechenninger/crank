"""Load cluster snapshots from JSON/JSONL."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
    WorkloadState,
)


def _parse_collected_at(raw: object | None) -> datetime:
    if raw is None:
        return datetime.now(UTC)
    if isinstance(raw, datetime):
        if raw.tzinfo is None:
            return raw.replace(tzinfo=UTC)
        return raw.astimezone(UTC)
    if isinstance(raw, str):
        text = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError(f"invalid collected_at: {raw!r}")


def snapshot_from_dict(data: dict[str, Any]) -> ClusterSnapshot:
    """Build a ClusterSnapshot from a JSON object (e.g. one JSONL row)."""
    payload = dict(data)
    name = payload.pop("name", "unknown")
    context = payload.pop("context", None)
    payload.pop("label", None)
    identity = ClusterIdentity(name=name, context=context)
    snapshot = ClusterSnapshot(
        identity=identity,
        collected_at=_parse_collected_at(payload.pop("collected_at", None)),
        nodes=NodeState(**payload.pop("nodes", {})),
        pods=PodState(**payload.pop("pods", {})),
        events=EventSummary(**payload.pop("events", {})),
        workloads=WorkloadState(
            deployments_total=int(payload.pop("deployments_total", 0)),
            deployments_unavailable=int(payload.pop("deployments_unavailable", 0)),
            statefulsets_total=int(payload.pop("statefulsets_total", 0)),
            statefulsets_not_ready=int(payload.pop("statefulsets_not_ready", 0)),
            daemonsets_total=int(payload.pop("daemonsets_total", 0)),
            daemonsets_misscheduled=int(payload.pop("daemonsets_misscheduled", 0)),
        ),
        namespaces=int(payload.pop("namespaces", 0)),
        searchable_text=tuple(payload.pop("searchable_text", [])),
    )
    if payload:
        logger.warning("unknown snapshot fields ignored: %s", sorted(payload))
    return snapshot


def load_snapshots_jsonl(path: Path | Traversable) -> list[ClusterSnapshot]:
    """Load snapshots from a JSONL file (one JSON object per line)."""
    snapshots: list[ClusterSnapshot] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()  # type: ignore[assignment]
            if not line:
                continue
            snapshots.append(snapshot_from_dict(json.loads(line)))
    return snapshots
