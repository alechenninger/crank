"""Train crank models from labeled snapshot datasets."""

from __future__ import annotations

import json
from pathlib import Path

from crank.features.extractor import FeatureExtractor
from crank.model.scorer import ClusterScorer
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
)


def _snapshot_from_dict(data: dict) -> ClusterSnapshot:
    nodes = NodeState(**data.get("nodes", {}))
    pods = PodState(**data.get("pods", {}))
    events = EventSummary(**data.get("events", {}))
    identity = ClusterIdentity(
        name=data.get("name", "unknown"),
        context=data.get("context"),
    )
    return ClusterSnapshot(
        identity=identity,
        collected_at=data.get("collected_at"),  # type: ignore[arg-type]
        nodes=nodes,
        pods=pods,
        events=events,
        namespaces=int(data.get("namespaces", 0)),
        deployments_unavailable=int(data.get("deployments_unavailable", 0)),
        statefulsets_not_ready=int(data.get("statefulsets_not_ready", 0)),
        daemonsets_misscheduled=int(data.get("daemonsets_misscheduled", 0)),
        searchable_text=tuple(data.get("searchable_text", [])),
    )


def train_from_dataset(dataset_path: Path, model_output: Path) -> None:
    """
    Train from JSON lines: each row has snapshot fields + label (0-100).

    Example:
      {"name": "prod-us", "label": 85, "nodes": {...}, "pods": {...}, ...}
    """
    snapshots: list[ClusterSnapshot] = []
    labels: list[float] = []
    extractor = FeatureExtractor()
    features_list = []

    with dataset_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            label = float(row.pop("label"))
            snapshots.append(_snapshot_from_dict(row))
            labels.append(label)
            features_list.append(extractor.extract(snapshots[-1]))

    if len(snapshots) < 3:
        raise ValueError("need at least 3 labeled snapshots to train")

    ClusterScorer.train(snapshots, labels, features_list, model_output)
