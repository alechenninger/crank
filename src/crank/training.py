"""Train crank models from labeled snapshot datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crank.features.extractor import FeatureExtractor
from crank.model.scorer import ClusterScorer
from crank.snapshots import snapshot_from_dict
from crank.types import ClusterSnapshot

logger = logging.getLogger(__name__)


def train_from_dataset(dataset_path: Path, model_output: Path) -> dict[str, Any]:
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
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "label" not in row:
                raise ValueError(f"line {line_no}: missing required field 'label'")
            label = float(row.pop("label"))
            snapshots.append(snapshot_from_dict(row))
            labels.append(label)
            features_list.append(extractor.extract(snapshots[-1]))

    metrics = ClusterScorer.train(snapshots, labels, features_list, model_output)
    return metrics
