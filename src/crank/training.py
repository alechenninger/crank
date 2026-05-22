"""Train crank models from ranked-session snapshot datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from crank.config import ScoringConfig
from crank.features.extractor import FeatureExtractor
from crank.keywords.matcher import KeywordMatcher
from crank.model.scorer import ClusterScorer
from crank.snapshots import snapshot_from_dict
from crank.types import ClusterSnapshot, FeatureVector

logger = logging.getLogger(__name__)


def train_from_dataset(
    dataset_path: Path,
    model_output: Path,
    config: ScoringConfig | None = None,
) -> dict[str, Any]:
    """
    Train from ranked sessions in JSONL format.

    Each row has a ``session`` identifier and an ordinal ``rank``
    (1 = most attention needed). Pairs are generated within each session
    so rankings from different days or conditions don't contaminate each
    other. Keyword rules from *config* produce keyword features during
    training so the model learns interactions between cluster health and
    operator-interest context.

    Example row::

        {"session": "2026-05-19-triage", "rank": 1, "name": "prod-eu-pci",
         "nodes": {...}, "pods": {...}, ...}
    """
    if config is None:
        config = ScoringConfig()

    snapshots: list[ClusterSnapshot] = []
    ranks: list[int] = []
    session_ids: list[str] = []
    extractor = FeatureExtractor()
    matcher = KeywordMatcher(config)
    features_list: list[FeatureVector] = []

    with dataset_path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "session" not in row:
                raise ValueError(f"line {line_no}: missing required field 'session'")
            if "rank" not in row:
                raise ValueError(f"line {line_no}: missing required field 'rank'")
            session_id = str(row.pop("session"))
            rank = int(row.pop("rank"))
            snapshot = snapshot_from_dict(row)
            snapshots.append(snapshot)
            ranks.append(rank)
            session_ids.append(session_id)
            _, areas = matcher.match(snapshot)
            features_list.append(extractor.extract_full(snapshot, areas))

    metrics = ClusterScorer.train(
        snapshots, ranks, session_ids, features_list, model_output,
    )
    return metrics
