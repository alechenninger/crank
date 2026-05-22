"""Train crank models from ranked-session snapshot datasets."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression

from crank.config import ScoringConfig
from crank.features.extractor import FEATURE_NAMES, FeatureExtractor
from crank.keywords.matcher import KeywordMatcher
from crank.model.scorer import MODEL_SCHEMA_VERSION
from crank.snapshots import snapshot_from_dict
from crank.types import ClusterSnapshot, FeatureVector

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 10


def _generate_pairs(
    X: np.ndarray,
    ranks: list[int],
    session_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate pairwise training samples from ranked sessions.

    For each session, every pair (i, j) where rank[i] < rank[j]
    (lower rank = more attention) produces two symmetric rows so that
    LogisticRegression sees both classes:

    - X[i] - X[j] with label 1  (preferred first)
    - X[j] - X[i] with label 0  (non-preferred first)
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, sid in enumerate(session_ids):
        groups[sid].append(idx)

    diffs: list[np.ndarray] = []
    labels: list[int] = []
    for indices in groups.values():
        if len(indices) < 2:
            continue
        for i, j in combinations(indices, 2):
            if ranks[i] < ranks[j]:
                preferred, other = i, j
            elif ranks[j] < ranks[i]:
                preferred, other = j, i
            else:
                continue
            diffs.append(X[preferred] - X[other])
            labels.append(1)
            diffs.append(X[other] - X[preferred])
            labels.append(0)
    return np.array(diffs), np.array(labels)


def train_model(
    snapshots: list[ClusterSnapshot],
    ranks: list[int],
    session_ids: list[str],
    features: list[FeatureVector],
    output: Path,
) -> dict[str, Any]:
    """Train pairwise logistic regression from ranked sessions."""
    if len(snapshots) < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"need at least {MIN_TRAINING_SAMPLES} snapshots to train, "
            f"got {len(snapshots)}"
        )

    X = np.array([f.values for f in features])
    X_pairs, y_pairs = _generate_pairs(X, ranks, session_ids)

    if len(X_pairs) == 0:
        raise ValueError(
            "no valid pairs generated; each session needs at least 2 ranked snapshots"
        )

    clf = LogisticRegression(fit_intercept=False, C=1.0, max_iter=1000)
    clf.fit(X_pairs, y_pairs)
    coef = clf.coef_[0]

    raw_scores = X @ coef
    cal_min = float(raw_scores.min())
    cal_max = float(raw_scores.max())

    pair_correct = 0
    pair_total = 0
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, sid in enumerate(session_ids):
        groups[sid].append(idx)
    for indices in groups.values():
        for i, j in combinations(indices, 2):
            if ranks[i] == ranks[j]:
                continue
            pair_total += 1
            if (ranks[i] < ranks[j]) == (raw_scores[i] > raw_scores[j]):
                pair_correct += 1

    pairwise_accuracy = pair_correct / pair_total if pair_total > 0 else 0.0

    # Higher raw score should correlate with lower rank (more attention),
    # so negate ranks for a positive tau when ordering is correct.
    tau, _ = kendalltau(raw_scores, [-r for r in ranks])

    metrics: dict[str, Any] = {
        "pairwise_accuracy": float(pairwise_accuracy),
        "kendall_tau": float(tau),
        "n_pairs": len(X_pairs),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "coef": coef,
        "calibration_min": cal_min,
        "calibration_max": cal_max,
        "feature_names": FEATURE_NAMES,
        "schema_version": MODEL_SCHEMA_VERSION,
        "trained_at": datetime.now(UTC).isoformat(),
        "sklearn_version": sklearn.__version__,
        "metrics": metrics,
        "n_samples": len(snapshots),
    }
    joblib.dump(payload, output)
    logger.info(
        "Trained model: n=%s pairs=%s pairwise_acc=%.3f tau=%.3f",
        len(snapshots),
        len(X_pairs),
        pairwise_accuracy,
        tau,
    )
    return metrics


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
            areas = matcher.match(snapshot)
            features_list.append(extractor.extract_full(snapshot, areas))

    return train_model(snapshots, ranks, session_ids, features_list, model_output)
