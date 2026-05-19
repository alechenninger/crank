"""ML and heuristic cluster risk scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest

from crank.config import ScoringConfig
from crank.features.extractor import FEATURE_NAMES
from crank.types import ClusterSnapshot, FeatureVector


# Hand-tuned weights aligned with feature order (used when no trained model exists).
HEURISTIC_WEIGHTS: tuple[float, ...] = (
    18.0,  # node_not_ready_ratio
    12.0,  # node_pressure_ratio
    15.0,  # pod_failed_ratio
    10.0,  # pod_pending_ratio
    20.0,  # pod_crash_loop_ratio
    14.0,  # pod_image_pull_ratio
    12.0,  # pod_not_ready_ratio
    16.0,  # pod_privileged_ratio
    14.0,  # pod_run_as_root_ratio
    8.0,   # pod_missing_limits_ratio
    10.0,  # pending_age_hours
    8.0,   # event_warning_rate
    12.0,  # event_error_rate
    14.0,  # event_failed_scheduling_rate
    16.0,  # event_backoff_rate
    15.0,  # event_evicted_rate
    18.0,  # event_oom_rate
    12.0,  # deployment_unavailable_ratio
    11.0,  # statefulset_not_ready_ratio
    10.0,  # daemonset_misscheduled_ratio
)


class HeuristicScorer:
    """Deterministic baseline scorer from engineered features."""

    def score(self, features: FeatureVector) -> float:
        raw = sum(w * v for w, v in zip(HEURISTIC_WEIGHTS, features.values, strict=True))
        return float(min(raw, 100.0))

    def top_features(self, features: FeatureVector, limit: int = 5) -> tuple[tuple[str, float], ...]:
        contributions = [
            (name, w * val)
            for name, w, val in zip(
                features.names, HEURISTIC_WEIGHTS, features.values, strict=True
            )
        ]
        contributions.sort(key=lambda x: x[1], reverse=True)
        return tuple((n, round(v, 3)) for n, v in contributions[:limit] if v > 0)


class ClusterScorer:
    """
    Hybrid scorer: Gradient Boosting when a model is trained, else heuristic.

    An IsolationForest anomaly score can augment ranking for outlier clusters.
    """

    def __init__(self, config: ScoringConfig) -> None:
        self._config = config
        self._heuristic = HeuristicScorer()
        self._regressor: GradientBoostingRegressor | None = None
        self._anomaly: IsolationForest | None = None
        if config.model_path and config.model_path.exists():
            payload = joblib.load(config.model_path)
            self._regressor = payload.get("regressor")
            self._anomaly = payload.get("anomaly")

    @property
    def has_trained_model(self) -> bool:
        return self._regressor is not None

    def score_vector(self, features: FeatureVector) -> float:
        heuristic = self._heuristic.score(features)
        if self._regressor is None:
            return heuristic
        X = np.array([features.values])
        ml = float(self._regressor.predict(X)[0])
        ml = max(0.0, min(ml, 100.0))
        if self._anomaly is not None:
            # decision_function: lower = more anomalous; invert to a 0-15 boost.
            decision = float(self._anomaly.decision_function(X)[0])
            anomaly_boost = max(0.0, min(15.0, (0.5 - decision) * 10.0))
            ml = min(100.0, ml + anomaly_boost)
        weight = self._config.ml_weight
        return weight * ml + (1.0 - weight) * heuristic

    def top_features(self, features: FeatureVector, limit: int = 5) -> tuple[tuple[str, float], ...]:
        return self._heuristic.top_features(features, limit=limit)

    @staticmethod
    def train(
        snapshots: list[ClusterSnapshot],
        labels: list[float],
        features: list[FeatureVector],
        output: Path,
    ) -> None:
        """Train regressor + anomaly detector from labeled snapshots."""
        X = np.array([f.values for f in features])
        y = np.array(labels, dtype=float)
        regressor = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
        regressor.fit(X, y)
        anomaly = IsolationForest(contamination=0.15, random_state=42)
        anomaly.fit(X)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "regressor": regressor,
                "anomaly": anomaly,
                "feature_names": FEATURE_NAMES,
            },
            output,
        )
