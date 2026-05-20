"""ML and heuristic cluster risk scoring."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from crank.config import ScoringConfig
from crank.features.extractor import FEATURE_NAMES
from crank.types import ClusterSnapshot, FeatureVector, ScoringMode

logger = logging.getLogger(__name__)

MIN_TRAINING_SAMPLES = 10
MODEL_SCHEMA_VERSION = 1

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


class ModelSchemaError(ValueError):
    """Raised when a saved model is incompatible with the current feature schema."""


def _validate_feature_names(stored: object) -> None:
    if stored is None:
        raise ModelSchemaError("model payload missing feature_names")
    if not isinstance(stored, (list, tuple)):
        raise ModelSchemaError(f"model feature_names has invalid type: {type(stored)!r}")
    if tuple(stored) != FEATURE_NAMES:
        raise ModelSchemaError(
            f"model feature_names {stored!r} do not match current schema {FEATURE_NAMES!r}"
        )


class HeuristicScorer:
    """Deterministic baseline scorer from engineered features."""

    def score(self, features: FeatureVector) -> float:
        raw = sum(w * v for w, v in zip(HEURISTIC_WEIGHTS, features.values, strict=True))
        return float(min(raw, 100.0))

    def top_features(
        self, features: FeatureVector, limit: int = 5
    ) -> tuple[tuple[str, float], ...]:
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
            _validate_feature_names(payload.get("feature_names"))
            schema = payload.get("schema_version")
            if schema is not None and schema != MODEL_SCHEMA_VERSION:
                logger.warning(
                    "Model schema_version %s differs from current %s",
                    schema,
                    MODEL_SCHEMA_VERSION,
                )
            self._regressor = payload.get("regressor")
            self._anomaly = payload.get("anomaly")
            logger.info("Loaded trained model from %s", config.model_path)

    @property
    def has_trained_model(self) -> bool:
        return self._regressor is not None

    def scoring_mode(self) -> ScoringMode:
        return ScoringMode.BLENDED if self._regressor is not None else ScoringMode.HEURISTIC

    def score_vector(self, features: FeatureVector) -> float:
        heuristic = self._heuristic.score(features)
        if self._regressor is None:
            return heuristic
        X = np.array([features.values])
        ml = float(self._regressor.predict(X)[0])
        ml = max(0.0, min(ml, 100.0))
        if self._anomaly is not None:
            decision = float(self._anomaly.decision_function(X)[0])
            anomaly_boost = max(0.0, min(15.0, (0.5 - decision) * 10.0))
            ml = min(100.0, ml + anomaly_boost)
        weight = self._config.ml_weight
        return weight * ml + (1.0 - weight) * heuristic

    def top_features(
        self, features: FeatureVector, limit: int = 5
    ) -> tuple[tuple[str, float], ...]:
        if self._regressor is not None and hasattr(self._regressor, "feature_importances_"):
            importances = self._regressor.feature_importances_
            contributions = [
                (name, float(imp) * val)
                for name, imp, val in zip(
                    features.names, importances, features.values, strict=True
                )
            ]
            contributions.sort(key=lambda x: x[1], reverse=True)
            top = tuple((n, round(v, 3)) for n, v in contributions[:limit] if v > 0)
            if top:
                return top
        return self._heuristic.top_features(features, limit=limit)

    @staticmethod
    def train(
        snapshots: list[ClusterSnapshot],
        labels: list[float],
        features: list[FeatureVector],
        output: Path,
    ) -> dict[str, Any]:
        """Train regressor + anomaly detector from labeled snapshots."""
        if len(snapshots) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"need at least {MIN_TRAINING_SAMPLES} labeled snapshots to train, "
                f"got {len(snapshots)}"
            )

        X = np.array([f.values for f in features])
        y = np.array(labels, dtype=float)

        metrics: dict[str, Any] = {}
        if len(snapshots) >= 12:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.25, random_state=42
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = X, y

        regressor = GradientBoostingRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            random_state=42,
        )
        regressor.fit(X_train, y_train)
        val_pred = regressor.predict(X_val)
        metrics["mae"] = float(mean_absolute_error(y_val, val_pred))
        metrics["r2"] = float(r2_score(y_val, val_pred))

        contamination = min(0.15, max(0.05, 2.0 / len(snapshots)))
        anomaly = IsolationForest(contamination=contamination, random_state=42)
        anomaly.fit(X)

        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "regressor": regressor,
            "anomaly": anomaly,
            "feature_names": FEATURE_NAMES,
            "schema_version": MODEL_SCHEMA_VERSION,
            "trained_at": datetime.now(UTC).isoformat(),
            "sklearn_version": sklearn.__version__,
            "metrics": metrics,
            "n_samples": len(snapshots),
        }
        joblib.dump(payload, output)
        logger.info(
            "Trained model: n=%s mae=%.2f r2=%.3f",
            len(snapshots),
            metrics["mae"],
            metrics["r2"],
        )
        return metrics
