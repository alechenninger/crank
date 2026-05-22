"""ML and heuristic cluster risk scoring."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

from crank.features.extractor import FEATURE_NAMES, N_BASE_FEATURES
from crank.types import FeatureVector, ScoringMode

logger = logging.getLogger(__name__)

MODEL_SCHEMA_VERSION = 3

HEURISTIC_WEIGHTS: tuple[float, ...] = (
    # --- base features (cluster state + events) ---
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
    # --- keyword features (already weighted by keyword rules, pass through) ---
    1.0,   # keyword_reliability
    1.0,   # keyword_security
    1.0,   # keyword_capacity
    1.0,   # keyword_compliance
    1.0,   # keyword_platform
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
    Pairwise-trained pointwise scorer, with heuristic fallback.

    When a trained model is available, ``score = coef . features`` is
    calibrated to 0-100 via min-max scaling learned at training time.
    Without a model, a hand-tuned weighted heuristic is used.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self._heuristic = HeuristicScorer()
        self._coef: np.ndarray | None = None
        self._cal_min: float = 0.0
        self._cal_max: float = 1.0
        if model_path and model_path.exists():
            payload = joblib.load(model_path)
            _validate_feature_names(payload.get("feature_names"))
            schema = payload.get("schema_version")
            if schema is not None and schema != MODEL_SCHEMA_VERSION:
                logger.warning(
                    "Model schema_version %s differs from current %s",
                    schema,
                    MODEL_SCHEMA_VERSION,
                )
            self._coef = payload.get("coef")
            self._cal_min = float(payload.get("calibration_min", 0.0))
            self._cal_max = float(payload.get("calibration_max", 1.0))
            logger.info("Loaded trained model from %s", model_path)

    @property
    def has_trained_model(self) -> bool:
        return self._coef is not None

    @property
    def scoring_mode(self) -> ScoringMode:
        return ScoringMode.ML if self._coef is not None else ScoringMode.HEURISTIC

    def score_vector(self, features: FeatureVector) -> float:
        if self._coef is None:
            return self._heuristic.score(features)
        raw = float(np.dot(self._coef, features.values))
        span = self._cal_max - self._cal_min
        if span <= 0:
            return 50.0
        scaled = 100.0 * (raw - self._cal_min) / span
        return max(0.0, min(scaled, 100.0))

    def top_features(
        self, features: FeatureVector, limit: int = 5
    ) -> tuple[tuple[str, float], ...]:
        if self._coef is not None:
            contributions = [
                (name, float(coef) * val)
                for name, coef, val in zip(
                    features.names, self._coef, features.values, strict=True
                )
            ]
            contributions.sort(key=lambda x: x[1], reverse=True)
            top = tuple((n, round(v, 3)) for n, v in contributions[:limit] if v > 0)
            if top:
                return top
        return self._heuristic.top_features(features, limit=limit)

    def keyword_contribution(self, features: FeatureVector) -> float:
        """Portion of the score attributable to keyword features."""
        kw_values = features.values[N_BASE_FEATURES:]
        if self._coef is not None:
            kw_coefs = self._coef[N_BASE_FEATURES:]
            return max(0.0, float(np.dot(kw_coefs, kw_values)))
        kw_weights = HEURISTIC_WEIGHTS[N_BASE_FEATURES:]
        return max(0.0, sum(w * v for w, v in zip(kw_weights, kw_values, strict=True)))
