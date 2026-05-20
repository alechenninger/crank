"""Convert cluster snapshots into normalized ML feature vectors."""

from __future__ import annotations

import math

from crank.types import ClusterSnapshot, FeatureVector

# Indices of ratio features in FEATURE_NAMES (bounded 0–1); others are rates or scaled age.
RATIO_FEATURE_INDICES: frozenset[int] = frozenset(
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 19}
)

FEATURE_NAMES: tuple[str, ...] = (
    "node_not_ready_ratio",
    "node_pressure_ratio",
    "pod_failed_ratio",
    "pod_pending_ratio",
    "pod_crash_loop_ratio",
    "pod_image_pull_ratio",
    "pod_not_ready_ratio",
    "pod_privileged_ratio",
    "pod_run_as_root_ratio",
    "pod_missing_limits_ratio",
    "pending_age_hours",
    "event_warning_rate",
    "event_error_rate",
    "event_failed_scheduling_rate",
    "event_backoff_rate",
    "event_evicted_rate",
    "event_oom_rate",
    "deployment_unavailable_ratio",
    "statefulset_not_ready_ratio",
    "daemonset_misscheduled_ratio",
)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return min(numerator / denominator, 1.0)


def _rate(count: int, window_hours: float) -> float:
    """Events per hour, log-scaled for stability."""
    if window_hours <= 0:
        return 0.0
    per_hour = count / window_hours
    return math.log1p(per_hour)


class FeatureExtractor:
    """Hybrid state + event feature engineering."""

    def extract(self, snapshot: ClusterSnapshot) -> FeatureVector:
        nodes = snapshot.nodes
        pods = snapshot.pods
        events = snapshot.events
        node_denom = max(nodes.total, 1)
        pod_denom = max(pods.total, 1)
        deploy_denom = max(
            snapshot.deployments_total,
            snapshot.deployments_unavailable,
            1,
        )
        sts_denom = max(
            snapshot.statefulsets_total,
            snapshot.statefulsets_not_ready,
            1,
        )
        ds_denom = max(
            snapshot.daemonsets_total,
            snapshot.daemonsets_misscheduled,
            1,
        )
        pressure = nodes.memory_pressure + nodes.disk_pressure + nodes.pid_pressure

        values = (
            _safe_ratio(nodes.not_ready, node_denom),
            _safe_ratio(pressure, node_denom),
            _safe_ratio(pods.failed, pod_denom),
            _safe_ratio(pods.pending, pod_denom),
            _safe_ratio(pods.crash_loop_backoff, pod_denom),
            _safe_ratio(pods.image_pull_backoff, pod_denom),
            _safe_ratio(pods.not_ready, pod_denom),
            _safe_ratio(pods.privileged, pod_denom),
            _safe_ratio(pods.run_as_root, pod_denom),
            _safe_ratio(pods.missing_resource_limits, pod_denom),
            min(pods.oldest_pending_seconds / 3600.0, 168.0) / 168.0,
            _rate(events.warnings, events.window_hours),
            _rate(events.errors, events.window_hours),
            _rate(events.failed_scheduling, events.window_hours),
            _rate(events.backoff, events.window_hours),
            _rate(events.evicted, events.window_hours),
            _rate(events.oom_killed, events.window_hours),
            _safe_ratio(snapshot.deployments_unavailable, deploy_denom),
            _safe_ratio(snapshot.statefulsets_not_ready, sts_denom),
            _safe_ratio(snapshot.daemonsets_misscheduled, ds_denom),
        )
        return FeatureVector(names=FEATURE_NAMES, values=values)
