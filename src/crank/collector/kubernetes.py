"""Live Kubernetes API collector (state + events)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from kubernetes import client, config
from kubernetes.client import ApiClient, AppsV1Api, Configuration, CoreV1Api
from kubernetes.config.incluster_config import InClusterConfigLoader

from crank.collector.pagination import list_all_pages
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
    WorkloadState,
)

logger = logging.getLogger(__name__)


def _container_waiting_reason(pod: Any) -> str | None:
    for cs in pod.status.container_statuses or []:
        if cs.state and cs.state.waiting:
            reason: str | None = cs.state.waiting.reason
            return reason
    return None


def _pod_not_ready(pod: Any) -> bool:
    for cond in pod.status.conditions or []:
        if cond.type == "Ready" and cond.status != "True":
            return True
    return False


def _runs_as_root(pod: Any) -> bool:
    spec = pod.spec
    if spec is None:
        return False
    if spec.security_context and spec.security_context.run_as_non_root is False:
        return True
    for c in spec.containers or []:
        sc = c.security_context
        if sc and sc.run_as_user == 0:
            return True
        if sc and sc.run_as_non_root is False:
            return True
    return False


def _missing_limits(pod: Any) -> bool:
    for c in pod.spec.containers or []:
        if not c.resources or not c.resources.limits:
            return True
    return False


def _to_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC)


def _create_api_clients(
    *,
    kubeconfig: str | None,
    context: str | None,
) -> tuple[CoreV1Api, AppsV1Api]:
    """Create API clients without mutating the global default configuration."""
    api_client: ApiClient
    if kubeconfig:
        api_client = config.new_client_from_config(
            config_file=kubeconfig,
            context=context,
        )
    else:
        try:
            cfg = Configuration()
            InClusterConfigLoader(
                token_filename=None, cert_filename=None
            ).load_and_set(client_configuration=cfg)
            api_client = ApiClient(configuration=cfg)
        except config.ConfigException:
            api_client = config.new_client_from_config(context=context)
    return CoreV1Api(api_client), AppsV1Api(api_client)


class KubernetesCollector:
    """Hybrid collector: current resource state plus recent events."""

    def __init__(
        self,
        *,
        cluster_name: str,
        context: str | None = None,
        event_window_hours: float = 24.0,
        kubeconfig: str | None = None,
    ) -> None:
        self._core, self._apps = _create_api_clients(
            kubeconfig=kubeconfig,
            context=context,
        )
        self._identity = ClusterIdentity(
            name=cluster_name,
            context=context,
        )
        self._event_window_hours = event_window_hours

    def collect(self) -> ClusterSnapshot:
        now = datetime.now(UTC)
        logger.info("Collecting snapshot for cluster %s", self._identity.name)
        nodes = self._collect_nodes()
        pods, pod_text = self._collect_pods(now)
        events, event_text = self._collect_events(now)
        namespaces = sum(1 for _ in list_all_pages(self._core.list_namespace))
        workloads = self._collect_workloads()
        searchable = tuple(pod_text + event_text)
        return ClusterSnapshot(
            identity=self._identity,
            collected_at=now,
            nodes=nodes,
            pods=pods,
            events=events,
            workloads=workloads,
            namespaces=namespaces,
            searchable_text=searchable,
        )

    def _collect_nodes(self) -> NodeState:
        state = NodeState()
        for node in list_all_pages(self._core.list_node):  # type: ignore[var-annotated]
            state.total += 1
            ready = False
            for cond in node.status.conditions or []:
                if cond.type == "Ready" and cond.status == "True":
                    ready = True
                if cond.type == "MemoryPressure" and cond.status == "True":
                    state.memory_pressure += 1
                if cond.type == "DiskPressure" and cond.status == "True":
                    state.disk_pressure += 1
                if cond.type == "PIDPressure" and cond.status == "True":
                    state.pid_pressure += 1
            if ready:
                state.ready += 1
            else:
                state.not_ready += 1
            if node.spec and node.spec.unschedulable:
                state.unschedulable += 1
        return state

    def _collect_pods(self, now: datetime) -> tuple[PodState, list[str]]:
        state = PodState()
        text: list[str] = []
        for pod in list_all_pages(self._core.list_pod_for_all_namespaces):  # type: ignore[var-annotated]
            state.total += 1
            phase = pod.status.phase or "Unknown"
            ns = pod.metadata.namespace or ""
            name = pod.metadata.name or ""
            text.append(f"{ns}/{name}")
            labels = " ".join(f"{k}={v}" for k, v in (pod.metadata.labels or {}).items())
            if labels:
                text.append(labels)

            if phase == "Running":
                state.running += 1
            elif phase == "Pending":
                state.pending += 1
                created = pod.metadata.creation_timestamp
                if created:
                    age = (now - _to_utc(created)).total_seconds()
                    state.oldest_pending_seconds = max(state.oldest_pending_seconds, age)
            elif phase == "Failed":
                state.failed += 1
            elif phase == "Succeeded":
                state.succeeded += 1
            else:
                state.unknown += 1

            reason = _container_waiting_reason(pod)
            if reason == "CrashLoopBackOff":
                state.crash_loop_backoff += 1
                text.append("crashloop")
            elif reason == "ImagePullBackOff":
                state.image_pull_backoff += 1
                text.append("imagepull")

            if _pod_not_ready(pod):
                state.not_ready += 1
            if pod.spec and pod.spec.host_network:
                state.host_network += 1
                text.append("hostnetwork")
            if _runs_as_root(pod):
                state.run_as_root += 1
                text.append("runasroot")
            if _missing_limits(pod):
                state.missing_resource_limits += 1
            for c in pod.spec.containers or []:
                sc = c.security_context
                if sc and sc.privileged:
                    state.privileged += 1
                    text.append("privileged")
                    break
        return state, text

    def _collect_events(self, now: datetime) -> tuple[EventSummary, list[str]]:
        cutoff = now - timedelta(hours=self._event_window_hours)
        summary = EventSummary(window_hours=self._event_window_hours)
        text: list[str] = []
        for event in list_all_pages(self._core.list_event_for_all_namespaces):  # type: ignore[var-annotated]
            ts = event.last_timestamp or event.event_time or event.first_timestamp
            if ts and _to_utc(ts) < cutoff:
                continue
            summary.total += 1
            etype = (event.type or "").lower()
            reason = (event.reason or "").lower()
            message = (event.message or "").lower()
            blob = f"{reason} {message}"
            text.append(blob)

            if etype == "warning":
                summary.warnings += 1
            if "error" in etype or "error" in message:
                summary.errors += 1
            if "failedscheduling" in reason:
                summary.failed_scheduling += 1
            if "backoff" in reason:
                summary.backoff += 1
            if reason == "unhealthy":
                summary.unhealthy += 1
            if "evict" in reason or "evicted" in message:
                summary.evicted += 1
            if "oom" in reason or "oomkilled" in message:
                summary.oom_killed += 1
            if "failedmount" in reason:
                summary.failed_mount += 1
            if "errimagepull" in reason or "imagepullbackoff" in reason:
                summary.image_pull_failed += 1
            if "node not ready" in message or reason == "node_not_ready":
                summary.node_not_ready += 1
            if "certificate" in message and "expir" in message:
                summary.certificate_expiry += 1
        return summary, text

    def _collect_workloads(self) -> WorkloadState:
        state = WorkloadState()
        for d in list_all_pages(self._apps.list_deployment_for_all_namespaces):  # type: ignore[var-annotated]
            state.deployments_total += 1
            desired = d.spec.replicas or 0
            avail = d.status.available_replicas or 0
            if avail < desired:
                state.deployments_unavailable += 1
        for s in list_all_pages(self._apps.list_stateful_set_for_all_namespaces):  # type: ignore[var-annotated]
            state.statefulsets_total += 1
            ready = s.status.ready_replicas or 0
            desired = s.spec.replicas or 0
            if ready < desired:
                state.statefulsets_not_ready += 1
        for ds in list_all_pages(self._apps.list_daemon_set_for_all_namespaces):  # type: ignore[var-annotated]
            state.daemonsets_total += 1
            if (ds.status.number_misscheduled or 0) > 0:
                state.daemonsets_misscheduled += 1
        return state
