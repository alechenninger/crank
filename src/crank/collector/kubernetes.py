"""Live Kubernetes API collector (state + events)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kubernetes import client, config
from kubernetes.client import CoreV1Api

from crank.collector.base import ClusterCollector
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
)


def _container_waiting_reason(pod) -> str | None:
    for cs in pod.status.container_statuses or []:
        if cs.state and cs.state.waiting:
            return cs.state.waiting.reason
    return None


def _pod_not_ready(pod) -> bool:
    for cond in pod.status.conditions or []:
        if cond.type == "Ready" and cond.status != "True":
            return True
    return False


def _runs_as_root(pod) -> bool:
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


def _missing_limits(pod) -> bool:
    for c in pod.spec.containers or []:
        if not c.resources or not c.resources.limits:
            return True
    return False


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
        if kubeconfig:
            config.load_kube_config(config_file=kubeconfig, context=context)
        else:
            try:
                config.load_incluster_config()
            except config.ConfigException:
                config.load_kube_config(context=context)
        self._core: CoreV1Api = client.CoreV1Api()
        self._apps = client.AppsV1Api()
        self._identity = ClusterIdentity(
            name=cluster_name,
            context=context,
        )
        self._event_window_hours = event_window_hours

    def collect(self) -> ClusterSnapshot:
        now = datetime.now(UTC)
        nodes = self._collect_nodes()
        pods, pod_text = self._collect_pods(now)
        events, event_text = self._collect_events(now)
        namespaces = len(self._core.list_namespace().items)
        deploy_unavail, sts_not_ready, ds_mis = self._collect_workloads()
        searchable = tuple(pod_text + event_text)
        return ClusterSnapshot(
            identity=self._identity,
            collected_at=now,
            nodes=nodes,
            pods=pods,
            events=events,
            namespaces=namespaces,
            deployments_unavailable=deploy_unavail,
            statefulsets_not_ready=sts_not_ready,
            daemonsets_misscheduled=ds_mis,
            searchable_text=searchable,
        )

    def _collect_nodes(self) -> NodeState:
        state = NodeState()
        for node in self._core.list_node().items:
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
        for pod in self._core.list_pod_for_all_namespaces().items:
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
                    age = (now - created.replace(tzinfo=UTC)).total_seconds()
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
        # Core events API remains the practical source across cluster versions.
        for event in self._core.list_event_for_all_namespaces().items:
            ts = event.last_timestamp or event.event_time or event.first_timestamp
            if ts and ts.replace(tzinfo=UTC) < cutoff:
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
            if "oom" in reason.lower() or "oomkilled" in message:
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

    def _collect_workloads(self) -> tuple[int, int, int]:
        deploy_unavail = 0
        for d in self._apps.list_deployment_for_all_namespaces().items:
            desired = d.spec.replicas or 0
            avail = d.status.available_replicas or 0
            if avail < desired:
                deploy_unavail += 1
        sts_not_ready = 0
        for s in self._apps.list_stateful_set_for_all_namespaces().items:
            ready = s.status.ready_replicas or 0
            desired = s.spec.replicas or 0
            if ready < desired:
                sts_not_ready += 1
        ds_mis = 0
        for ds in self._apps.list_daemon_set_for_all_namespaces().items:
            if (ds.status.number_misscheduled or 0) > 0:
                ds_mis += 1
        return deploy_unavail, sts_not_ready, ds_mis
