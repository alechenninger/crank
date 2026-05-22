"""Kubernetes collector tests with fake API clients."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from crank.collector.kubernetes import KubernetesCollector


class _FakeListResponse:
    """Mimics the Kubernetes list API response with pagination support."""

    def __init__(self, items: list[Any], continue_token: str | None = None) -> None:
        self.items = items
        self.metadata = SimpleNamespace(_continue=continue_token)


class FakeApi:
    """In-memory fake for CoreV1Api / AppsV1Api list methods.

    Register items for each list method, then pass to the collector via patch.
    """

    def __init__(self) -> None:
        self._resources: dict[str, list[Any]] = {}

    def set_resources(self, method_name: str, items: list[Any]) -> None:
        self._resources[method_name] = items

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        items = self._resources.get(name, [])

        def list_fn(**_kwargs: Any) -> _FakeListResponse:
            return _FakeListResponse(items)

        return list_fn


def _node(ready: bool = True, memory_pressure: bool = False) -> SimpleNamespace:
    conditions = [
        SimpleNamespace(type="Ready", status="True" if ready else "False"),
    ]
    if memory_pressure:
        conditions.append(SimpleNamespace(type="MemoryPressure", status="True"))
    return SimpleNamespace(
        status=SimpleNamespace(conditions=conditions),
        spec=SimpleNamespace(unschedulable=False),
    )


def _pod(
    *,
    phase: str = "Running",
    reason: str | None = None,
    namespace: str = "default",
    name: str = "p1",
) -> SimpleNamespace:
    waiting = None
    if reason:
        waiting = SimpleNamespace(reason=reason)
    container_statuses = [SimpleNamespace(state=SimpleNamespace(waiting=waiting))]
    return SimpleNamespace(
        metadata=SimpleNamespace(
            namespace=namespace,
            name=name,
            labels={},
            creation_timestamp=datetime.now(UTC),
        ),
        status=SimpleNamespace(
            phase=phase,
            container_statuses=container_statuses,
            conditions=[SimpleNamespace(type="Ready", status="True")],
        ),
        spec=SimpleNamespace(
            host_network=False,
            containers=[SimpleNamespace(security_context=None, resources=None)],
            security_context=None,
        ),
    )


def _deployment(desired: int = 3, available: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(replicas=desired),
        status=SimpleNamespace(available_replicas=available),
    )


def _statefulset(desired: int = 3, ready: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        spec=SimpleNamespace(replicas=desired),
        status=SimpleNamespace(ready_replicas=ready),
    )


def _daemonset(misscheduled: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        status=SimpleNamespace(number_misscheduled=misscheduled),
    )


def _event(reason: str, message: str = "", hours_ago: float = 1.0) -> SimpleNamespace:
    ts = datetime.now(UTC) - timedelta(hours=hours_ago)
    return SimpleNamespace(
        type="Warning",
        reason=reason,
        message=message,
        last_timestamp=ts,
        event_time=None,
        first_timestamp=None,
    )


def _make_collector(
    core: FakeApi,
    apps: FakeApi,
    cluster_name: str = "test",
) -> KubernetesCollector:
    with patch("crank.collector.kubernetes._create_api_clients") as mock:
        mock.return_value = (core, apps)
        return KubernetesCollector(cluster_name=cluster_name)


@patch("crank.collector.kubernetes._create_api_clients")
def test_collect_aggregates_nodes_pods_and_workloads(mock_clients: Any) -> None:
    core = FakeApi()
    apps = FakeApi()
    mock_clients.return_value = (core, apps)

    core.set_resources("list_node", [_node(ready=False), _node()])
    core.set_resources("list_pod_for_all_namespaces", [_pod(reason="CrashLoopBackOff")])
    core.set_resources("list_event_for_all_namespaces", [_event("OOMKilling", "oomkilled")])
    core.set_resources("list_namespace", [SimpleNamespace()])
    apps.set_resources("list_deployment_for_all_namespaces", [_deployment(desired=3, available=1)])

    collector = KubernetesCollector(cluster_name="test")
    snap = collector.collect()

    assert snap.identity.name == "test"
    assert snap.nodes.total == 2
    assert snap.nodes.not_ready == 1
    assert snap.pods.crash_loop_backoff == 1
    assert snap.events.oom_killed == 1
    assert snap.deployments_total == 1
    assert snap.deployments_unavailable == 1
    assert snap.statefulsets_total == 0
    assert snap.daemonsets_total == 0


@patch("crank.collector.kubernetes._create_api_clients")
def test_collect_statefulsets_and_daemonsets(mock_clients: Any) -> None:
    core = FakeApi()
    apps = FakeApi()
    mock_clients.return_value = (core, apps)

    core.set_resources("list_node", [_node()])
    core.set_resources("list_pod_for_all_namespaces", [_pod()])
    core.set_resources("list_event_for_all_namespaces", [])
    core.set_resources("list_namespace", [SimpleNamespace()])
    apps.set_resources(
        "list_stateful_set_for_all_namespaces",
        [_statefulset(desired=5, ready=3), _statefulset(desired=2, ready=2)],
    )
    apps.set_resources(
        "list_daemon_set_for_all_namespaces",
        [_daemonset(misscheduled=2), _daemonset(misscheduled=0)],
    )

    collector = KubernetesCollector(cluster_name="test-workloads")
    snap = collector.collect()

    assert snap.statefulsets_total == 2
    assert snap.statefulsets_not_ready == 1
    assert snap.daemonsets_total == 2
    assert snap.daemonsets_misscheduled == 1
