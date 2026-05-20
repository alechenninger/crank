"""Kubernetes collector tests with fake API clients."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from crank.collector.kubernetes import KubernetesCollector


def _list_response(items: list[Any]) -> SimpleNamespace:
    return SimpleNamespace(items=items, metadata=SimpleNamespace(_continue=None))


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


@patch("crank.collector.kubernetes._create_api_clients")
def test_collect_aggregates_nodes_pods_and_workloads(mock_clients: MagicMock) -> None:
    core = MagicMock()
    apps = MagicMock()
    mock_clients.return_value = (core, apps)

    core.list_node.side_effect = [_list_response([_node(ready=False), _node()])]
    core.list_pod_for_all_namespaces.side_effect = [
        _list_response([_pod(reason="CrashLoopBackOff")])
    ]
    core.list_event_for_all_namespaces.side_effect = [
        _list_response([_event("OOMKilling", "oomkilled")])
    ]
    core.list_namespace.side_effect = [_list_response([SimpleNamespace()])]
    apps.list_deployment_for_all_namespaces.side_effect = [
        _list_response([_deployment(desired=3, available=1)])
    ]
    apps.list_stateful_set_for_all_namespaces.side_effect = [_list_response([])]
    apps.list_daemon_set_for_all_namespaces.side_effect = [_list_response([])]

    collector = KubernetesCollector(cluster_name="test")
    snap = collector.collect()

    assert snap.identity.name == "test"
    assert snap.nodes.total == 2
    assert snap.nodes.not_ready == 1
    assert snap.pods.crash_loop_backoff == 1
    assert snap.events.oom_killed == 1
    assert snap.deployments_total == 1
    assert snap.deployments_unavailable == 1
