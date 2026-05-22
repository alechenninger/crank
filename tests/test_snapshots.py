"""Snapshot loading tests."""

import logging
from pathlib import Path

from crank.snapshots import load_snapshots_jsonl, snapshot_from_dict


def test_snapshot_from_dict_defaults_collected_at() -> None:
    snap = snapshot_from_dict({"name": "c"})
    assert snap.collected_at.tzinfo is not None


def test_load_demo_jsonl() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "demo_clusters.jsonl"
    snaps = load_snapshots_jsonl(path)
    assert len(snaps) == 4
    assert snaps[0].deployments_total >= snaps[0].deployments_unavailable


def test_parse_collected_at_string() -> None:
    snap = snapshot_from_dict(
        {"name": "c", "collected_at": "2024-01-15T12:00:00+00:00"}
    )
    assert snap.collected_at.year == 2024


def test_snapshot_from_dict_preserves_all_workload_fields() -> None:
    data = {
        "name": "full",
        "context": "ctx",
        "nodes": {"total": 10, "ready": 8, "not_ready": 2, "memory_pressure": 1},
        "pods": {"total": 50, "running": 45, "crash_loop_backoff": 3, "privileged": 1},
        "events": {"window_hours": 12, "warnings": 20, "oom_killed": 2},
        "namespaces": 5,
        "deployments_total": 30,
        "deployments_unavailable": 4,
        "statefulsets_total": 10,
        "statefulsets_not_ready": 2,
        "daemonsets_total": 8,
        "daemonsets_misscheduled": 1,
        "searchable_text": ["prod payment"],
    }
    snap = snapshot_from_dict(data)
    assert snap.identity.name == "full"
    assert snap.identity.context == "ctx"
    assert snap.nodes.total == 10
    assert snap.nodes.not_ready == 2
    assert snap.nodes.memory_pressure == 1
    assert snap.pods.total == 50
    assert snap.pods.crash_loop_backoff == 3
    assert snap.pods.privileged == 1
    assert snap.events.window_hours == 12
    assert snap.events.oom_killed == 2
    assert snap.namespaces == 5
    assert snap.deployments_total == 30
    assert snap.deployments_unavailable == 4
    assert snap.statefulsets_total == 10
    assert snap.statefulsets_not_ready == 2
    assert snap.daemonsets_total == 8
    assert snap.daemonsets_misscheduled == 1
    assert snap.searchable_text == ("prod payment",)


def test_snapshot_from_dict_warns_on_unknown_keys(caplog: logging.Handler) -> None:
    """Misspelled field names should produce a warning, not silently vanish."""
    with caplog.at_level(logging.WARNING, logger="crank.snapshots"):  # type: ignore[union-attr]
        snapshot_from_dict({"name": "c", "deplyoments_total": 5})
    assert any("deplyoments_total" in rec.message for rec in caplog.records)  # type: ignore[union-attr]


def test_snapshot_from_dict_warns_on_training_fields(caplog: logging.Handler) -> None:
    """Training-specific fields (session, rank) now trigger a warning."""
    with caplog.at_level(logging.WARNING, logger="crank.snapshots"):  # type: ignore[union-attr]
        snap = snapshot_from_dict({
            "name": "c",
            "session": "2026-05-19-triage",
            "rank": 1,
            "nodes": {"total": 5},
        })
    assert snap.identity.name == "c"
    assert snap.nodes.total == 5
    assert any("rank" in rec.message for rec in caplog.records)  # type: ignore[union-attr]
