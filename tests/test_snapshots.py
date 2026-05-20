"""Snapshot loading tests."""

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
