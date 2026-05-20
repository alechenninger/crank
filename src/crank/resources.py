"""Packaged example data paths."""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def demo_clusters_path() -> Path:
    """Return path to bundled demo_clusters.jsonl."""
    ref = resources.files("crank") / "data" / "demo_clusters.jsonl"
    with resources.as_file(ref) as path:
        return path
