"""Packaged example data paths."""

from __future__ import annotations

from importlib.resources.abc import Traversable

from importlib import resources


def demo_clusters_resource() -> Traversable:
    """Return a Traversable for the bundled demo_clusters.jsonl."""
    return resources.files("crank") / "data" / "demo_clusters.jsonl"
