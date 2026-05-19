"""crank command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import click

from crank.collector.fake import FakeCollector
from crank.collector.kubernetes import KubernetesCollector
from crank.config import load_config
from crank.report import format_json, format_table
from crank.scoring.ranker import ClusterRanker
from crank.training import train_from_dataset
from crank.types import (
    ClusterIdentity,
    ClusterSnapshot,
    EventSummary,
    NodeState,
    PodState,
)


def _load_demo_snapshots(path: Path) -> list[ClusterSnapshot]:
    snapshots: list[ClusterSnapshot] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            name = data.pop("name", "demo")
            identity = ClusterIdentity(name=name, context=data.pop("context", None))
            snapshots.append(
                ClusterSnapshot(
                    identity=identity,
                    collected_at=data.get("collected_at"),  # type: ignore[arg-type]
                    nodes=NodeState(**data.get("nodes", {})),
                    pods=PodState(**data.get("pods", {})),
                    events=EventSummary(**data.get("events", {})),
                    namespaces=int(data.get("namespaces", 1)),
                    deployments_unavailable=int(data.get("deployments_unavailable", 0)),
                    statefulsets_not_ready=int(data.get("statefulsets_not_ready", 0)),
                    daemonsets_misscheduled=int(data.get("daemonsets_misscheduled", 0)),
                    searchable_text=tuple(data.get("searchable_text", [])),
                )
            )
    return snapshots


@click.group()
@click.version_option()
def main() -> None:
    """Rank Kubernetes clusters by operator attention (ML + keywords)."""


@main.command("score")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--cluster", "cluster_name", required=True, help="Logical cluster name")
@click.option("--context", default=None, help="kubeconfig context")
@click.option("--kubeconfig", type=click.Path(path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--demo", is_flag=True, help="Use built-in demo snapshot (no API)")
def score(
    config_path: Path | None,
    cluster_name: str,
    context: str | None,
    kubeconfig: Path | None,
    fmt: str,
    demo: bool,
) -> None:
    """Score a single cluster from live API or demo mode."""
    cfg = load_config(config_path)
    ranker = ClusterRanker(cfg)
    if demo:
        demo_path = Path(__file__).resolve().parents[2] / "examples" / "demo_clusters.jsonl"
        snaps = _load_demo_snapshots(demo_path)
        match = next((s for s in snaps if s.identity.name == cluster_name), snaps[0])
        results = ranker.rank_snapshots([match])
    else:
        collector = KubernetesCollector(
            cluster_name=cluster_name,
            context=context,
            event_window_hours=cfg.event_window_hours,
            kubeconfig=str(kubeconfig) if kubeconfig else None,
        )
        results = ranker.rank([collector])
    output = format_json(results) if fmt == "json" else format_table(results)
    click.echo(output)


@main.command("rank")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option(
    "--clusters",
    default=None,
    help='JSON map name->context, e.g. \'{"prod":"ctx1","staging":"ctx2"}\'',
)
@click.option("--kubeconfig", type=click.Path(path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--demo", is_flag=True, help="Rank example clusters from examples/")
def rank(
    config_path: Path | None,
    clusters: str,
    kubeconfig: Path | None,
    fmt: str,
    demo: bool,
) -> None:
    """Rank multiple clusters (highest attention first)."""
    cfg = load_config(config_path)
    ranker = ClusterRanker(cfg)
    if demo:
        demo_path = Path(__file__).resolve().parents[2] / "examples" / "demo_clusters.jsonl"
        results = ranker.rank_snapshots(_load_demo_snapshots(demo_path))
    elif not clusters:
        raise click.UsageError("--clusters is required unless --demo is set")
    else:
        mapping: dict[str, str | None] = json.loads(clusters)
        collectors = [
            KubernetesCollector(
                cluster_name=name,
                context=ctx,
                event_window_hours=cfg.event_window_hours,
                kubeconfig=str(kubeconfig) if kubeconfig else None,
            )
            for name, ctx in mapping.items()
        ]
        results = ranker.rank(collectors)
    output = format_json(results) if fmt == "json" else format_table(results)
    click.echo(output)


@main.command("train")
@click.option("--dataset", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def train(dataset: Path, output: Path) -> None:
    """Train ML model from labeled JSONL snapshots."""
    train_from_dataset(dataset, output)
    click.echo(f"Model written to {output}")


if __name__ == "__main__":
    main()
