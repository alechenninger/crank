"""crank command-line interface."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click

from crank.collector.kubernetes import KubernetesCollector
from crank.config import load_config
from crank.logging_config import configure_logging
from crank.report import format_json, format_table
from crank.resources import demo_clusters_path
from crank.scoring.ranker import ClusterRanker
from crank.snapshots import load_snapshots_jsonl
from crank.training import train_from_dataset

logger = logging.getLogger(__name__)


def _resolve_config(path: Path | None) -> Path | None:
    if path is not None and not path.exists():
        raise click.BadParameter(f"config file not found: {path}")
    return path


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
@click.version_option()
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Rank Kubernetes clusters by operator attention (ML + keywords)."""
    configure_logging(verbose=verbose)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


@main.command("score")
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.option("--cluster", "cluster_name", required=True, help="Logical cluster name")
@click.option("--context", default=None, help="kubeconfig context")
@click.option("--kubeconfig", type=click.Path(path_type=Path), default=None)
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table")
@click.option("--demo", is_flag=True, help="Use built-in demo snapshot (no API)")
@click.pass_context
def score(
    ctx: click.Context,
    config_path: Path | None,
    cluster_name: str,
    context: str | None,
    kubeconfig: Path | None,
    fmt: str,
    demo: bool,
) -> None:
    """Score a single cluster from live API or demo mode."""
    exit_code = 0
    try:
        cfg = load_config(_resolve_config(config_path))
        ranker = ClusterRanker(cfg)
        if demo:
            snaps = load_snapshots_jsonl(demo_clusters_path())
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
        if not results:
            click.echo("No clusters scored.", err=True)
            sys.exit(1)
        output = format_json(results) if fmt == "json" else format_table(results)
        click.echo(output)
    except click.ClickException:
        raise
    except Exception as exc:
        logger.exception("score failed")
        click.echo(f"error: {exc}", err=True)
        exit_code = 1
    sys.exit(exit_code)


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
@click.pass_context
def rank(
    ctx: click.Context,
    config_path: Path | None,
    clusters: str,
    kubeconfig: Path | None,
    fmt: str,
    demo: bool,
) -> None:
    """Rank multiple clusters (highest attention first)."""
    exit_code = 0
    try:
        cfg = load_config(_resolve_config(config_path))
        ranker = ClusterRanker(cfg)
        if demo:
            results = ranker.rank_snapshots(load_snapshots_jsonl(demo_clusters_path()))
        elif not clusters:
            raise click.UsageError("--clusters is required unless --demo is set")
        else:
            try:
                mapping: dict[str, str | None] = json.loads(clusters)
            except json.JSONDecodeError as exc:
                raise click.BadParameter(f"invalid --clusters JSON: {exc}") from exc
            collectors = [
                KubernetesCollector(
                    cluster_name=name,
                    context=ctx_name,
                    event_window_hours=cfg.event_window_hours,
                    kubeconfig=str(kubeconfig) if kubeconfig else None,
                )
                for name, ctx_name in mapping.items()
            ]
            results = ranker.rank(collectors)
        if not results:
            click.echo("No clusters scored.", err=True)
            sys.exit(1)
        output = format_json(results) if fmt == "json" else format_table(results)
        click.echo(output)
    except click.ClickException:
        raise
    except Exception as exc:
        logger.exception("rank failed")
        click.echo(f"error: {exc}", err=True)
        exit_code = 1
    sys.exit(exit_code)


@main.command("train")
@click.option("--dataset", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--output", type=click.Path(path_type=Path), required=True)
def train(dataset: Path, output: Path) -> None:
    """Train ML model from labeled JSONL snapshots."""
    exit_code = 0
    try:
        metrics = train_from_dataset(dataset, output)
        click.echo(f"Model written to {output}")
        if metrics:
            click.echo(
                f"validation: mae={metrics.get('mae', 0):.2f} r2={metrics.get('r2', 0):.3f}"
            )
    except Exception as exc:
        logger.exception("train failed")
        click.echo(f"error: {exc}", err=True)
        exit_code = 1
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
