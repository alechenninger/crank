"""Domain types for cluster snapshots and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class AttentionArea(StrEnum):
    """High-level operational themes that influence ranking."""

    RELIABILITY = "reliability"
    SECURITY = "security"
    CAPACITY = "capacity"
    COMPLIANCE = "compliance"
    PLATFORM = "platform"


@dataclass(frozen=True)
class ClusterIdentity:
    """Identifies a cluster in multi-cluster rankings."""

    name: str
    context: str | None = None
    api_server: str | None = None


@dataclass
class NodeState:
    """Aggregated node health for a cluster snapshot."""

    total: int = 0
    ready: int = 0
    not_ready: int = 0
    unschedulable: int = 0
    memory_pressure: int = 0
    disk_pressure: int = 0
    pid_pressure: int = 0


@dataclass
class PodState:
    """Aggregated workload health."""

    total: int = 0
    running: int = 0
    pending: int = 0
    failed: int = 0
    succeeded: int = 0
    unknown: int = 0
    crash_loop_backoff: int = 0
    image_pull_backoff: int = 0
    not_ready: int = 0
    privileged: int = 0
    host_network: int = 0
    run_as_root: int = 0
    missing_resource_limits: int = 0
    oldest_pending_seconds: float = 0.0


@dataclass
class EventSummary:
    """Recent Kubernetes event signals (typically last N hours)."""

    window_hours: float = 24.0
    total: int = 0
    warnings: int = 0
    errors: int = 0
    failed_scheduling: int = 0
    backoff: int = 0
    unhealthy: int = 0
    evicted: int = 0
    oom_killed: int = 0
    failed_mount: int = 0
    image_pull_failed: int = 0
    node_not_ready: int = 0
    certificate_expiry: int = 0


@dataclass
class WorkloadState:
    """Aggregated higher-level workload health (deployments, statefulsets, daemonsets)."""

    deployments_total: int = 0
    deployments_unavailable: int = 0
    statefulsets_total: int = 0
    statefulsets_not_ready: int = 0
    daemonsets_total: int = 0
    daemonsets_misscheduled: int = 0


@dataclass
class ClusterSnapshot:
    """Point-in-time cluster state plus recent events."""

    identity: ClusterIdentity
    collected_at: datetime
    nodes: NodeState = field(default_factory=NodeState)
    pods: PodState = field(default_factory=PodState)
    events: EventSummary = field(default_factory=EventSummary)
    workloads: WorkloadState = field(default_factory=WorkloadState)
    namespaces: int = 0
    # Raw text blobs for keyword matching (names, reasons, messages).
    searchable_text: tuple[str, ...] = field(default_factory=tuple)

    @property
    def deployments_total(self) -> int:
        return self.workloads.deployments_total

    @property
    def deployments_unavailable(self) -> int:
        return self.workloads.deployments_unavailable

    @property
    def statefulsets_total(self) -> int:
        return self.workloads.statefulsets_total

    @property
    def statefulsets_not_ready(self) -> int:
        return self.workloads.statefulsets_not_ready

    @property
    def daemonsets_total(self) -> int:
        return self.workloads.daemonsets_total

    @property
    def daemonsets_misscheduled(self) -> int:
        return self.workloads.daemonsets_misscheduled


@dataclass(frozen=True)
class FeatureVector:
    """Named numeric features fed to the scorer."""

    names: tuple[str, ...]
    values: tuple[float, ...]

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.names, self.values, strict=True))


@dataclass
class AreaContribution:
    """How much an attention area contributed to the final score."""

    area: AttentionArea
    score: float
    matched_keywords: tuple[str, ...] = field(default_factory=tuple)


class ScoringMode(StrEnum):
    """How the attention score was produced."""

    HEURISTIC = "heuristic"
    ML = "ml"


@dataclass(frozen=True)
class ScoredCluster:
    """Scored but not yet ranked cluster output."""

    identity: ClusterIdentity
    total_score: float
    base_score: float
    scoring_mode: ScoringMode
    keyword_boost: float
    area_contributions: tuple[AreaContribution, ...]
    top_features: tuple[tuple[str, float], ...]
    summary: str


@dataclass(frozen=True)
class ClusterScore:
    """Final ranked output for one cluster."""

    identity: ClusterIdentity
    rank: int
    total_score: float
    base_score: float
    scoring_mode: ScoringMode
    keyword_boost: float
    area_contributions: tuple[AreaContribution, ...]
    top_features: tuple[tuple[str, float], ...]
    summary: str
