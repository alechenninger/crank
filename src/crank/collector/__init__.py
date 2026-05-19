"""Cluster data collectors."""

from crank.collector.base import ClusterCollector
from crank.collector.fake import FakeCollector
from crank.collector.kubernetes import KubernetesCollector

__all__ = ["ClusterCollector", "FakeCollector", "KubernetesCollector"]
