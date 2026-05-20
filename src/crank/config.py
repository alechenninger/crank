"""Configuration loading for crank scoring."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from crank.types import AttentionArea

logger = logging.getLogger(__name__)


@dataclass
class KeywordRule:
    """Keyword or regex-like substring that boosts an attention area."""

    pattern: str
    area: AttentionArea
    weight: float = 1.0
    case_sensitive: bool = False


def _default_keyword_rules() -> list[KeywordRule]:
    return list(DEFAULT_KEYWORD_RULES)


@dataclass
class ScoringConfig:
    """Tunable scoring parameters."""

    event_window_hours: float = 24.0
    keyword_rules: list[KeywordRule] = field(default_factory=_default_keyword_rules)
    keyword_boost_cap: float = 25.0
    model_path: Path | None = None
    # Blend: final = ml_weight * ml + (1 - ml_weight) * heuristic when a model exists.
    ml_weight: float = 0.7


DEFAULT_KEYWORD_RULES: list[KeywordRule] = [
    KeywordRule("crashloop", AttentionArea.RELIABILITY, 2.0),
    KeywordRule("oom", AttentionArea.CAPACITY, 2.5),
    KeywordRule("evict", AttentionArea.CAPACITY, 2.0),
    KeywordRule("privileged", AttentionArea.SECURITY, 3.0),
    KeywordRule("hostpath", AttentionArea.SECURITY, 2.0),
    KeywordRule("cert", AttentionArea.COMPLIANCE, 2.0),
    KeywordRule("expir", AttentionArea.COMPLIANCE, 2.5),
    KeywordRule("backoff", AttentionArea.RELIABILITY, 1.5),
    KeywordRule("failedscheduling", AttentionArea.CAPACITY, 2.0),
    KeywordRule("notready", AttentionArea.RELIABILITY, 2.0),
    KeywordRule("vault", AttentionArea.SECURITY, 1.5),
    KeywordRule("istio", AttentionArea.PLATFORM, 1.0),
    KeywordRule("cilium", AttentionArea.PLATFORM, 1.0),
    KeywordRule("calico", AttentionArea.PLATFORM, 1.0),
    KeywordRule("prometheus", AttentionArea.PLATFORM, 1.0),
    KeywordRule("argocd", AttentionArea.PLATFORM, 1.5),
    KeywordRule("flux", AttentionArea.PLATFORM, 1.5),
    KeywordRule("pci", AttentionArea.COMPLIANCE, 3.0),
    KeywordRule("hipaa", AttentionArea.COMPLIANCE, 3.0),
    KeywordRule("prod", AttentionArea.RELIABILITY, 1.5),
    KeywordRule("payment", AttentionArea.RELIABILITY, 2.5),
]


def _parse_area(value: str) -> AttentionArea:
    try:
        return AttentionArea(value.lower())
    except ValueError as exc:
        raise ValueError(f"unknown attention area: {value}") from exc


def _rule_key(rule: KeywordRule) -> str:
    return rule.pattern if rule.case_sensitive else rule.pattern.lower()


def merge_keyword_rules(yaml_rules: list[KeywordRule]) -> list[KeywordRule]:
    """Merge YAML rules onto defaults; YAML entries override matching patterns."""
    merged: dict[str, KeywordRule] = {_rule_key(r): r for r in DEFAULT_KEYWORD_RULES}
    for rule in yaml_rules:
        merged[_rule_key(rule)] = rule
    return list(merged.values())


def discover_config_path() -> Path | None:
    """Return the first existing default config file path, if any."""
    candidates = (
        Path("config/crank.yaml"),
        Path.home() / ".config" / "crank" / "config.yaml",
    )
    for candidate in candidates:
        if candidate.exists():
            logger.debug("Using config file %s", candidate)
            return candidate
    return None


def load_config(path: Path | None) -> ScoringConfig:
    """Load YAML config, discover defaults, or return in-code defaults."""
    if path is None:
        path = discover_config_path()
    if path is None or not path.exists():
        return ScoringConfig(keyword_rules=list(DEFAULT_KEYWORD_RULES))

    with path.open(encoding="utf-8") as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    yaml_rules: list[KeywordRule] = []
    for item in raw.get("keywords", []):
        yaml_rules.append(
            KeywordRule(
                pattern=item["pattern"],
                area=_parse_area(item["area"]),
                weight=float(item.get("weight", 1.0)),
                case_sensitive=bool(item.get("case_sensitive", False)),
            )
        )
    rules = merge_keyword_rules(yaml_rules) if yaml_rules else list(DEFAULT_KEYWORD_RULES)

    scoring = raw.get("scoring", {})
    return ScoringConfig(
        event_window_hours=float(raw.get("event_window_hours", 24.0)),
        keyword_rules=rules,
        keyword_boost_cap=float(scoring.get("keyword_boost_cap", 25.0)),
        model_path=Path(scoring["model_path"]) if scoring.get("model_path") else None,
        ml_weight=float(scoring.get("ml_weight", 0.7)),
    )
