"""Configuration loading tests."""

from pathlib import Path

from crank.config import DEFAULT_KEYWORD_RULES, load_config, merge_keyword_rules
from crank.types import AttentionArea


def test_default_config_has_keywords() -> None:
    cfg = load_config(None)
    assert len(cfg.keyword_rules) == len(DEFAULT_KEYWORD_RULES)


def test_load_yaml_config_merges_keywords() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "crank.yaml"
    cfg = load_config(path)
    assert cfg.event_window_hours == 24
    assert any(r.area == AttentionArea.SECURITY for r in cfg.keyword_rules)
    # YAML has 5 patterns; merge keeps defaults too.
    assert len(cfg.keyword_rules) >= len(DEFAULT_KEYWORD_RULES)


def test_yaml_override_replaces_default_weight() -> None:
    from crank.config import KeywordRule

    merged = merge_keyword_rules(
        [KeywordRule("oom", AttentionArea.CAPACITY, weight=99.0)]
    )
    oom = next(r for r in merged if r.pattern == "oom")
    assert oom.weight == 99.0
