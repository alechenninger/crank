"""Configuration loading tests."""

from crank.config import DEFAULT_KEYWORD_RULES, load_config
from crank.types import AttentionArea
from pathlib import Path


def test_default_config_has_keywords() -> None:
    cfg = load_config(None)
    assert len(cfg.keyword_rules) == len(DEFAULT_KEYWORD_RULES)


def test_load_yaml_config() -> None:
    path = Path(__file__).resolve().parents[1] / "config" / "crank.yaml"
    cfg = load_config(path)
    assert cfg.event_window_hours == 24
    assert any(r.area == AttentionArea.SECURITY for r in cfg.keyword_rules)
