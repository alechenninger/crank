"""Logging setup for crank."""

from __future__ import annotations

import logging
import sys


def configure_logging(*, verbose: bool = False) -> None:
    """Configure root logger for CLI and library use."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )
