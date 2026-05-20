"""Paginated Kubernetes list helpers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, TypeVar

T = TypeVar("T")


def list_all_pages(
    list_fn: Callable[..., Any],
    *,
    limit: int = 500,
    **kwargs: Any,
) -> Iterator[T]:
    """Yield all items from a paginated Kubernetes list API call."""
    continue_token: str | None = None
    while True:
        if continue_token:
            page = list_fn(_continue=continue_token, limit=limit, **kwargs)
        else:
            page = list_fn(limit=limit, **kwargs)
        yield from page.items
        continue_token = page.metadata._continue  # noqa: SLF001
        if not continue_token:
            break
