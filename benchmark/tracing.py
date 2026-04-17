"""LangFuse tracing — additive observability layer alongside MLflow.

When LANGFUSE_PUBLIC_KEY is not set, all tracing is disabled with zero overhead.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)


def _noop_observe(name: str | None = None) -> Callable:
    """Identity decorator used when LangFuse is disabled."""
    def decorator(fn: Callable) -> Callable:
        return fn
    return decorator


def _get_observe():
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        from langfuse import observe
        return observe
    return _noop_observe


# Module-level decorator: real @observe when key is set, no-op otherwise.
observe = _get_observe()


def setup_langfuse() -> str | None:
    """Configure LangFuse from env vars.

    Returns the host URL if tracing is enabled, None otherwise.
    The SDK reads env vars lazily — no explicit Client() needed.
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        logger.info("LangFuse tracing disabled (LANGFUSE_PUBLIC_KEY not set)")
        return None

    host = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    logger.info("LangFuse tracing enabled: host=%s", host)
    return host
