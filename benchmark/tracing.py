"""LangFuse tracing — additive observability layer alongside MLflow.

When LANGFUSE_PUBLIC_KEY is not set, all tracing is disabled with zero overhead.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


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
