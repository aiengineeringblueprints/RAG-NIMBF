"""MLflow tracing — OpenTelemetry-based observability via MLflow.

When MLflow tracing is enabled, pipeline functions are traced with
parent-child span relationships. Optionally exports traces via OTLP
to an external OpenTelemetry collector.
"""
from __future__ import annotations

import functools
import logging
import os
from typing import Any, Callable

import mlflow

logger = logging.getLogger(__name__)

_OTEL_EXPORTER_CONFIGURED = False


def setup_tracing() -> str | None:
    """Configure MLflow tracing and optional OTLP exporter.

    Returns the tracking URI if tracing is enabled, None otherwise.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Try connecting to MLflow server; fall back to SQLite
    try:
        import requests
        requests.get(f"{tracking_uri}/api/2.0/mlflow/experiments/search", timeout=2)
    except Exception:
        sqlite_uri = "sqlite:///mlflow.db"
        logger.warning(
            "MLflow server at %s unreachable, falling back to %s",
            tracking_uri, sqlite_uri,
        )
        tracking_uri = sqlite_uri

    mlflow.set_tracking_uri(tracking_uri)
    logger.info("MLflow tracking URI: %s", tracking_uri)

    # Optional OTLP exporter for external OTel backends
    global _OTEL_EXPORTER_CONFIGURED
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if otlp_endpoint and not _OTEL_EXPORTER_CONFIGURED:
        _configure_otlp_exporter(otlp_endpoint)
        _OTEL_EXPORTER_CONFIGURED = True

    return tracking_uri


def _configure_otlp_exporter(endpoint: str) -> None:
    """Configure OpenTelemetry OTLP exporter alongside MLflow tracing."""
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.resources import Resource

        from mlflow.tracing.provider import _get_tracer

        resource = Resource.create({"service.name": "rag-benchmark"})
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        processor = BatchSpanProcessor(exporter)

        tracer = _get_tracer()
        if hasattr(tracer, "provider") and isinstance(tracer.provider, TracerProvider):
            tracer.provider.add_span_processor(processor)
            logger.info("OTLP exporter configured: %s", endpoint)
        else:
            logger.warning("Could not attach OTLP exporter to MLflow tracer")
    except ImportError:
        logger.warning(
            "opentelemetry-exporter-otlp not installed. "
            "Install with: pip install opentelemetry-exporter-otlp"
        )
    except Exception as e:
        logger.warning("Failed to configure OTLP exporter: %s", e)


def _noop_observe(name: str | None = None) -> Callable:
    """Identity decorator used when tracing is disabled."""
    def decorator(fn: Callable) -> Callable:
        return fn
    return decorator


# Module-level decorator: real @observe when tracing is enabled, no-op otherwise.
# For now, always use no-op since LangFuse has been removed.
observe = _noop_observe


def setup_langfuse() -> str | None:
    """Deprecated: LangFuse has been removed.

    This function exists for backward compatibility and always returns None.
    """
    logger.info("LangFuse tracing has been removed, using MLflow tracing instead")
    return None
