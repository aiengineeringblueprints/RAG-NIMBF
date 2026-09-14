"""Compatibility helpers for legacy and managed RAG adapters."""

from __future__ import annotations

from typing import Any

from benchmark.adapters.base import (
    AdapterCapabilities,
    AdapterGenerationResult,
    PreparedTarget,
    RagSystemOutput,
    RetrievalResult,
)


class UnsupportedAdapterCapability(ValueError):
    """Raised when an experiment requires a feature the adapter lacks."""


def get_adapter_capabilities(adapter: Any) -> AdapterCapabilities:
    """Return explicit capabilities, with safe defaults for old adapters."""
    method = getattr(adapter, "capabilities", None)
    if method is None:
        return AdapterCapabilities(
            generation=callable(getattr(adapter, "answer", None))
        )
    capabilities = method()
    if not isinstance(capabilities, AdapterCapabilities):
        raise TypeError(
            f"{adapter.name}.capabilities() must return AdapterCapabilities, "
            f"got {type(capabilities).__name__}"
        )
    return capabilities


def require_adapter_capabilities(adapter: Any, *required: str) -> None:
    """Fail visibly when a requested provider feature is unsupported."""
    capabilities = get_adapter_capabilities(adapter)
    unknown = [name for name in required if not hasattr(capabilities, name)]
    if unknown:
        raise ValueError(f"Unknown adapter capabilities: {', '.join(unknown)}")
    unsupported = [name for name in required if not getattr(capabilities, name)]
    if unsupported:
        raise UnsupportedAdapterCapability(
            f"Adapter {adapter.name!r} does not support: {', '.join(unsupported)}"
        )


def prepare_adapter(
    adapter: Any,
    config: Any,
    data: list[dict],
    corpus: list[dict] | None = None,
) -> PreparedTarget:
    """Prepare either adapter generation and normalize its target handle."""
    prepared = adapter.prepare(config, data, corpus=corpus)
    if prepared is None:  # original adapters returned no handle
        return PreparedTarget(metadata={"legacy_adapter": True})
    if not isinstance(prepared, PreparedTarget):
        raise TypeError(
            f"{adapter.name}.prepare() must return PreparedTarget or None, "
            f"got {type(prepared).__name__}"
        )
    return prepared


def retrieve_adapter(
    adapter: Any,
    target: PreparedTarget,
    sample: dict,
    config: Any,
) -> RetrievalResult:
    """Run the managed retrieval stage with strict result normalization."""
    require_adapter_capabilities(adapter, "retrieval")
    retrieve = getattr(adapter, "retrieve", None)
    if not callable(retrieve):
        raise TypeError(
            f"Adapter {adapter.name!r} declares retrieval but has no retrieve()"
        )
    result = retrieve(target, sample, config)
    if not isinstance(result, RetrievalResult):
        raise TypeError(
            f"{adapter.name}.retrieve() must return RetrievalResult, "
            f"got {type(result).__name__}"
        )
    return result


def generate_adapter(
    adapter: Any,
    target: PreparedTarget,
    sample: dict,
    config: Any,
    retrieval: RetrievalResult | None = None,
) -> RagSystemOutput:
    """Generate through the managed API or fall back to legacy ``answer``."""
    generate = getattr(adapter, "generate", None)
    if callable(generate):
        result = generate(target, sample, config, retrieval=retrieval)
        if isinstance(result, AdapterGenerationResult):
            return result.to_legacy_output()
        if isinstance(result, RagSystemOutput):
            return result
        raise TypeError(
            f"{adapter.name}.generate() must return AdapterGenerationResult or "
            f"RagSystemOutput, got {type(result).__name__}"
        )

    answer = getattr(adapter, "answer", None)
    if not callable(answer):
        raise TypeError(
            f"Adapter {adapter.name!r} implements neither generate() nor answer()"
        )
    result = answer(sample, config)
    if not isinstance(result, RagSystemOutput):
        raise TypeError(
            f"{adapter.name}.answer() must return RagSystemOutput, "
            f"got {type(result).__name__}"
        )
    return result


def cleanup_adapter(adapter: Any, target: PreparedTarget, config: Any) -> None:
    """Clean up an owned managed target; legacy adapters are a no-op."""
    cleanup = getattr(adapter, "cleanup", None)
    if callable(cleanup):
        cleanup(target, config)


def adapter_stage_timings(
    adapter: Any, diagnostics: dict[str, Any]
) -> dict[str, float]:
    """Per-sample stage-timing contributions in a uniform ``{stage: seconds}`` shape.

    Adapters may implement ``diagnostic_stage_timings(diagnostics)`` to map
    their internal diagnostic keys onto benchmark stage names. Adapters
    without the method may instead embed a generic ``stage_timings`` dict in
    the per-sample diagnostics. The orchestrator only ever consumes the
    returned mapping — never the raw adapter-internal keys.
    """
    method = getattr(adapter, "diagnostic_stage_timings", None)
    raw = method(diagnostics) if callable(method) else None
    if raw is None:
        raw = diagnostics.get("stage_timings") or {}
    if not isinstance(raw, dict):
        raise TypeError(
            f"{adapter.name} stage timings must be a dict, got {type(raw).__name__}"
        )
    return {str(key): float(value) for key, value in raw.items()}


def adapter_aggregate_metrics(
    adapter: Any, diagnostics: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Run-level adapter metrics summary, or ``None`` if the adapter has none.

    Adapters may implement ``aggregate_metrics(diagnostics_list)`` to fold
    their per-sample diagnostics into one adapter-agnostic summary consumed
    by reporting and tracking.
    """
    method = getattr(adapter, "aggregate_metrics", None)
    if not callable(method):
        return None
    summary = method(diagnostics)
    if summary is not None and not isinstance(summary, dict):
        raise TypeError(
            f"{adapter.name}.aggregate_metrics() must return a dict or None, "
            f"got {type(summary).__name__}"
        )
    return summary
