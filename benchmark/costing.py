"""Token-cost helpers for API-backed model benchmarking."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelPricing:
    """USD pricing per one million tokens."""

    input_per_1m: float
    output_per_1m: float


def _float_env(name: str) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; expected a number", name, raw)
        return None


def _first_present(value: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _pricing_from_mapping(value: Mapping[str, Any]) -> ModelPricing | None:
    input_raw = _first_present(
        value, ("input_per_1m", "input", "prompt_per_1m", "prompt")
    )
    output_raw = _first_present(
        value, ("output_per_1m", "output", "completion_per_1m", "completion")
    )
    if input_raw is None or output_raw is None:
        return None
    try:
        return ModelPricing(float(input_raw), float(output_raw))
    except (TypeError, ValueError):
        return None


def _load_pricing_map() -> dict[str, ModelPricing]:
    raw = (
        os.getenv("LLM_MODEL_PRICING_USD_PER_1M")
        or os.getenv("MODEL_PRICING_USD_PER_1M")
        or ""
    ).strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Ignoring invalid model pricing JSON: %s", exc)
        return {}
    if not isinstance(parsed, dict):
        logger.warning("Ignoring model pricing JSON; expected an object")
        return {}

    pricing: dict[str, ModelPricing] = {}
    for model, value in parsed.items():
        if not isinstance(model, str) or not isinstance(value, dict):
            continue
        model_pricing = _pricing_from_mapping(value)
        if model_pricing is None:
            logger.warning("Ignoring invalid pricing entry for model %r", model)
            continue
        pricing[model] = model_pricing
    return pricing


def get_model_pricing(model_name: str | None) -> ModelPricing | None:
    """Return configured pricing for a model, if available.

    Per-model JSON wins. Global ``LLM_INPUT_COST_PER_1M_USD`` and
    ``LLM_OUTPUT_COST_PER_1M_USD`` act as a fallback for single-model runs.
    """
    if model_name:
        pricing_map = _load_pricing_map()
        for key in (
            model_name,
            model_name.split(":", 1)[-1],
            model_name.split("/")[-1],
        ):
            if key in pricing_map:
                return pricing_map[key]

    input_per_1m = _float_env("LLM_INPUT_COST_PER_1M_USD")
    output_per_1m = _float_env("LLM_OUTPUT_COST_PER_1M_USD")
    if input_per_1m is None or output_per_1m is None:
        return None
    return ModelPricing(input_per_1m, output_per_1m)


def estimate_cost_usd(
    *,
    model_name: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate USD cost from configured per-million-token model pricing."""
    pricing = get_model_pricing(model_name)
    if pricing is None:
        return None
    return (
        input_tokens / 1_000_000 * pricing.input_per_1m
        + output_tokens / 1_000_000 * pricing.output_per_1m
    )
