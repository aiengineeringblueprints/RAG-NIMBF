"""Evaluator registry hook for ``roberta_trace``.

The framework's evaluation wiring (see ``benchmark/evaluation.py`` and
``main.py``) calls :func:`get_evaluator` with the configured evaluator
type.  We keep a tiny module-level registry rather than a global plugin
system so that adding the RoBERTa evaluator does not perturb Ragas or
any other evaluation path.
"""
from __future__ import annotations

from typing import Callable, Protocol

from .inference import RobertaTraceEvalResult, RobertaTraceEvaluator


class EvaluatorProtocol(Protocol):
    def evaluate(
        self,
        questions,
        ground_truths,
        answers,
        contexts,
        **kwargs,
    ) -> RobertaTraceEvalResult:  # pragma: no cover - typing only
        ...


_EVALUATORS: dict[str, Callable[..., EvaluatorProtocol]] = {}


def register(
    name: str, factory: Callable[..., EvaluatorProtocol]
) -> None:
    """Register *factory* under *name*. Idempotent."""
    _EVALUATORS[name] = factory


def get_evaluator(name: str, **kwargs) -> EvaluatorProtocol:
    """Return the evaluator registered under *name*.

    Raises ``KeyError`` with the list of available names if unknown.
    """
    try:
        factory = _EVALUATORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown evaluator {name!r}. Registered: {sorted(_EVALUATORS)}"
        ) from exc
    return factory(**kwargs)


def available_evaluators() -> list[str]:
    return sorted(_EVALUATORS)


# ── default registrations ────────────────────────────────────────────
def _make_roberta_trace(**kwargs) -> RobertaTraceEvaluator:
    # Filter to the kwargs RobertaTraceEvaluator actually accepts so the
    # registry stays forward-compatible with new options.
    import inspect

    sig = inspect.signature(RobertaTraceEvaluator.__init__)
    accepted = {
        k: v for k, v in kwargs.items() if k in sig.parameters
    }
    return RobertaTraceEvaluator(**accepted)


register("roberta_trace", _make_roberta_trace)
