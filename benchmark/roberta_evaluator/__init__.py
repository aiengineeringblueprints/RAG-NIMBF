"""Finetuned RoBERTa-based RAG evaluator (TRACe metrics).

This package trains and serves a multi-task RoBERTa classifier that
predicts the four RAGBench TRACe labels — Utilization, Relevance,
Adherence, Completeness — directly from ``(question, context, response)``
tuples.  At inference time a single forward pass replaces a slow LLM-judge
call.

Public surface:
    - :class:`RobertaTraceEvaluator` – drop-in evaluator for the framework
    - :class:`RobertaTraceClassifier` – multi-task model (training + inference)
    - :func:`evaluate_with_roberta` – mirror of :func:`benchmark.evaluation.evaluate_results`
    - :class:`RobertaTraceEvalResult` – result dataclass, schema-compatible
      with :class:`benchmark.evaluation.EvaluationResult`

See ``NOTES_I_RoBERTa.md`` at the project root for the architecture, the
training recipe (per RAGBench arXiv:2407.11005v2), expected metrics vs
LLM-judge, and inference-speed comparison.
"""
from __future__ import annotations

from .inference import RobertaTraceEvaluator, RobertaTraceEvalResult
from .model import RobertaTraceClassifier, TRACE_METRICS
from .register import register, get_evaluator

__all__ = [
    "RobertaTraceEvaluator",
    "RobertaTraceEvalResult",
    "RobertaTraceClassifier",
    "TRACE_METRICS",
    "register",
    "get_evaluator",
]
