"""RAGChecker-style error attribution metrics (arXiv:2408.08067).

RAGChecker evaluates a RAG system at *claim level* rather than scoring the
whole response once.  Every claim the generator produces is classified into
one of five mutually-exclusive categories, and every ground-truth claim is
tagged with whether the response and the retrieved context cover it.  This
lets the benchmark attribute *why* a score is low: bad retrieval, noisy
context, or a hallucinating generator.

Category semantics (per answer claim):
    * ``correct``           – entailed in the retrieved context and in the
                              ground truth
    * ``self_knowledge``    – in the ground truth but *not* entailed in the
                              context (generator used internal knowledge)
    * ``relevant_noise``    – wrong, but entailed in a context chunk that is
                              relevant to the ground truth
    * ``irrelevant_noise``  – wrong, entailed only in an irrelevant chunk
    * ``hallucinated``      – wrong and not entailed in any retrieved chunk

Metrics exposed (prefix ``rc_`` for "RAGChecker"):
    * ``rc_claim_precision``  – correct / total answer claims
    * ``rc_claim_recall``     – ground-truth claims covered by the response
    * ``rc_claim_f1``         – harmonic mean of the above (system score)
    * ``rc_faithfulness``     – fraction of answer claims entailed in context
    * ``rc_context_recall``   – ground-truth claims covered by retrieved chunks
    * ``rc_context_precision``– fraction of retrieved chunks judged relevant
    * ``rc_hallucination``    – unsupported answer claims (lower is better)
    * ``rc_noise_sensitivity_relevant``   – (lower is better)
    * ``rc_noise_sensitivity_irrelevant`` – (lower is better)
    * ``rc_self_knowledge``   – correct claims from model memory (lower is
                                better for a pure-RAG system)

Like :mod:`benchmark.trace_metrics`, this module supports two modes:
    1. LLM-judge mode (preferred): a critic LLM classifies the claims.
    2. Heuristic fallback: token-overlap estimates for the metrics that have
       a tractable lexical proxy (faithfulness, context recall/precision).
       The remaining metrics are reported as ``None``.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import mlflow

from benchmark.custom_metrics import is_refusal_answer

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ErrorAttributionResult:
    """Aggregated RAGChecker-style scores across all samples.

    Per-sample dicts additionally include ``rc_mode`` ("llm" or
    "heuristic") so downstream consumers can tell how the score was
    derived.
    """

    metric_means: dict[str, float]
    per_sample_scores: list[dict[str, float | None]] = field(default_factory=list)
    samples_with_valid_scores: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ── Category constants ───────────────────────────────────────────────

_CLAIM_CATEGORIES = {
    "correct": "correct",
    "self_knowledge": "self_knowledge",
    "relevant_noise": "relevant_noise",
    "irrelevant_noise": "irrelevant_noise",
    "hallucinated": "hallucinated",
}


# ── Token helpers ────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [
        t.strip(".,;:!?\"'()[]{}")
        for t in text.lower().split()
        if t.strip(".,;:!?\"'()[]{}")
    ]


# ── Heuristic fallback ───────────────────────────────────────────────

def _heuristic_faithfulness(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in the retrieved context."""
    ans_tokens = _tokenize(answer)
    if not ans_tokens:
        return 0.0
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(_tokenize(c))
    if not ctx_tokens:
        return 0.0
    matched = sum(1 for t in ans_tokens if t in ctx_tokens)
    return matched / len(ans_tokens)


def _heuristic_context_recall(
    ground_truth: str, contexts: Sequence[str]
) -> float:
    """Fraction of ground-truth tokens covered by the retrieved context."""
    gt_tokens = _tokenize(ground_truth)
    if not gt_tokens:
        return 0.0
    ctx_tokens = set()
    for c in contexts:
        ctx_tokens.update(_tokenize(c))
    if not ctx_tokens:
        return 0.0
    matched = sum(1 for t in gt_tokens if t in ctx_tokens)
    return matched / len(gt_tokens)


def _heuristic_context_precision(contexts: Sequence[str]) -> float | None:
    """Return None — chunk relevance is not tractable lexically here."""
    return None


# ── LLM-judge prompt ─────────────────────────────────────────────────

_JUDGE_PROMPT = """You are a strict evaluator for a retrieval-augmented QA system.

QUESTION:
<<<
{question}
>>>

GROUND TRUTH ANSWER:
<<<
{ground_truth}
>>>

RETRIEVED CONTEXT CHUNKS (each labelled with an index):
<<<
{context}
>>>

RESPONSE (generated answer):
<<<
{response}
>>>

Decompose the RESPONSE into distinct factual claims.  For EACH claim assign
exactly one category:

- "correct": the claim is entailed by the GROUND TRUTH and supported by the
  RETRIEVED CONTEXT.
- "self_knowledge": the claim matches the GROUND TRUTH but is NOT entailed by
  any retrieved chunk (the model used its own knowledge).
- "relevant_noise": the claim is WRONG but is entailed by a chunk that is
  relevant to the ground truth.
- "irrelevant_noise": the claim is WRONG and entailed only by chunks that are
  irrelevant to the ground truth.
- "hallucinated": the claim is WRONG and entailed by no retrieved chunk.

Also:
1. List the indices of retrieved chunks that are relevant to answering the
   GROUND TRUTH (see RETRIEVED CONTEXT CHUNKS).
2. Decompose the GROUND TRUTH into claims.  For each, say whether the RESPONSE
   covers it ("covered_in_response") and whether the RETRIEVED CONTEXT
   contains it ("covered_in_context").

Respond with ONE line of compact JSON and nothing else, exactly in this shape:

{{
  "relevant_context_indices": [0, 2],
  "answer_claims": [
    {{"text": "...", "category": "correct"}}
  ],
  "ground_truth_claims": [
    {{"text": "...", "covered_in_response": true, "covered_in_context": true}}
  ]
}}
"""


def _build_judge_prompt(
    question: str, ground_truth: str, response: str, contexts: Sequence[str]
) -> str:
    ctx_lines = []
    for i, c in enumerate(contexts):
        if c and c.strip():
            ctx_lines.append(f"[{i}] {c.strip()}")
    ctx_text = "\n\n".join(ctx_lines)
    max_ctx_chars = 6000
    if len(ctx_text) > max_ctx_chars:
        ctx_text = ctx_text[:max_ctx_chars] + " […truncated…]"
    return _JUDGE_PROMPT.format(
        question=question.strip(),
        ground_truth=ground_truth.strip(),
        context=ctx_text,
        response=response.strip(),
    )


_JUDGE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> dict | None:
    """Parse the judge's JSON reply, tolerating surrounding prose."""
    if not raw:
        return None
    text = raw.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JUDGE_JSON_RE.search(text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    if not isinstance(obj, dict):
        return None
    return obj


def _as_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"true", "yes", "1"}
    return bool(val)


# ── Aggregation from judge output ────────────────────────────────────

def _aggregate_judge(
    judge: dict, contexts: Sequence[str]
) -> dict[str, float | None]:
    """Compute the metric set from one sample's judge output."""
    answer_claims = judge.get("answer_claims") or []
    gt_claims = judge.get("ground_truth_claims") or []
    rel_indices = judge.get("relevant_context_indices") or []

    total = len(answer_claims)
    counts: dict[str, int] = {}
    for claim in answer_claims:
        if not isinstance(claim, dict):
            continue
        cat = str(claim.get("category", "")).strip().lower()
        if cat not in _CLAIM_CATEGORIES:
            cat = "hallucinated"
        counts[cat] = counts.get(cat, 0) + 1

    correct = counts.get("correct", 0)
    self_knowledge = counts.get("self_knowledge", 0)
    relevant_noise = counts.get("relevant_noise", 0)
    irrelevant_noise = counts.get("irrelevant_noise", 0)
    hallucinated = counts.get("hallucinated", 0)

    # Ground-truth claim coverage.
    gt_total = 0
    gt_in_response = 0
    gt_in_context = 0
    for claim in gt_claims:
        if not isinstance(claim, dict):
            continue
        gt_total += 1
        if _as_bool(claim.get("covered_in_response")):
            gt_in_response += 1
        if _as_bool(claim.get("covered_in_context")):
            gt_in_context += 1

    def _rate(n: int) -> float | None:
        return n / total if total > 0 else None

    claim_precision = _rate(correct)
    claim_recall = gt_in_response / gt_total if gt_total > 0 else None
    claim_f1: float | None = None
    if claim_precision is not None and claim_recall is not None and (
        claim_precision + claim_recall
    ) > 0:
        claim_f1 = 2.0 * claim_precision * claim_recall / (
            claim_precision + claim_recall
        )

    # Faithfulness: fraction of answer claims entailed by context
    # (everything except hallucinated and self_knowledge).
    entailed = total - hallucinated - self_knowledge
    faithfulness = entailed / total if total > 0 else None

    context_precision: float | None = None
    if contexts:
        context_precision = len(rel_indices) / len(contexts)

    context_recall = gt_in_context / gt_total if gt_total > 0 else None

    return {
        "rc_claim_precision": claim_precision,
        "rc_claim_recall": claim_recall,
        "rc_claim_f1": claim_f1,
        "rc_faithfulness": faithfulness,
        "rc_context_recall": context_recall,
        "rc_context_precision": context_precision,
        "rc_hallucination": _rate(hallucinated),
        "rc_noise_sensitivity_relevant": _rate(relevant_noise),
        "rc_noise_sensitivity_irrelevant": _rate(irrelevant_noise),
        "rc_self_knowledge": _rate(self_knowledge),
    }


# ── Public entry point ───────────────────────────────────────────────

@mlflow.trace(name="error_attribution", span_type="func")
def compute_error_attribution(
    questions: list[str],
    ground_truths: list[str],
    contexts: list[list[str]],
    answers: list[str],
    *,
    llm_model: str = "gemma3:4b",
    critic_llm_model: str | None = None,
    ollama_base_url: str = "http://localhost:11434",
    ollama_api_key: str | None = None,
    openai_compat_base_url: str | None = None,
    openai_compat_api_key: str | None = None,
    critic_ollama_base_url: str | None = None,
    critic_ollama_api_key: str | None = None,
    critic_openai_compat_base_url: str | None = None,
    critic_openai_compat_api_key: str | None = None,
    critic_max_tokens: int = 1024,
    relevance_threshold: float = 0.5,
    llm_judge: bool = True,
    judge_caller: Callable[[str], str] | None = None,
) -> ErrorAttributionResult:
    """Compute RAGChecker-style error-attribution metrics.

    Parameters mirror :func:`benchmark.evaluation.evaluate_results` and
    :func:`benchmark.trace_metrics.compute_trace_metrics` so callers can
    reuse the same critic wiring.

    ``llm_judge``
        If True (default) attempt to use the critic LLM.  If the LLM is
        unreachable or every sample fails to parse, fall back to the
        heuristic automatically.
    ``judge_caller``
        Test seam: a callable that takes the prompt and returns the raw
        model output.  Production callers leave ``None``.
    """
    if not questions:
        return ErrorAttributionResult(
            metric_means={},
            per_sample_scores=[],
            samples_with_valid_scores={},
            error="No questions provided.",
        )

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({"metrics.num_questions": len(questions)})

    caller = judge_caller
    use_llm = llm_judge
    if use_llm and caller is None:
        caller = _build_default_caller(
            llm_model=llm_model,
            critic_llm_model=critic_llm_model,
            ollama_base_url=ollama_base_url,
            ollama_api_key=ollama_api_key,
            openai_compat_base_url=openai_compat_base_url,
            openai_compat_api_key=openai_compat_api_key,
            critic_ollama_base_url=critic_ollama_base_url,
            critic_ollama_api_key=critic_ollama_api_key,
            critic_openai_compat_base_url=critic_openai_compat_base_url,
            critic_openai_compat_api_key=critic_openai_compat_api_key,
            critic_max_tokens=critic_max_tokens,
        )
        if caller is None:
            use_llm = False
            logger.warning(
                "RAGChecker critic LLM unavailable; falling back to "
                "heuristic token-overlap scores."
            )

    per_sample: list[dict[str, float | None]] = []
    accum: dict[str, list[float]] = {}
    llm_failures = 0

    for q, gt, ctx, ans in zip(questions, ground_truths, contexts, answers):
        sample: dict[str, float | None] = {}

        if not ctx or all(not str(c).strip() for c in ctx):
            sample["rc_claim_precision"] = None
            sample["rc_claim_recall"] = None
            sample["rc_claim_f1"] = None
            sample["rc_faithfulness"] = None
            sample["rc_context_recall"] = None
            sample["rc_context_precision"] = None
            sample["rc_hallucination"] = None
            sample["rc_noise_sensitivity_relevant"] = None
            sample["rc_noise_sensitivity_irrelevant"] = None
            sample["rc_self_knowledge"] = None
            sample["rc_mode"] = 0.0
            per_sample.append(sample)
            continue

        if is_refusal_answer(ans):
            # A refusal adds no claims and covers nothing.
            sample["rc_claim_precision"] = 0.0
            sample["rc_claim_recall"] = 0.0
            sample["rc_claim_f1"] = 0.0
            sample["rc_faithfulness"] = 1.0
            sample["rc_context_recall"] = 0.0
            sample["rc_context_precision"] = 0.0
            sample["rc_hallucination"] = 0.0
            sample["rc_noise_sensitivity_relevant"] = 0.0
            sample["rc_noise_sensitivity_irrelevant"] = 0.0
            sample["rc_self_knowledge"] = 0.0
            sample["rc_mode"] = 0.0
            for k, v in sample.items():
                if k != "rc_mode" and isinstance(v, float):
                    accum.setdefault(k, []).append(v)
            per_sample.append(sample)
            continue

        metrics: dict[str, float | None] | None = None
        if use_llm and caller is not None:
            try:
                raw = caller(_build_judge_prompt(q, gt, ans, ctx))
            except Exception as exc:  # pragma: no cover - network path
                logger.debug("RAGChecker judge call failed: %s", exc)
                raw = ""
            judge = _parse_judge_response(raw)
            if judge is not None:
                metrics = _aggregate_judge(judge, ctx)
            else:
                llm_failures += 1

        if metrics is None:
            metrics = {
                "rc_claim_precision": None,
                "rc_claim_recall": None,
                "rc_claim_f1": None,
                "rc_faithfulness": _heuristic_faithfulness(ans, ctx),
                "rc_context_recall": _heuristic_context_recall(gt, ctx),
                "rc_context_precision": _heuristic_context_precision(ctx),
                "rc_hallucination": None,
                "rc_noise_sensitivity_relevant": None,
                "rc_noise_sensitivity_irrelevant": None,
                "rc_self_knowledge": None,
            }
            sample["rc_mode"] = 0.0
        else:
            sample["rc_mode"] = 1.0

        for k, v in metrics.items():
            sample[k] = v
            if isinstance(v, (int, float)) and not math.isnan(v):
                accum.setdefault(k, []).append(float(v))

        per_sample.append(sample)

    error: str | None = None
    if use_llm and per_sample and llm_failures == len(per_sample):
        error = (
            "All RAGChecker LLM-judge responses failed to parse; reported "
            "scores are heuristic token-overlap estimates."
        )

    metric_means: dict[str, float] = {
        k: sum(v) / len(v) for k, v in accum.items() if v
    }
    valid_counts = {k: len(v) for k, v in accum.items() if v}

    return ErrorAttributionResult(
        metric_means=metric_means,
        per_sample_scores=per_sample,
        samples_with_valid_scores=valid_counts,
        error=error,
    )


# ── Critic LLM wiring ────────────────────────────────────────────────

def _build_default_caller(
    *,
    llm_model: str,
    critic_llm_model: str | None,
    ollama_base_url: str,
    ollama_api_key: str | None,
    openai_compat_base_url: str | None,
    openai_compat_api_key: str | None,
    critic_ollama_base_url: str | None,
    critic_ollama_api_key: str | None,
    critic_openai_compat_base_url: str | None,
    critic_openai_compat_api_key: str | None,
    critic_max_tokens: int,
) -> Callable[[str], str] | None:
    """Construct a callable that sends a single prompt to the critic LLM."""
    try:
        from benchmark.providers import (
            parse_model_id,
            get_chat_model,
            wrap_for_ragas,
        )
    except ImportError:
        return None

    effective_model = critic_llm_model or llm_model
    provider, model_name = parse_model_id(effective_model)

    if provider == "openai":
        base = critic_openai_compat_base_url or openai_compat_base_url or ""
        key = critic_openai_compat_api_key or openai_compat_api_key
    else:
        base = critic_ollama_base_url or ollama_base_url
        key = critic_ollama_api_key or ollama_api_key

    try:
        chat = get_chat_model(
            provider=provider,
            model_name=model_name,
            base_url=base,
            api_key=key,
            max_tokens=critic_max_tokens,
            temperature=0.0,
        )
    except Exception as exc:
        logger.debug("RAGChecker critic chat model init failed: %s", exc)
        return None

    chat = wrap_for_ragas(chat)

    def _call(prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        result = chat.invoke([HumanMessage(content=prompt)])
        content: Any = getattr(result, "content", "")
        if isinstance(content, list):
            content = json.dumps(content)
        return str(content) if content is not None else ""

    return _call
