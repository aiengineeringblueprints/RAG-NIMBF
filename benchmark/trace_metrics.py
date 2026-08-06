"""TRACe framework metrics from RAGBench (arXiv:2407.11005v2).

The TRACe paper defines four dimensions:
    * Adherence   — already covered by RAGAS ``faithfulness``
    * Relevance   — already covered by RAGAS ``context_recall``
    * Utilization — fraction of retrieved-context tokens the response
      actually attributes to (NEW)
    * Completeness — fraction of relevant context spans covered by the
      response (NEW)

This module implements the two new dimensions.  It supports two modes:

    1. LLM-judge mode (preferred).  A critic LLM marks which response
       token spans are attributable to the context (Utilization) and
       which context claims are missing from the response
       (Completeness).
    2. Heuristic fallback.  Token n-gram overlap between answer and
       context.  Used automatically when no judge LLM is reachable or
       when the caller passes ``llm_judge=False``.

Results are merged into the existing ``custom_metric_means`` dict so
that MLflow logging, CSV / JSON export, and visualization continue to
work without further plumbing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import mlflow

logger = logging.getLogger(__name__)


# ── Result container ─────────────────────────────────────────────────

@dataclass(frozen=True)
class TraceMetricsResult:
    """Aggregated TRACe Utilization / Completeness scores.

    ``metric_means`` keys:

        ``trace_utilization``   — mean Utilization (T)
        ``trace_completeness``  — mean Completeness (C)

    Per-sample dicts additionally include ``trace_mode`` ("llm" or
    "heuristic") so downstream consumers can tell how the score was
    derived.
    """

    metric_means: dict[str, float]
    per_sample_scores: list[dict[str, float | None]] = field(default_factory=list)
    samples_with_valid_scores: dict[str, int] = field(default_factory=dict)
    error: str | None = None


# ── Helpers shared with custom_metrics ───────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [
        t.strip(".,;:!?\"'()[]{}")
        for t in text.lower().split()
        if t.strip(".,;:!?\"'()[]{}")
    ]


_REFUSAL_PATTERNS = (
    "i cannot answer", "i can't answer", "i cannot provide", "i can't provide",
    "i am unable to answer", "i'm unable to answer", "i am not able to answer",
    "not possible to answer", "cannot be answered", "not sufficient",
    "insufficient", "no information", "i don't have", "i do not have",
    "does not contain", "does not provide", "not contained", "not provided",
    "not available in", "not addressed", "context does not", "provided context",
    "given context",
)


def _is_refusal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return True
    return any(p in t for p in _REFUSAL_PATTERNS)


# ── Heuristic fallback ───────────────────────────────────────────────
#
# Token n-gram overlap.  Cheap, deterministic, no model required.
# Not as accurate as the LLM judge but provides a sane baseline and
# lets the pipeline run in environments without a critic endpoint.

def _heuristic_utilization(answer: str, contexts: Sequence[str]) -> float:
    """Fraction of answer tokens that appear in the context.

    This is the symmetric view of the paper's definition (paper measures
    fraction of context tokens used by the answer; we measure fraction
    of answer tokens grounded in context, which is more stable to
    answer length and avoids penalising concise correct answers).
    """
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


def _heuristic_completeness(
    answer: str, contexts: Sequence[str], *, ngram: int = 4
) -> float:
    """Fraction of context n-grams covered by the answer.

    Long context → many n-grams → fine-grained Completeness signal.
    """
    ans_ngrams = set()
    a_tokens = _tokenize(answer)
    for i in range(len(a_tokens) - ngram + 1):
        ans_ngrams.add(tuple(a_tokens[i : i + ngram]))

    if not ans_ngrams:
        return 0.0

    total_ctx_ngrams = 0
    covered = 0
    for c in contexts:
        c_tokens = _tokenize(c)
        for i in range(len(c_tokens) - ngram + 1):
            total_ctx_ngrams += 1
            if tuple(c_tokens[i : i + ngram]) in ans_ngrams:
                covered += 1

    if total_ctx_ngrams == 0:
        return 0.0
    return covered / total_ctx_ngrams


# ── LLM-judge prompt ─────────────────────────────────────────────────
#
# A single prompt elicits both signals.  Keeping it to one round-trip
# per sample matters for local models where latency dominates.

_JUDGE_PROMPT = """You are a strict evaluator for a retrieval-augmented QA system.

CONTEXT (retrieved passages, concatenated):
<<<
{context}
>>>

ANSWER (generated response):
<<<
{answer}
>>>

Evaluate the answer along TWO axes.  Be harsh and literal.

1. UTILIZATION (T): What fraction of the ANSWER's content is directly
   supported by — i.e. attributable to — the CONTEXT?  Phrases copied
   from context, paraphrases of context, and facts stated in context
   count as supported.  Generic connective language ("the answer is",
   "in summary") is ignored (neither adds nor subtracts).  Information
   in the answer that is NOT in the context lowers the score.  Answer a
   single float in [0, 1].

2. COMPLETENESS (C): How much of the relevant information present in
   the CONTEXT is covered by the ANSWER?  A high score means the answer
   captures all the substantive claims the context offers for the
   question; a low score means the answer cherry-picks a subset.  Do
   not penalise the answer for context that is irrelevant to the
   implicit question.  Answer a single float in [0, 1].

Respond with ONE line of compact JSON and nothing else, exactly in
this shape:

{{"utilization": <float 0..1>, "completeness": <float 0..1>}}
"""


def _build_judge_prompt(answer: str, contexts: Sequence[str]) -> str:
    # Cap context length to keep the prompt token-efficient; the judge
    # only needs a representative window, not the full retrieval set.
    ctx_text = "\n\n".join(c.strip() for c in contexts if c and c.strip())
    max_ctx_chars = 6000
    if len(ctx_text) > max_ctx_chars:
        ctx_text = ctx_text[:max_ctx_chars] + " […truncated…]"
    return _JUDGE_PROMPT.format(context=ctx_text, answer=answer.strip())


_JUDGE_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _parse_judge_response(raw: str) -> tuple[float | None, float | None]:
    """Extract (utilization, completeness) from the judge's reply.

    Tolerates both clean JSON and JSON embedded in surrounding prose.
    Returns ``(None, None)`` on parse failure.
    """
    if not raw:
        return None, None
    text = raw.strip()

    # Fast path: response is already a clean JSON object.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = _JUDGE_JSON_RE.search(text)
        if not match:
            return None, None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None, None

    if not isinstance(obj, dict):
        return None, None

    def _to_float(key: str) -> float | None:
        val = obj.get(key)
        if isinstance(val, (int, float)):
            f = float(val)
        elif isinstance(val, str):
            try:
                f = float(val.strip())
            except ValueError:
                return None
        else:
            return None
        if f < 0.0 or f > 1.0:
            # Clamp slightly out-of-range judge answers; reject larger
            # excursions as obvious parsing mistakes.
            if -0.05 <= f <= 1.05:
                return min(1.0, max(0.0, f))
            return None
        return f

    return _to_float("utilization"), _to_float("completeness")


# ── Public entry point ───────────────────────────────────────────────

@mlflow.trace(name="trace_metrics", span_type="func")
def compute_trace_metrics(
    questions: list[str],
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
    critic_max_tokens: int = 512,
    llm_judge: bool = True,
    judge_caller: Callable[[str], str] | None = None,
) -> TraceMetricsResult:
    """Compute TRACe Utilization and Completeness.

    Parameters mirror :func:`benchmark.evaluation.evaluate_results` so
    callers can re-use the same critic wiring.

    ``llm_judge``
        If True (default) attempt to use the critic LLM.  If the LLM is
        unreachable or every sample fails to parse, fall back to the
        heuristic automatically.
    ``judge_caller``
        Test seam: a callable that takes the prompt and returns the
        raw model output.  Production callers leave this ``None`` and
        the function builds its own LangChain chat model.
    """
    if not questions:
        return TraceMetricsResult(
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
            # Critic model could not be initialised — degrade gracefully.
            use_llm = False
            logger.warning(
                "TRACe critic LLM unavailable; falling back to heuristic "
                "token-overlap scores."
            )

    per_sample: list[dict[str, float | None]] = []
    util_acc: list[float] = []
    comp_acc: list[float] = []
    llm_failures = 0

    for q, ctx, ans in zip(questions, contexts, answers):
        sample: dict[str, float | None] = {}

        if not ctx or all(not str(c).strip() for c in ctx):
            # No context → nothing to utilise and nothing to complete.
            sample["trace_utilization"] = 0.0
            sample["trace_completeness"] = 0.0
            sample["trace_mode"] = 0.0  # heuristic
            util_acc.append(0.0)
            comp_acc.append(0.0)
            per_sample.append(sample)
            continue

        if _is_refusal(ans):
            # A refusal attributes nothing and covers nothing.
            sample["trace_utilization"] = 0.0
            sample["trace_completeness"] = 0.0
            sample["trace_mode"] = 0.0
            util_acc.append(0.0)
            comp_acc.append(0.0)
            per_sample.append(sample)
            continue

        u: float | None = None
        c: float | None = None
        if use_llm and caller is not None:
            try:
                raw = caller(_build_judge_prompt(ans, ctx))
            except Exception as exc:  # pragma: no cover - network path
                logger.debug("TRACe judge call failed: %s", exc)
                raw = ""
            u, c = _parse_judge_response(raw)
            if u is None or c is None:
                llm_failures += 1

        if u is None or c is None:
            u = _heuristic_utilization(ans, ctx)
            c = _heuristic_completeness(ans, ctx)
            sample["trace_mode"] = 0.0  # heuristic
        else:
            sample["trace_mode"] = 1.0  # llm

        sample["trace_utilization"] = u
        sample["trace_completeness"] = c
        util_acc.append(u)
        comp_acc.append(c)
        per_sample.append(sample)

    # If LLM mode was requested but every sample failed, surface a soft
    # error so users know they got heuristics instead of judgements.
    error: str | None = None
    if use_llm and per_sample and llm_failures == len(per_sample):
        error = (
            "All TRACe LLM-judge responses failed to parse; reported "
            "scores are heuristic token-overlap estimates."
        )

    metric_means: dict[str, float] = {}
    if util_acc:
        metric_means["trace_utilization"] = sum(util_acc) / len(util_acc)
    if comp_acc:
        metric_means["trace_completeness"] = sum(comp_acc) / len(comp_acc)

    valid_counts = {
        "trace_utilization": len(util_acc),
        "trace_completeness": len(comp_acc),
    }

    return TraceMetricsResult(
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
    """Construct a callable that sends a single prompt to the critic LLM.

    Returns ``None`` if the model cannot be created (e.g. missing
    dependency or unreachable host).  The caller handles the fallback.
    """
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
        logger.debug("TRACe critic chat model init failed: %s", exc)
        return None

    chat = wrap_for_ragas(chat)

    def _call(prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        result = chat.invoke([HumanMessage(content=prompt)])
        content: Any = getattr(result, "content", "")
        if isinstance(content, list):
            # Some OpenAI-compatible servers return tool-call / multi-part
            # payloads; coerce to a single string for parsing.
            content = json.dumps(content)
        return str(content) if content is not None else ""

    return _call
