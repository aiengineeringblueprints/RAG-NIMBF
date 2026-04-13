import logging
import re
import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 10  # seconds

from benchmark.metrics import get_gpu_usage
from benchmark.providers import get_chat_model

# Matches <think ...>...</think > blocks (attributes + multiline content)
_THINK_TAG_PATTERN = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)

AnswerStripMode = Literal["full", "tags_only", "off"]

# Last numeric / percent token on a line (commas as thousands separators allowed)
_VALUE_NUMBER_PATTERN = re.compile(
    r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?|-?\d+\.\d+%?"
)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks only."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


def strip_thinking(text: str, mode: AnswerStripMode = "tags_only") -> str:
    """Normalize LLM output for scoring / logging.

    * ``off`` — only strip leading/trailing whitespace (no tag removal).
    * ``tags_only`` — remove ``<think>...</think>`` blocks; keep remaining text.
    * ``full`` — like ``tags_only``, then drop output that still looks like
      bare reasoning prose (see module docstring history).
    """
    if mode == "off":
        return text.strip()

    cleaned = strip_think_tags(text)

    if mode == "tags_only":
        return cleaned

    # full
    if _looks_like_thinking(cleaned):
        return ""
    return cleaned


def extract_concise_fallback(text: str) -> str:
    """If the model buried a value after reasoning, take yes/no or last number-like token.

    Scans non-empty lines from bottom to top. Intended for FinQA-style numeric answers.
    """
    if not text or not text.strip():
        return ""

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    for line in reversed(lines):
        low = line.lower().rstrip(".,!? ")
        tokens = low.split()
        if len(tokens) == 1 and tokens[0] in ("yes", "no"):
            return tokens[0]

        matches = list(_VALUE_NUMBER_PATTERN.finditer(line))
        if matches:
            return matches[-1].group(0).replace(",", "")

    return ""


def _looks_like_thinking(text: str) -> bool:
    """Return True if *text* reads like internal reasoning, not an answer."""
    if not text:
        return False

    first_line = text.split("\n", 1)[0].strip().lower()

    # Common thinking starters used by Qwen / DeepSeek models
    _THINKING_STARTS = (
        "okay,",
        "okay ",
        "let's see",
        "let me",
        "looking at",
        "first,",
        "well,",
        "hmm",
        "so,",
        "so ",
        "now ",
        "the user",
        "i need to",
        "i should",
        "i'll",
        "to answer",
        "to find",
        "to calculate",
        "to determine",
    )

    # If the entire text ends mid-sentence (trailing comma, "of", "the", etc.)
    # it was cut off and is almost certainly thinking, not a final answer.
    _CUTOFF_ENDS = (
        " of", " the", " a", " an", " is", " in", " to", " for",
        " and", " that", " with", " which", " it",
        ",",
    )

    text_lower = text.strip().lower()

    # Check for cutoff: answer ends mid-sentence without punctuation
    ends_with_punct = text_lower[-1] in ".!?" if text_lower else False
    if not ends_with_punct:
        for marker in _CUTOFF_ENDS:
            if text_lower.endswith(marker):
                return True

    # Check if starts with thinking markers
    for marker in _THINKING_STARTS:
        if first_line.startswith(marker):
            return True

    return False


@dataclass
class GenerationResult:
    answer: str
    ttft_seconds: float
    total_seconds: float
    token_count: int
    tokens_per_second: float
    gpu_usage: dict | None
    raw_content: str = ""
    raw_reasoning: str | None = None


def get_llm(
    *,
    provider: str,
    model_name: str,
    base_url: str,
    api_key: str | None = None,
    max_new_tokens: int = 256,
) -> BaseChatModel:
    """Create a chat model for the given provider."""
    return get_chat_model(
        provider=provider,
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=max_new_tokens,
        temperature=0.0,
    )


def generate_answer(
    llm: BaseChatModel,
    question: str,
    contexts: list[str],
    *,
    system_prompt: str = (
        "Answer the question using ONLY the provided context. "
        "Return ONLY the raw value — a number, percentage, ratio, or yes/no. "
        "Do NOT include units, explanations, reasoning, or full sentences. "
        "Examples: 494.0 | 0.12 | -0.46 | 1 | 5.8"
    ),
    human_template: str = "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    strip_mode: AnswerStripMode = "tags_only",
    value_fallback: bool = True,
) -> GenerationResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    context_text = "\n\n".join(contexts)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_template.format(context=context_text, question=question)),
    ]

    start = time.perf_counter()
    response = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = llm.invoke(messages)
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise
            delay = _BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(
                "LLM invoke failed (attempt %d/%d): %s  — retrying in %ds",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)
    total = time.perf_counter() - start

    raw_content = str(response.content) if response.content else ""
    reasoning_kw = response.additional_kwargs.get("reasoning_content")
    raw_reasoning = str(reasoning_kw) if reasoning_kw else None

    combined = raw_content
    if not combined.strip() and raw_reasoning:
        combined = raw_reasoning

    tag_clean = strip_think_tags(combined)
    if strip_mode == "off":
        answer = combined.strip()
    else:
        answer = strip_thinking(combined, strip_mode)

    if not answer and value_fallback and tag_clean:
        answer = extract_concise_fallback(tag_clean)

    # Use actual token counts from usage metadata when available
    usage = getattr(response, "usage_metadata", None)
    if usage:
        token_count = usage.get("output_tokens", 0)
    else:
        token_count = 0

    gpu = get_gpu_usage()

    return GenerationResult(
        answer=answer,
        ttft_seconds=total,  # non-streaming: no per-token timing, use total latency
        total_seconds=total,
        token_count=token_count,
        tokens_per_second=token_count / total if total > 0 and token_count > 0 else 0,
        gpu_usage=gpu,
        raw_content=raw_content,
        raw_reasoning=raw_reasoning,
    )
