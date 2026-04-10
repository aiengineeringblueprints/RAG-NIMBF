import logging
import re
import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_MAX_RETRIES = 5
_BASE_DELAY = 10  # seconds

from benchmark.metrics import get_gpu_usage
from benchmark.providers import get_chat_model

# Matches <think ...>...</think > blocks (attributes + multiline content)
_THINK_TAG_PATTERN = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning/thinking from LLM output.

    Handles two formats:
    1. Tag-wrapped: ``<think ...>reasoning</think >`` (Qwen3, DeepSeek-R1)
    2. Bare thinking: the model outputs reasoning text that reads like
       internal monologue (e.g. "Okay, let's see...") without ever
       producing a final answer.  In this case the entire output is
       discarded.

    After stripping tags, if no substantive answer remains (i.e. the
    content looks like thinking prose), an empty string is returned so
    RAGAS treats it as a failed generation rather than scoring the
    reasoning as the "answer".
    """
    # 1. Remove explicit <think ...>...</think > blocks
    cleaned = _THINK_TAG_PATTERN.sub("", text).strip()

    # 2. If the cleaned result still looks like thinking prose
    #    (first-person reasoning, no clear answer), discard it entirely.
    #    Heuristic: starts with common thinking markers and never
    #    transitions into a declarative answer.
    if _looks_like_thinking(cleaned):
        return ""

    return cleaned


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

    answer = strip_thinking(str(response.content))

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
    )
