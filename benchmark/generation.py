import re
import time
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from benchmark.metrics import get_gpu_usage
from benchmark.providers import get_chat_model

# Matches <think ...>...</think > blocks (attributes + multiline content)
_THINK_PATTERN = re.compile(r"<think[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove reasoning/thinking tags from LLM output.

    Models like Qwen3 and DeepSeek-R1 wrap internal reasoning in
    ``<think ...>...</think >`` tags.  These inflate RAGAS scores
    (especially answer_relevancy) because the evaluator sees the
    reasoning text instead of just the actual answer.
    """
    return _THINK_PATTERN.sub("", text).strip()


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


def generate_answer(llm: BaseChatModel, question: str, contexts: list[str]) -> GenerationResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    context_text = "\n\n".join(contexts)
    messages = [
        SystemMessage(content="Answer the question based only on the provided context. Be concise and precise."),
        HumanMessage(content=f"Context:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"),
    ]

    start = time.perf_counter()
    response = llm.invoke(messages)
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
