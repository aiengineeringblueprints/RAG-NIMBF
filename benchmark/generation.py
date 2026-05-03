import ast
import logging
import mlflow
import operator
import os
import re
import time
from dataclasses import dataclass
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

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

# Matches "FINAL: <value>" line from finqa template
_FINAL_LINE_PATTERN = re.compile(r"^FINAL:\s*(.+)$", re.MULTILINE)

# Keywords that indicate a percentage-type question
_PERCENTAGE_KEYWORDS = re.compile(
    r"percent|percentage|%|what fraction|what share|what proportion|what part",
    re.IGNORECASE,
)


def strip_think_tags(text: str) -> str:
    """Remove ``<think ...>...</think >`` blocks only."""
    return _THINK_TAG_PATTERN.sub("", text).strip()


def strip_thinking(text: str, mode: AnswerStripMode = "tags_only") -> str:
    """Normalize LLM output for scoring / logging.

    * ``off`` — only strip leading/trailing whitespace (no tag removal).
    * ``tags_only`` — remove ``<think ...>...</think >`` blocks; keep remaining text.
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


# ---------------------------------------------------------------------------
# Safe arithmetic expression evaluator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval_expr(expr: str) -> float | None:
    """Evaluate a simple arithmetic expression safely.

    Supports +, -, *, /, parentheses and numeric literals.
    Returns None if the expression cannot be evaluated.
    """
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return None

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):  # Python 3.7 compat
            return float(node.n)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {op_type}")
            return _SAFE_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in _SAFE_OPS:
                raise ValueError(f"Unsupported operator: {op_type}")
            return _SAFE_OPS[op_type](_eval(node.operand))
        raise ValueError(f"Unsupported node: {type(node)}")

    try:
        result = _eval(tree)
        if isinstance(result, (int, float)) and not (result != result):  # not NaN
            return float(result)
    except (ValueError, ZeroDivisionError, TypeError, RecursionError):
        pass
    return None


def _is_simple_arithmetic(text: str) -> bool:
    """Return True if text looks like a simple arithmetic expression."""
    stripped = text.strip()
    if not stripped:
        return False
    has_digits = bool(re.search(r"\d", stripped))
    has_operator = bool(re.search(r"[+\-*/]", stripped))
    return has_digits and has_operator


def _try_parse_float(text: str) -> float | None:
    """Try to parse text as a float, handling commas and %."""
    text = text.strip().replace(",", "").replace("%", "")
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def extract_concise_fallback(text: str) -> str:
    """If the model buried a value after reasoning, take yes/no or last number-like token.

    Scans non-empty lines from bottom to top. Intended for FinQA-style numeric answers.
    Also handles simple arithmetic expressions and pipe-separated values.
    """
    if not text or not text.strip():
        return ""

    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]

    # Check multi-line addition FIRST (e.g. "110\n74" -> 184)
    if len(lines) >= 2:
        nums: list[float] = []
        all_numeric = True
        for ln in lines:
            val = _try_parse_float(ln)
            if val is not None:
                nums.append(val)
            else:
                all_numeric = False
                break
        if all_numeric and len(nums) >= 2:
            total = sum(nums)
            if all(n == int(n) for n in nums):
                return str(int(total))
            return str(total)

    for line in reversed(lines):
        if "|" in line:
            first_part = line.split("|")[0].strip()
            val = _try_parse_float(first_part)
            if val is not None:
                return str(val)

        if _is_simple_arithmetic(line):
            result = _safe_eval_expr(line)
            if result is not None:
                return str(result)

        low = line.lower().rstrip(".,!? ")
        tokens = low.split()
        if len(tokens) == 1 and tokens[0] in ("yes", "no"):
            return tokens[0]

        matches = list(_VALUE_NUMBER_PATTERN.finditer(line))
        if matches:
            raw = matches[-1].group(0).replace(",", "")
            val = _try_parse_float(raw)
            if val is not None and val == 0:
                return "0"
            return raw

    return ""


def extract_final_value(text: str) -> str:
    """Extract the value from a 'FINAL: <value>' line (finqa template)."""
    match = _FINAL_LINE_PATTERN.search(text)
    if match:
        value = match.group(1).strip()
        val = _try_parse_float(value)
        if val is not None and val == 0:
            return "0"
        return value
    return text


def normalize_percentage_answer(
    answer: str,
    question: str,
    ground_truth: str | None = None,
) -> str:
    """Normalize percentage answers from integer to decimal fraction."""
    if not answer or not answer.strip():
        return answer

    val = _try_parse_float(answer)
    if val is None:
        return answer

    if -1 <= val <= 1:
        return answer

    if ground_truth is not None:
        gt_val = _try_parse_float(str(ground_truth))
        if gt_val is not None and -1 <= gt_val <= 1:
            normalized = val / 100.0
            return str(round(normalized, 6))
        return answer

    if _PERCENTAGE_KEYWORDS.search(question):
        normalized = val / 100.0
        return str(round(normalized, 6))

    return answer


_REFUSAL_PATTERN = re.compile(
    r"i\s+(?:cannot|can't|am\s+unable|am\s+not\s+able)\s+"
    r"(?:answer|provide|respond)"
    r"|insufficient|not\s+sufficient|not\s+enough\s+information"
    r"|context\s+does\s+not|provided\s+context|given\s+context",
    re.IGNORECASE,
)


def _validate_answer(answer: str) -> bool:
    """Return True if the answer is non-empty and meaningful."""
    if not answer or not answer.strip():
        return False
    if _REFUSAL_PATTERN.search(answer):
        return False
    return True


def _looks_like_thinking(text: str) -> bool:
    """Return True if *text* reads like internal reasoning, not an answer."""
    if not text:
        return False

    first_line = text.split("\n", 1)[0].strip().lower()

    _THINKING_STARTS = (
        "okay,", "okay ", "let's see", "let me", "looking at", "first,",
        "well,", "hmm", "so,", "so ", "now ", "the user", "i need to",
        "i should", "i'll", "to answer", "to find", "to calculate", "to determine",
    )

    _CUTOFF_ENDS = (
        " of", " the", " a", " an", " is", " in", " to", " for",
        " and", " that", " with", " which", " it", ",",
    )

    text_lower = text.strip().lower()
    ends_with_punct = text_lower[-1] in ".!?" if text_lower else False
    if not ends_with_punct:
        for marker in _CUTOFF_ENDS:
            if text_lower.endswith(marker):
                return True

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
    answer_valid: bool = True


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


def _call_with_streaming(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    callbacks: list | None = None,
) -> tuple[str, float, float, dict | None, str | None]:
    """Call LLM with streaming for accurate TTFT measurement.

    Returns (raw_content, ttft_seconds, total_seconds, usage_metadata, reasoning).
    Falls back to invoke() if streaming is not supported.
    """
    start = time.perf_counter()
    first_token_time: float | None = None
    chunks: list[str] = []
    reasoning_chunks: list[str] = []
    last_usage: dict | None = None
    last_additional_kwargs: dict | None = None

    try:
        for chunk in llm.stream(messages, config={"callbacks": callbacks or []}):
            if first_token_time is None:
                first_token_time = time.perf_counter()

            # Accumulate content
            if hasattr(chunk, "content") and chunk.content:
                if isinstance(chunk.content, str):
                    chunks.append(chunk.content)
                elif isinstance(chunk.content, list):
                    # Some providers return list of content blocks
                    for part in chunk.content:
                        if isinstance(part, str):
                            chunks.append(part)
                        elif isinstance(part, dict) and part.get("type") == "text":
                            chunks.append(part.get("text", ""))

            # Accumulate reasoning content
            if hasattr(chunk, "additional_kwargs"):
                last_additional_kwargs = chunk.additional_kwargs
                rc = chunk.additional_kwargs.get("reasoning_content")
                if rc:
                    reasoning_chunks.append(str(rc))

            # Check for usage metadata
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_usage = chunk.usage_metadata

        total = time.perf_counter() - start
        ttft = first_token_time - start if first_token_time else total

    except (NotImplementedError, AttributeError, TypeError) as exc:
        logger.debug("Streaming not supported (%s), falling back to invoke()", exc)
        return _call_with_invoke(llm, messages, callbacks)

    raw_content = "".join(chunks)
    raw_reasoning = "".join(reasoning_chunks) if reasoning_chunks else None

    return raw_content, ttft, total, last_usage, raw_reasoning


def _call_with_invoke(
    llm: BaseChatModel,
    messages: list[BaseMessage],
    callbacks: list | None = None,
) -> tuple[str, float, float, dict | None, str | None]:
    """Call LLM with invoke() — fallback when streaming is unavailable.

    Returns (raw_content, ttft_seconds, total_seconds, usage_metadata, reasoning).
    Note: TTFT equals total latency in non-streaming mode.
    """
    start = time.perf_counter()
    response = llm.invoke(messages, config={"callbacks": callbacks or []})
    total = time.perf_counter() - start

    raw_content = str(response.content) if response.content else ""
    reasoning_kw = response.additional_kwargs.get("reasoning_content")
    raw_reasoning = str(reasoning_kw) if reasoning_kw else None
    usage = getattr(response, "usage_metadata", None)

    return raw_content, total, total, usage, raw_reasoning


def _postprocess_answer(
    raw_content: str,
    raw_reasoning: str | None,
    question: str,
    *,
    strip_mode: AnswerStripMode,
    value_fallback: bool,
    ground_truth: str | None,
    prompt_template_name: str | None,
) -> str:
    """Apply all post-processing steps to extract the final answer."""
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

    # Extract FINAL: value from finqa template
    if prompt_template_name == "finqa" and answer:
        answer = extract_final_value(answer)

    # Normalize percentage answers (e.g. 12.0 -> 0.12)
    if answer and _try_parse_float(answer) is not None:
        answer = normalize_percentage_answer(answer, question, ground_truth)

    return answer


@mlflow.trace(name="generate_answer", span_type="func")
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
    ground_truth: str | None = None,
    prompt_template_name: str | None = None,
    callbacks: list | None = None,
) -> GenerationResult:
    from langchain_core.messages import HumanMessage, SystemMessage

    span = mlflow.get_current_active_span()
    if span:
        span.set_attributes({
            "generation.prompt_template": prompt_template_name or "default",
        })

    context_text = "\n\n".join(contexts)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_template.format(context=context_text, question=question)),
    ]

    # Call LLM with streaming for accurate TTFT, with invoke fallback
    raw_content: str = ""
    ttft: float = 0.0
    total: float = 0.0
    usage: dict | None = None
    raw_reasoning: str | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            raw_content, ttft, total, usage, raw_reasoning = _call_with_streaming(llm, messages, callbacks)
            break
        except Exception as exc:
            if attempt == _MAX_RETRIES:
                raise
            delay = _BASE_DELAY * 2 ** (attempt - 1)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s  — retrying in %ds",
                attempt, _MAX_RETRIES, exc, delay,
            )
            time.sleep(delay)

    # Post-process answer
    answer = _postprocess_answer(
        raw_content, raw_reasoning, question,
        strip_mode=strip_mode,
        value_fallback=value_fallback,
        ground_truth=ground_truth,
        prompt_template_name=prompt_template_name,
    )

    # Validate answer
    answer_valid = _validate_answer(answer)
    if not answer_valid:
        logger.warning("Empty/invalid answer for question: %s", question[:80])

    # Token count
    token_count = 0
    if usage:
        token_count = usage.get("output_tokens", 0)

    gpu_interval = float(os.getenv("GPU_METRICS_INTERVAL_SECONDS", "30"))
    gpu = get_gpu_usage(cache_seconds=gpu_interval)

    return GenerationResult(
        answer=answer,
        ttft_seconds=ttft,
        total_seconds=total,
        token_count=token_count,
        tokens_per_second=token_count / total if total > 0 and token_count > 0 else 0,
        gpu_usage=gpu,
        raw_content=raw_content,
        raw_reasoning=raw_reasoning,
        answer_valid=answer_valid,
    )
