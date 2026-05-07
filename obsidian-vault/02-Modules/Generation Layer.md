# Generation Layer

Source: [benchmark/generation.py](../benchmark/generation.py)

The generation layer owns model invocation, streaming metrics, answer cleanup, and validity checks.

Key responsibilities:

- Build generator LLM with `get_llm()`.
- Prefer streaming through `_call_with_streaming()` when available.
- Fall back to normal invoke through `_call_with_invoke()`.
- Capture answer text, raw content, reasoning content, time-to-first-token, total latency, token count, tokens per second, and GPU usage.
- Strip thinking content and apply answer-specific fallbacks with `_postprocess_answer()`.

Answer cleanup:

- `strip_think_tags()` removes explicit `<think>...</think>` blocks.
- `strip_thinking()` supports modes from [[Configuration Reference]].
- `extract_concise_fallback()` and `extract_final_value()` help handle verbose reasoning models.
- `normalize_percentage_answer()` attempts consistent numeric/percentage formatting.
- `_validate_answer()` marks empty/refusal/thinking-like answers as invalid.

Result model:

- `GenerationResult` carries final answer, raw content, reasoning, timing, token metrics, GPU usage, and validity.

Related notes:

- [[Providers and Models]]
- [[Prompt Templates]]
- [[Evaluation and Metrics]]

