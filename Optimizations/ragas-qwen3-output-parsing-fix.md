# RAGAS Output Parsing Failure with Qwen3 on vLLM

**Date:** 2026-04-02
**Status:** Fixed
**Affected versions:** ragas 0.4.3, langchain-openai 1.1.12, langchain-core 1.2.23

---

## Problem

When using Qwen3-32B-AWQ (served via vLLM) as the RAGAS critic/evaluator LLM, two categories of output parsing errors occur during evaluation:

### Error 1: `fix_output_format` parse failure

```
[ERROR] ragas.prompt.pydantic_prompt: Prompt fix_output_format failed to parse output:
The output parser failed to parse the output including retries.
[ERROR] ragas.prompt.pydantic_prompt: Prompt response_relevance_prompt failed to parse output:
The output parser failed to parse the output including retries.
[ERROR] ragas.executor: Exception raised in Job[61]:
RagasOutputParserException(The output parser failed to parse the output including retries.)
```

### Error 2: `StringIO` validation error

```
[ERROR] ragas.executor: Exception raised in Job[63]: OutputParserException(
Failed to parse StringIO from completion
{"classifications": [{"statement": "-0.07329842931937183",
"reason": "...", "attributed": 1}]}.
Got: 1 validation error for StringIO
text
  Field required [type=missing, input_value={'classifications': [...]}},
  input_type=dict]
```

---

## Root Cause

Two interacting issues:

1. **Intermittent malformed output (Error 1):** The `answer_relevance` metric asks the critic LLM to generate questions from the answer and return structured JSON. Smaller or less capable models occasionally produce output that deviates from the expected JSON schema. RAGAS retries with a `fix_output_format` prompt, but if that also fails, the sample is marked as failed. This is inherent to using local models and is handled gracefully by RAGAS (failed samples get `None` scores and are excluded from the mean).

2. **Auto-parsed JSON content (Error 2):** The primary fixable issue. Qwen3 models natively output structured JSON. The `langchain-openai 1.1.x` client auto-detects this and parses the JSON string into a Python `dict` or `list` in `message.content`. RAGAS's output parser expects `message.content` to be a raw string — when it receives a pre-parsed dict instead, the Pydantic `StringIO` validator fails because it can't find the `text` field.

### Data flow

```
vLLM response (JSON string)
  -> langchain-openai ChatOpenAI (auto-parses JSON to dict)
    -> message.content = {"classifications": [...]}  (dict, not string)
      -> RAGAS output parser (expects string)
        -> CRASH: StringIO validation error
```

---

## Fix

A wrapper class `_ContentAsStringChatModel` in `benchmark/providers.py` that intercepts the LLM response and coerces any non-string `message.content` back to a JSON string before RAGAS processes it.

### Files changed

| File | Change |
|------|--------|
| `benchmark/providers.py` | Added `_ContentAsStringChatModel` wrapper class and `wrap_for_ragas()` helper |
| `benchmark/evaluation.py` | Wrapped critic chat model: `LangchainLLMWrapper(wrap_for_ragas(critic_chat))` |

### The wrapper

```python
class _ContentAsStringChatModel(BaseChatModel):
    """Ensures message.content is always a string.

    Coerces dict/list content (from auto-parsed JSON) back to a JSON string
    so RAGAS's output parser receives the format it expects.
    """

    _wrapped: BaseChatModel

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = self._wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._coerce_result(result)

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        result = await self._wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._coerce_result(result)

    @staticmethod
    def _coerce_result(result):
        for gen in result.generations:
            msg = gen.message
            if not isinstance(msg.content, str):
                text = json.dumps(msg.content)
                # Replace with string-content AIMessage
                ...
        return result
```

### Where it's applied

```python
# benchmark/evaluation.py
critic_chat = get_chat_model(...)
critic_llm = LangchainLLMWrapper(wrap_for_ragas(critic_chat))  # <-- wrapped here
```

Only the critic/evaluator LLM is wrapped. The generator LLM (question answering) does not need it.

---

## Notes

- Error 1 (intermittent parse failures) may still occur with very small models. This is expected behavior — RAGAS handles it by assigning `None` scores to failed samples. Using a larger/capable critic model (27B+) minimizes these failures.
- Error 2 is fully resolved by the wrapper.
- The wrapper adds negligible overhead (just a type check and potential `json.dumps` per response).
