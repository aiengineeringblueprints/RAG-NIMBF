# Ragas OutputParserException — StringIO Validation Error

## Error

```
OutputParserException: Failed to parse StringIO from completion
{"question": "What was the revenue in 2007 and 2006?", "noncommittal": 0}.
Got: 1 validation error for StringIO
text
  Field required [type=missing, input_value={...}, input_type=dict]
```

## Root Cause

The evaluation LLM returns structured JSON instead of the plain-text format Ragas expects. Ragas's `StringIO` parser requires a `text` field, but the model outputs a dict like `{"question": "...", "noncommittal": 0}`.

This is a **model compatibility issue** — the model doesn't consistently follow Ragas's prompt instructions for output formatting.

## Impact

- Non-fatal: Ragas logs the error and skips the affected sample.
- `max_retries=2` in `RunConfig` retries before giving up.
- Skipped samples get `None` scores in the results.
- Partial results are still collected (observed ~50% success rate).

## Fixes

### 1. Use a stronger eval model (Recommended)

Set `EVAL_CRITIC_LLM` to a more instruction-following model:

```env
EVAL_CRITIC_LLM=gemma3:27b
```

Larger models follow formatting instructions more reliably.

### 2. Increase max tokens

The model may truncate output and fall back to JSON:

```env
EVAL_CRITIC_MAX_TOKENS=8192
```

### 3. Accept partial results

If the error rate is low, the evaluation still produces usable metrics. Check `samples_with_valid_scores` in the results to gauge coverage.

## Affected Code

- `benchmark/evaluation.py` — `RunConfig` at line 121
- `benchmark/providers.py` — `ChatOpenAI` creation at line 77
- `config.py` — `EVAL_CRITIC_LLM` and `EVAL_CRITIC_MAX_TOKENS` env vars
