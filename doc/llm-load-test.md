# LLM Load Testing

The framework ships an integrated version of `LLM_Performance_Tests-main` that
tests every selected generator LLM with the current endpoint, dataset
questions, and `max_new_tokens` value.

Enable it in the manifest:

```yaml
settings:
  llm_performance_enabled: true
  llm_performance_source: generation
  llm_performance_call_counts: [1, 3, 6, 10]
  llm_performance_warmup: true
  llm_performance_timeout_seconds: 60
```

The same settings can be supplied through `.env`:

```bash
LLM_PERFORMANCE_ENABLED=true
LLM_PERFORMANCE_SOURCE=generation
LLM_PERFORMANCE_CALL_COUNTS=1,3,10
python main.py
```

## Sources

With `llm_performance_source: generation`, the framework reuses the real RAG
answer-generation calls, including their complete context prompts. It adds no
LLM requests and records mean/median/P95 latency, TTFT, estimated TPOT, output
tokens/s, requests/s, success rate, and generation wall time.

Set `llm_performance_source: load_test` when you explicitly need the additional
sequential-versus-parallel load profiles. For each configured call count, that
mode runs distinct dataset questions once sequentially and once concurrently
and also reports parallel speedup and inter-token latency.

## Results

Results appear in the terminal report, MLflow metrics (prefix `llm_perf_`), the
summary CSV, the aggregate JSON, and detailed files under
`results/runN/llm_performance/`.

In `load_test` mode, an identical
`(provider, model, endpoint, max_new_tokens, load profile)` is measured only
once per `main.py` run and reused across configurations. In `generation` mode,
every RAG configuration keeps its own observed timings because context and
prompt length can differ. External black-box RAG adapters are skipped because
their internal generator endpoint cannot be inferred safely.
