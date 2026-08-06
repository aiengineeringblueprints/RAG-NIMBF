# Notes — TRACe framework (RAGBench §3.2)

Implementation of the two new TRACe dimensions that RAGAS does not
already cover: **Utilization (T)** and **Completeness (C)**. The other
two TRACe dimensions map to existing RAGAS metrics:

| TRACe dimension   | Implementation in this repo               |
| ----------------- | ----------------------------------------- |
| Adherence         | `ragas_faithfulness`                      |
| Relevance         | `ragas_context_recall`                    |
| **Utilization**   | `trace_utilization` (new)                 |
| **Completeness**  | `trace_completeness` (new)                |

## Files changed

| File                                | Change                                                |
| ----------------------------------- | ----------------------------------------------------- |
| `benchmark/trace_metrics.py`        | NEW. `TraceMetricsResult`, `compute_trace_metrics`.   |
| `main.py`                           | Imports module; adds stage 4c (TRACe) after custom    |
|                                     | metrics. Merges TRACe results into `custom_result`    |
|                                     | so MLflow / CSV / JSON export work unchanged.         |
| `config.py`                         | New field `trace_metrics_enabled: bool = False` and   |
|                                     | matching `TRACE_METRICS_ENABLED` env var.             |
| `benchmark/orchestration/matrix.py` | YAML settings coercion for the new boolean flag.      |
| `tests/test_trace_metrics.py`       | NEW. 18 tests covering heuristic primitives, edge     |
|                                     | cases, refusal answers, full utilization, mocked LLM  |
|                                     | judge, parser unit tests.                             |

No new dependencies required.

## How to enable

Opt-in via YAML:

```yaml
settings:
  trace_metrics_enabled: true
  eval_critic_llm: ollama:gemma3:12b
```

Or via environment variable:

```bash
export TRACE_METRICS_ENABLED=true
```

When enabled, the framework builds a critic LLM using the same wiring
as the RAGAS evaluation (`eval_critic_llm`, per-role URLs and keys),
so no separate endpoint is needed. If the critic cannot be initialised
or its responses fail to parse, the module falls back to a heuristic
token-overlap score and records a soft error.

## How it integrates

1. `main.py` runs TRACe after RAGAS and after custom metrics.
2. TRACe results are merged into the existing `CustomMetricsResult`:
   - Means land in `custom_metric_means` under keys `trace_utilization`
     and `trace_completeness`.
   - Per-sample scores land in `custom_scores` next to IR/NLG scores.
3. `benchmark/tracking.py` already logs every `custom_*` key to MLflow
   with the `custom_` prefix, so TRACe scores appear as
   `custom_trace_utilization` / `custom_trace_completeness` in MLflow
   runs without any tracking-side change.
4. CSV / JSON / Markdown exports (`benchmark/reporting/exports.py`)
   pick them up through the same `custom_metric_means` channel.

A new `stage_timings` entry `trace_metrics` records wall-clock time.

## LLM-judge design

A single prompt per sample elicits both Utilization and Completeness in
one round-trip. The judge returns compact JSON:

```json
{"utilization": <float 0..1>, "completeness": <float 0..1>}
```

The parser is tolerant of surrounding prose and string-typed numbers.
Out-of-range values are clamped only inside `[−0.05, 1.05]`; larger
excursions are rejected and the sample falls back to the heuristic.

The prompt caps the context window at 6 000 characters to keep the
prompt token-efficient on local 4B–12B critic models.

## Heuristic fallback

When the judge is unavailable:

- **Utilization** = fraction of answer tokens that appear in the
  context. This is the symmetric view of the paper's definition (paper
  measures context tokens used by the answer). The answer-side view is
  more stable to answer length and avoids penalising concise correct
  answers.
- **Completeness** = fraction of context 4-grams covered by the answer.
  Long contexts yield many n-grams, giving a fine-grained coverage
  signal.

## Example output (heuristic mode)

```
custom_metric_means = {
  ...,
  "trace_utilization":  0.83,
  "trace_completeness": 0.41,
}
per_sample[0]["custom_scores"] = {
  ...,
  "trace_utilization": 0.90,
  "trace_completeness": 0.35,
  "trace_mode": 0.0,        # 0.0 = heuristic, 1.0 = llm judge
}
```

## Limitations vs. paper

1. **Span-level attribution.** RAGBench annotates which response token
   *spans* are attributable to context. The LLM judge here returns a
   single scalar per axis, not spans. This matches how the paper
   *scores* systems but loses the diagnostic detail of the span
   markup.
2. **Utilization definition.** The heuristic inverts the paper's
   direction (answer-tokens-over-context rather than
   context-tokens-over-answer) for stability with short answers. The
   LLM-judge mode follows the paper's spirit: it asks what fraction of
   the answer is supported by the context.
3. **No utilised-token counting in judge mode.** The judge emits a
   scalar in `[0, 1]`; we do not ask it to enumerate spans. This was a
   deliberate token-budget trade-off for local critics.
4. **Single-axis relevance gating for Completeness.** The paper's
   Completeness is conditioned on context relevance. We rely on the
   judge to "not penalise the answer for context that is irrelevant to
   the implicit question" rather than scoring relevance separately.
5. **No TRACe overall score.** RAGBench reports a combined TRACe
   score; this implementation exposes the two new components only.
   Adherence/Relevance already live as separate RAGAS metrics, so a
   downstream consumer can combine all four if desired.
