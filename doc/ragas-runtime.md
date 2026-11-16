# RAGAS Runtime: Metric Selection, Critic Workers, Sample Size

RAGAS with a critic LLM dominates runtime on long-form datasets. On RAGBench
TechQA (top-k 10) one config spends ~3 h of ~3.5 h in `ragas_eval`
(13 M input / 2.1 M output critic tokens per config, 500 questions); see
[enterprise-rag-techqa-refusal-analysis.md](enterprise-rag-techqa-refusal-analysis.md#3-runtime).
This guide lists the three levers and what each costs.

## 1. Select which RAGAS metrics run

| Setting | Where | Default |
|---|---|---|
| `ragas_metrics` | manifest `settings:` (list or comma string) | full set |
| `RAGAS_METRICS` | `.env` / environment (comma-separated) | full set |

Available names (the full set, in this order): `faithfulness`,
`answer_relevancy`, `answer_correctness`, `context_precision`,
`context_recall`, `semantic_similarity`.

```yaml
settings:
  ragas_enabled: true
  ragas_metrics: [faithfulness, answer_correctness, context_recall, semantic_similarity]
```

Rules:

- Blank or missing (`RAGAS_METRICS=` / `ragas_metrics: []`) means the full
  set. To skip RAGAS entirely use `ragas_enabled: false`.
- Unknown or duplicate names fail config validation.
- Put `ragas_metrics` under `settings:`, not `matrix:`. Under `matrix:` a list
  is expanded into one config per element, i.e. one metric per config.
- Skipped metrics are never built, so they cost no critic calls. They stay
  `None` in results: absent from MLflow, `null` in JSON, empty cells in CSV,
  dropped from the markdown RAGAS table, and left out (not counted as 0) of
  composite rankings and paper charts.
- Resume (`worker --run-dir`) identifies configs by name, which does not
  include the metric selection. Changing `ragas_metrics` for an existing run
  directory does not re-evaluate completed configs; use a new run directory.

Relative critic cost per metric (TechQA, 10 contexts/question):

- `context_precision`: one critic call **per retrieved context** — the
  largest single cost at top-k 10–12, and it grows with top-k.
- `faithfulness`, `answer_correctness`, `context_recall`: statement
  extraction plus per-statement verdicts; grow with answer/reference length.
- `answer_relevancy`: generates several questions per answer plus embeddings.
- `semantic_similarity`: embeddings only, no critic LLM call.

## 2. `EVAL_CRITIC_MAX_WORKERS` against the hosted critic

`EVAL_CRITIC_MAX_WORKERS` (manifest `eval_critic_max_workers`) is the number
of parallel RAGAS critic jobs. Default `1` suits a local, serial critic; the
hosted coassist.de endpoint (`openai:deepseek-v4-flash`) ran with `8`.

### Measurement (2026-11-16)

Critic-only replay: the first 10 TechQA samples of
`results/Enterprise-RAG/run6/benchmark_20261111_180200.json` (stored Blueprint
answers, 10 contexts each) were fed straight into `evaluate_results`, so no
retrieval or generation time is included. Critic `openai:deepseek-v4-flash`
(thinking control `none`, max tokens 16384, retries 10), embeddings
`nomic-embed-text` on local Ollama. One repetition per setting, run from a
workstation.

| Setting | Workers | Metrics | Wall time | Critic input / output tokens | Errors |
|---|---|---|---|---|---|
| baseline | 8 | all 6 | 224 s (22.4 s/q) | 592 k / 39 k | none |
| more workers | 16 | all 6 | 184 s (18.4 s/q) | 592 k / 40 k | none |
| more workers, no `context_precision` | 16 | 5 | 105 s (10.5 s/q) | 332 k / 31 k | none |

The baseline matches run6 (10,692 s / 500 questions = 21.4 s/q at 8 workers).

Findings:

- 16 workers are 18 % faster than 8, not 2×. Every metric got a valid score
  for all 10 samples at 16 workers (RAGAS retries, if any, are not counted). With only 10
  samples (~60 metric jobs) the slowest jobs dominate the tail, so the gain on
  500 questions is likely somewhat larger, but expect well below linear.
- Dropping `context_precision` removes 44 % of critic input tokens and,
  together with 16 workers, halves the wall time (−53 % vs. baseline).
- Scores move between repetitions even at critic temperature 0
  (faithfulness 0.80 vs. 0.78, context recall 0.60 / 0.63 / 0.54 on the same
  10 samples). Treat differences below ~0.03–0.05 on small samples as noise.

### Recommendation

- Hosted critic (coassist.de, `deepseek-v4-flash`): set
  `EVAL_CRITIC_MAX_WORKERS=16`. Going higher is untested; try 24 only with a
  watch on 429/5xx retries in the logs.
- Local Ollama critic: keep `1`–`2`; Ollama serves requests mostly serially.
- Chunking / top-k sweeps: drop `context_precision`
  (`ragas_metrics: [faithfulness, answer_relevancy, answer_correctness, context_recall, semantic_similarity]`)
  and use `sample_size` 150 (section 3). Estimate per TechQA config:
  150 × 10.5 s ≈ 26 min of `ragas_eval` instead of ~3 h. Re-run the final
  candidates with the full set and 500 questions.

## 3. Smaller `sample_size` for sweeps

Critic cost is linear in the number of questions. For chunking / top-k sweeps
use `sample_size` 150–200, then re-run only the best configs with the full
500. Sampling is deterministic (fixed shuffle seed or first rows), so a
smaller `sample_size` gives the leading subset of the larger one and configs
with the same `sample_size` see the same questions. Pin `sample_size` in the
manifest — `.env` may carry another dataset's `DATASET_SAMPLE_SIZE`.

At ~21 s critic time per TechQA question with 8 workers (run6), 150 questions
take ~55 min of `ragas_eval` instead of ~3 h for 500.
