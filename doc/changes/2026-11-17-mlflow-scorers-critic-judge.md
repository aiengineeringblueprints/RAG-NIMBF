# Post-run MLflow scorers judged by the critic LLM

Date: 2026-11-17

## Summary
Adds an opt-in post-run step (issue #44, parent #40) that scores a finished
config run's per-sample traces with MLflow built-in judges and
RAGAS-via-MLflow scorers, selected in YAML via `mlflow_scorers`. The judge is
the configured critic LLM, so local OpenAI-compatible models (Ollama, vLLM,
NIM on the Spark) work. Results land on the same MLflow run as
`mlflow_judge_<name>_mean`, next to the direct `ragas_*` metrics. The
env-only judge prototype in `benchmark/tracking.py` is removed.

## Changes

### Config
- `config.py`: `MLFLOW_SCORER_NAMES`, `parse_mlflow_scorers`, new field
  `BenchmarkConfig.mlflow_scorers` (default `()`), env fallback
  `MLFLOW_SCORERS`. Validation rejects unknown and duplicate names and a
  non-empty selection without `mlflow_tracing_enabled` (scorers read traces).
- `benchmark/orchestration/matrix.py`: manifest coercion (list or
  comma-separated string).

### Scoring step
- New `benchmark/mlflow_scorers.py`:
  - `resolve_judge_endpoint(config)` maps `eval_critic_llm` + critic URL/key
    resolution onto MLflow's `openai:/<model>` provider. `ollama:` critics get
    `<base>/v1`; `openai:` critics keep their compat base URL; keyless local
    servers get a placeholder key (MLflow's OpenAI provider requires one).
    Built-in judges also get `eval_critic_temperature` / `eval_critic_max_tokens`.
  - `SCORER_FACTORIES` / `build_scorers` map config names to scorer classes.
    RAGAS-via-MLflow scorers that need embeddings are left out (MLflow would
    default them to OpenAI embeddings).
  - `score_run_traces(run_id, names, endpoint)` reads the run-linked traces
    (`search_traces(run_id=...)`), runs `mlflow.genai.evaluate(data=traces)`
    inside the config run, and logs `mlflow_judge_<name>_mean`. The
    `OPENAI_API_BASE` / `OPENAI_API_KEY` overrides are scoped to construction
    and scoring and restored afterwards. Failures are tagged
    (`mlflow_scorers=failed`) and never fail the benchmark.
  - `score_config_run(run_id, config)` — entry point used by the worker.
- `benchmark/orchestration/worker.py`: after `log_benchmark_run`, calls
  `score_config_run` when the config selects scorers.

### Removed prototype
- `benchmark/tracking.py`: `_log_genai_rag_judges` removed together with
  `MLFLOW_GENAI_JUDGES_ENABLED` / `MLFLOW_GENAI_JUDGE_MODEL`. It replayed
  per-sample outputs through a fake RETRIEVER span and judged with a hosted
  default model (`openai:/gpt-4o-mini`); the new step scores the real traces
  from #42/#43 with the critic LLM.

### Docs and manifests
- `doc/results-and-mlflow.md`: new section *MLflow scorers (post-run judges)*.
- `doc/env-vars.md`: `MLFLOW_SCORERS` row (the README links this table).
- `doc/00-README.md`: index entry updated.
- `experiments/mlflow-scorers-smoke.yaml`: smoke manifest.

## Behaviour changes and migration
- `MLFLOW_GENAI_JUDGES_ENABLED` / `MLFLOW_GENAI_JUDGE_MODEL` no longer have any
  effect, and the `mlflow_genai_judges` run tag is gone. Use
  `mlflow_tracing_enabled: true` plus `mlflow_scorers: [retrieval_relevance,
  retrieval_groundedness, retrieval_sufficiency]` instead; the judge is now
  `eval_critic_llm`.
- Default is off; existing manifests are unaffected.

## PR #53 review fixes
- `openai:` critic without a compat base URL (hosted OpenAI): previously the
  judge env was set to an empty base URL and the placeholder key, overriding
  the user's real `OPENAI_API_KEY`. `JudgeEndpoint.base_url` / `api_key` are
  now optional and `judge_environment` only overrides non-`None` values; the
  placeholder key is used only for custom endpoints.
- `score_config_run` is fully non-fatal: endpoint resolution errors are logged
  and skipped, so a completed config is never marked failed by the judge step.
- A scorer whose `<ScorerName>/mean` metric is missing is logged as a warning
  instead of being dropped silently.
- Tests: `test_hosted_openai_critic_keeps_the_real_openai_environment`,
  `test_endpoint_resolution_failure_is_non_fatal`,
  `test_scorer_without_a_mean_metric_is_reported`.

## Verification
- Tests (all on the DGX Spark, MLflow 3.18):
  - `tests/test_config.py::TestMlflowScorersSetting` — default, manifest list
    and string, env fallback, unknown/duplicate rejected, tracing required.
  - `tests/test_mlflow_scorers.py` — judge endpoint resolution (Ollama `/v1`,
    shared Ollama URL fallback, OpenAI-compatible URL/key, placeholder key,
    inference params); scoring a real SQLite-backed config run produced by
    `run_single_benchmark` with a deterministic stand-in scorer (metric on the
    same run, generator not called again, critic endpoint visible during
    scoring, env restored, failure non-fatal); every configured name builds an
    MLflow scorer.
  - `tests/test_worker_loop.py` — worker scores each logged config run only
    when scorers are configured.
  - Full suite: 987 passed.
- Smoke run on the Spark: `experiments/mlflow-scorers-smoke.yaml`,
  `--run-dir results/run-mlflow-scorers-smoke`, fresh SQLite store, 3 SQuAD
  questions, generator `openai:qwen3.8-27b`, critic/judge local Ollama
  `gemma3:12b` (`openai:/gemma3:12b` at `localhost:11434/v1`). All 9 scorers
  ran in ~2 min; config run tagged `mlflow_scorers=logged`,
  `mlflow_tracing=linked_3`, with `mlflow_judge_*_mean`: correctness 1.0,
  relevance_to_query 1.0, retrieval_groundedness 1.0, retrieval_relevance
  0.333, retrieval_sufficiency 1.0, ragas_faithfulness 1.0,
  ragas_context_precision 1.0, ragas_context_recall 1.0,
  ragas_factual_correctness 0.357. Every trace carried yes/no judgements with
  rationales (built-in) and numeric RAGAS assessments; no extra MLflow runs
  were created (parent, config, summary only).

## Follow-ups
- `eval_critic_thinking_control` is not forwarded to the judges, and the
  RAGAS-via-MLflow scorers get no temperature/max_tokens. Reasoning critics
  (Qwen3, GLM on vLLM/NIM) may emit reasoning text the judge parser rejects;
  prefer a non-reasoning critic for `mlflow_scorers` until this is wired.
- A vLLM/NIM critic must use the `openai:` prefix; unprefixed / `ollama:`
  critics get `/v1` appended to the Ollama base URL.
- MLflow also logs its own `<ScorerName>/mean` keys on the run; the
  documented, prefixed names are `mlflow_judge_<name>_mean`.
- A CLI to score an older run by id (`score_run_traces` already supports it).
- Embedding-based RAGAS-via-MLflow scorers would need an embedding override
  pointing at `eval_critic_embedding`.
