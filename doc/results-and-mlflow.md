# Results, MLflow, and Post-Processing

## Output layout

Each run writes artifacts to `results/runN/`, including per-config JSON files,
QA logs, CSV/Markdown summaries, plots, reproducibility manifests, and MLflow
run data.

## Compacted reports

Each `benchmark_*.json` report holds only the summary; per-sample data
(answers, contexts, scores) sits next to it in `benchmark_*.samples.json.gz`
with each distinct context stored once. Load both with
`benchmark.reporting.exports.load_json_report(path)`. Older reports that still
inline `per_sample` load the same way and can be shrunk losslessly with
`python scripts/compact_results.py [results/...]` (`--dry-run` lists
candidates).

## LLM Answer Judge (re-scoring finished runs)

`answer_match` is a deterministic string-match scorer. When references are
short labels ("Yes"/"no") and the system answers in prose, it under-scores
heavily. The LLM answer judge re-scores the answers of finished runs as
`correct`, `incorrect`, or `declined` — offline, without a new benchmark run:

```bash
python scripts/judge_answers.py results/run1 results/run2
```

The script uses `EVAL_CRITIC_LLM` and its endpoint from `.env` (`--critic`
overrides, `--workers` defaults to `EVAL_CRITIC_MAX_WORKERS`). Each distinct
`(question, answer)` pair is judged once; verdicts are appended to a JSONL
cache as they arrive, so an interrupted judging run resumes. Samples with
`expected_behavior: decline` count as correct when judged declined. The judge
reports per-variant and per-category `judge_correct`/`judge_incorrect`/
`judge_declined` rates next to the stored `answer_match` score and never
modifies the stored run data.

## MLflow Run Comparison

Start the MLflow UI against the local SQLite store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

The framework requires MLflow ≥ 3.18. If MLflow reports an out-of-date
database schema (e.g. an `mlflow.db` created with an older version), back the
file up and migrate it once — the migration is one-way:

```bash
cp mlflow.db mlflow.db.backup-$(date +%Y%m%d)
mlflow db upgrade sqlite:///mlflow.db
```

See `changes/2026-11-17-mlflow-3-18-upgrade.md`.

In the `RAG-Benchmark` experiment you can compare per-config runs as a table by
selecting visible params, metrics, and tags. Each benchmark sweep also logs an
aggregate summary run named like `summary_runN_<timestamp>` with:

- `tables/`: `results_summary.csv` and `results_per_sample.csv`
- `reports/`: JSON and Markdown reports
- `plots/`: generated PNG and interactive HTML plots
- `reproducibility/`: manifest and package freeze

Use the aggregate run when you want all tables and plots for one sweep in one
place. MLflow utilities live in `scripts/` — see the scripts themselves for
usage; back up `mlflow.db` before running destructive cleanup scripts.

## MLflow Tracing

Opt-in per-sample tracing (issue #42). Enable it in the manifest:

```yaml
settings:
  mlflow_tracing_enabled: true   # env fallback: MLFLOW_TRACING_ENABLED=true; default off
```

Every sample then produces **one trace** in the `RAG-Benchmark` experiment,
identical in shape for the internal pipeline and external HTTP/MCP/plugin
adapters:

```
rag_sample            CHAIN      inputs: question · outputs: response · attrs: config name, sample index, adapter, cached
├─ adapter.generate   CHAIN      client-side span around the adapter call (adapter.retrieve for benchmark_stage=retrieve)
│  ├─ vector_search   func       internal only; one per search round (multi-hop, hybrid dense leg)
│  ├─ rerank          RERANKER   internal only, when a reranker is configured
│  ├─ generate_answer LLM        internal / MCP context mode
│  └─ agent_model_call LLM       MCP agentic mode, one per model round
└─ retrieved_contexts RETRIEVER  exactly the contexts the generator saw, as MLflow Documents
```

- There is exactly one RETRIEVER span per trace. It is built from the
  adapter's final contexts (after multi-hop merging, hybrid fusion and
  reranking), so MLflow retrieval judges see the generator's evidence.
  Document `id` comes from `doc_id` / `chunk_id` / `id` metadata, falling back
  to `ctx_<n>` for bare string contexts.
- External adapters are black boxes: their spans are client-side only.
- Checkpoint-resumed samples are traced too (`benchmark.cached=true`).
- Traces are recorded while the worker's parent run is active and are then
  linked to the config's child run (`mlflow_tracing` run tag:
  `linked_<n>` or `link_failed`). Open a config run → *Traces* to see them.
- Outside the per-sample loop tracing is switched off, so ingestion and
  evaluation helpers (`chunk_documents`, `build_vector_store`, RAGAS) no
  longer create stray root traces — also when the setting is off, which now
  records no traces at all.
- Traces dominate `mlflow.db` growth; keep tracing off for large sweeps you
  do not intend to inspect.

### Ground-truth expectations on traces

After the per-sample loop, async trace logging is flushed and each trace gets
the sample's ground truth as MLflow **expectations** (issue #43), under
MLflow's reserved names so built-in judges pick them up:

| Expectation | Value | When |
|---|---|---|
| `expected_response` | reference answer (`ground_truth`, stripped) | always, unless blank |
| `expected_retrieved_context` | `[{"doc_uri": <gold doc id>, "content": <gold text>}]` | sample has `metadata.gold_doc_id` (per-sample datasets, deduplicated shared corpora) |
| `expected_retrieved_context` | `[{"doc_uri": <id>}, ...]` (ids only) | sample has `metadata.gold_doc_ids` (multi-hop; the context mixes in distractors) |

`doc_uri` uses the same ids as the RETRIEVER span's document `id`, so
retrieval can be compared by id. If a sample carries both keys,
`gold_doc_id` wins. `content` is the dataset's reference context: identical to
the indexed gold document for per-sample datasets and deduplicated corpora,
but for `jsonl-shared` it is whatever the manifest's context field holds and
may be only a passage of the corpus document — compare by `doc_uri` there.

Expectations are written only after the samples ran, so the system under test
never sees them — also when the run crashes mid-loop: traces persisted before
the crash still get their expectations. Their source is
`CODE` / `dataset:<dataset_name>`. A failed write logs a warning and never
fails the benchmark.

A run's traces can then be scored later without re-running the pipeline:

```python
import mlflow
traces = mlflow.search_traces(run_id="<config run id>", return_type="list")
for t in traces:
    expectations = {a.name: a.value for a in t.info.assessments if a.expectation}
```

or pass the DataFrame from `mlflow.search_traces(run_id=...)` to
`mlflow.genai.evaluate(data=..., scorers=[...])`.

Smoke manifest: `experiments/mlflow-tracing-smoke.yaml`.

## MLflow DB maintenance

`mlflow.db` grows with every run (traces dominate) and SQLite never shrinks the
file on its own. Stop any `mlflow server` using the DB first, then:

```bash
# compact backup (also a dry run of VACUUM)
python -c "import sqlite3; sqlite3.connect('mlflow.db').execute(\"VACUUM INTO 'mlflow.db.backup-$(date +%Y%m%d)'\")"
# hard-delete runs that were deleted in the UI / via delete_run (also removes their artifacts)
mlflow gc --backend-store-uri sqlite:///mlflow.db --tracking-uri sqlite:///mlflow.db
# reclaim free pages
python -c "import sqlite3; sqlite3.connect('mlflow.db').execute('VACUUM')"
```

Runs stuck in `RUNNING` after a crashed worker can be closed with
`MlflowClient().set_terminated(run_id, status="KILLED")`. Keep parent runs
that still have children — deleting them flattens the run hierarchy in the UI.
