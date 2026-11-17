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
