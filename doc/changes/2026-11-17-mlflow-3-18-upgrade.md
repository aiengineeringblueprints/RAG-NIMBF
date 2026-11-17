# Upgrade MLflow to 3.18.0

Date: 2026-11-17

## Summary
MLflow is upgraded from the locked 3.15.1 to 3.18.0 (issue #41, parent #40).
3.18.0 fixes the DeepEval scorer returning another row's result under parallel
scoring, and the RAGAS scorers in `mlflow.genai` are no longer experimental.
Existing tracking (runs, params, metrics, artifacts) keeps working against the
existing `mlflow.db` backend after a one-off schema migration.

## Changes

### Dependencies
- `requirements.txt`: `mlflow>=2.0` → `mlflow>=3.18.0`.
- `requirements-lock.txt`: regenerated with
  `uv pip compile requirements.txt --python-version 3.12 --output-file requirements-lock.txt --upgrade-package mlflow --upgrade-package mlflow-skinny --upgrade-package mlflow-tracing`.
  `mlflow`, `mlflow-skinny`, `mlflow-tracing` are now `3.18.0`; all other
  pins were kept where still valid.
- The lock had drifted from `requirements.txt` since August, so the recompile
  also brings it back in line:
  - adds direct requirements that were missing: `apted`, `jiwer`, `lxml`
    (plus their transitive deps `rapidfuzz`, `dataclasses-json`,
    `marshmallow`, `typing-inspect`, `mypy-extensions`);
  - `langchain-community` 0.4.2 → 0.4.1 and `langchain-experimental`
    0.4.2 → 0.4.1, because `requirements.txt` caps
    `langchain-community>=0.4.1,<0.4.2` and the old lock violated that cap.

### Tests
- New `tests/test_dependencies.py` guards the requirements floor
  (`mlflow` spec must exclude 3.17.x and accept 3.18.0) and the locked
  versions of `mlflow`, `mlflow-skinny`, `mlflow-tracing` (≥ 3.18.0).

## Behaviour changes and migration
- **Schema migration required.** MLflow 3.18 refuses to open an `mlflow.db`
  with an older schema. Back it up and run `mlflow db upgrade` once per
  database; steps in [`../results-and-mlflow.md`](../results-and-mlflow.md#mlflow-run-comparison).
  The migration is one-way: MLflow < 3.18 clients/UIs cannot open the migrated
  DB. Restore the backup to roll back.
- Existing venvs: `pip install "mlflow==3.18.0" "mlflow-skinny==3.18.0" "mlflow-tracing==3.18.0"`
  upgrades just the MLflow family without reinstalling the rest of the lock
  (that is how the Spark venv was upgraded).
- Code using MLflow (`benchmark/tracking.py`, `mlflow.genai.evaluate`,
  `mlflow.evaluate`) needed no changes.

## Verification
All runs on the DGX Spark (`~/source/Benchmarking-Framework`, branch
`optimaise-connectivity`). The Spark venv had drifted from the old lock
(MLflow 3.13.0 instead of 3.15.1) and went 3.13.0 → 3.18.0; its other packages
already matched the new lock (`langchain-community` 0.4.1,
`langchain-experimental` 0.4.1, `apted` 1.0.3, `jiwer` 4.0.0, `lxml` 6.1.3),
so the runs below exercised the resynced lock. `pip check` clean.

- `mlflow.db` (3.7 GB) backed up to `mlflow.db.backup-20261117-pre-mlflow318`;
  `mlflow db upgrade` moved alembic revision `da6fb0208061` → `f4a9b2c7d1e6`
  in 1.7 s.
- `tests/test_dependencies.py`: fails against the old manifests (3.15.1 lock,
  `>=2.0` floor), passes after the change.
- Full suite: 830 passed, 1 failed. The failure
  (`tests/test_config.py::TestGetAllCombinations::test_hybrid_defaults`) is
  pre-existing and unrelated: the Spark `.env` sets `RETRIEVAL_STRATEGY=mmr`,
  which leaks into the default-config test. With
  `RETRIEVAL_STRATEGY=similarity`, `tests/test_config.py` passes (125/125).
- Smoke run (internal adapter, `squad`, 3 samples, `ollama:llama3.2:1b`,
  `nomic-embed-text:latest`, recursive 500/50, k=3,
  `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false`, run dir
  `results/mlflow318-smoke`): finished in 13.3 s and logged 3 new runs
  (406 → 409). The config run has 9 params, 29 metrics, `data/` and
  `reproducibility/` artifacts, tag `mlflow_classic_retriever_metrics=logged`;
  the summary run has `reports/` and `reproducibility/`.
- Existing runs stay readable after the migration (e.g. a 2026-06-02 run with
  160 metrics and `eval_results_table.json`, a 2026-11-15 run with 184
  metrics), and `mlflow server --backend-store-uri sqlite:///mlflow.db`
  serves the UI (HTTP 200) and returns runs through the REST API.

## Follow-ups
- The laptop's `mlflow.db` (and any other copy) needs the same backup +
  `mlflow db upgrade` before MLflow 3.18 can open it.
- `test_hybrid_defaults` is sensitive to the local `.env`; isolate config
  tests from `.env` in a separate change.
- Evaluate switching the RAGAS/DeepEval scorers to `mlflow.genai` now that
  they are stable (see #40, `doc/13-ragas-vs-mlflow-genai-judges.md` on
  branch `research/40-ragas-vs-mlflow-judges`).
