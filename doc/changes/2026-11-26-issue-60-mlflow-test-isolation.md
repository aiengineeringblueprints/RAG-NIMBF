# Test suite isolated from the machine-local MLflow database (issue #60)

Date: 2026-11-26

## Summary
Completes the test-isolation work for issue #60. The `.env`/shell isolation
from #60 was already on `dev` (commits `9d5c299b`, `c8f7a175`), but one of the
five issue-listed tests still failed on machines whose repo-root `mlflow.db`
is in a broken schema state: without a reachable MLflow server,
`benchmark/tracking.py` and `benchmark/tracing.py` fall back to
`sqlite:///mlflow.db` in the repo root — machine-local state a test suite must
never depend on.

## Changes
- `tests/conftest.py`: the autouse `isolated_benchmark_env` fixture now pins
  `MLFLOW_TRACKING_URI` to a throwaway per-test SQLite DB
  (`<tmp>/mlflow-*/mlflow-test.db`), so every MLflow touch lands in an
  isolated database.
- `benchmark/tracking.py` / `benchmark/tracing.py`: the "MLflow server
  unreachable → fall back to `sqlite:///mlflow.db`" path no longer overrides
  an explicitly configured SQLite tracking URI.
- `tests/test_env_isolation.py`: regression test asserting the pinned URI is
  an isolated `mlflow-test.db` and not the repo-root one.

## Behaviour changes and migration
None for normal runs. If you previously relied on setting
`MLFLOW_TRACKING_URI` to a SQLite URI, it is now honoured verbatim instead of
being replaced by `sqlite:///mlflow.db` when no MLflow server answers.

Note: the local `mlflow.db` is in an inconsistent state (alembic version
`c3d6457b6d8a` while index `index_metrics_run_uuid_key_step` already exists),
which makes any MLflow store initialisation against it fail with
`sqlite3.OperationalError: index index_metrics_run_uuid_key_step already
exists`. Back up `mlflow.db` and repair it separately; the test suite no
longer depends on that repair.

## Verification
- `tests/test_corpus_parser_ingestion.py::test_worker_pipeline_runs_parsed_corpus_unchanged`
  failed before the change, passes after.
- Full suite: 1178 passed.

## Follow-ups
- Repair/upgrade the repo-root `mlflow.db` schema (back up first; see
  `doc/results-and-mlflow.md`).
