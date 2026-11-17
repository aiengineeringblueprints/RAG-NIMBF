# Trace expectations: PR #52 review fixes

Date: 2026-11-17

## Summary
Follow-up to `2026-11-17-trace-ground-truth-expectations.md` (issue #43),
addressing the review on PR #52: expectations now survive a mid-loop crash,
blank reference answers are skipped, and the gold-key precedence and the
meaning of gold `content` are documented.

## Changes
- **Crash-safe expectations** (`benchmark/tracing.py`,
  `benchmark/orchestration/runner.py`): new context manager
  `expectations_logged_on_exit(source_id=...)` collects `(trace_id, sample)`
  pairs and calls `log_sample_expectations` in a `finally`. The runner's
  per-sample loop runs inside it, so traces persisted before an adapter
  exception no longer silently lack their ground truth.
- **Blank reference answers** (`sample_expectations`): `ground_truth` is
  stripped; whitespace-only answers produce no `expected_response`.
- **Documented contracts** (`sample_expectations` docstring,
  `doc/results-and-mlflow.md`): `gold_doc_id` wins over `gold_doc_ids`;
  `content` equals the indexed gold document for per-sample datasets and
  deduplicated corpora (both index the sample context joined with `"\n"`, see
  `benchmark/chunking.py`), but for `jsonl-shared` it is the manifest's context
  field and may be only a passage — compare by `doc_uri` there.

## Behaviour changes and migration
None. Crashed tracing-enabled runs now carry expectations on the traces
written before the crash.

## Verification
- `tests/test_mlflow_tracing.py`: new
  `test_traces_written_before_a_crash_still_get_expectations` (red before the
  fix: `KeyError: 'expected_response'`); the gold-context test gained a
  whitespace-only `ground_truth` sample (red before the fix).
- DGX Spark: `tests/test_mlflow_tracing.py` 13 passed; full suite 965 passed,
  3 failed — the same 3 pre-existing, `.env`-dependent failures as on `dev`.

## Follow-ups
None from this review.
