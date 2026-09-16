# ADR-003: The worker is the only orchestration loop

**Status:** Accepted

## Context

Orchestration logic (matrix expansion, resumable runs, per-cell progress,
reporting hooks) existed in more than one place: `main.py` had its own flow
and the worker in `benchmark/orchestration/worker.py` had another. Fixes to
resumption, parallelism, or result handling had to be applied twice, and
behavior drifted between entry points.

## Decision

`benchmark/orchestration/worker.py` (exposed as `python -m benchmark.worker`)
is the **only** orchestration loop. It is resumable and supports
`--keep-going` and `--run-dir`. `main.py` is a thin CLI that validates an
experiment manifest and delegates everything to the worker. Single-cell
execution lives in `benchmark/orchestration/runner.py`
(`run_single_benchmark`) but contains no loop or resumption logic.

## Consequences

- One place to change run semantics; no drift between entry points.
- `main.py` stays intentionally dumb: manifest → worker.
- Progress stores, checkpoint stores, and MLflow/tracking hooks are wired
  in exactly one place.
- Future architecture reviews must not re-propose a second orchestration
  loop or a "fat" `main.py`.
