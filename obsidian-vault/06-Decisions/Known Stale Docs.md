# Known Stale Docs

Some historical notes under `Optimizations/`, `info.md`, and paper drafts describe older project states. Verify against source and generated results before treating them as current.

Known stale or uncertain claims:

- `Optimizations/full-project-audit-report.md` reportedly says only 13 of 128 configs were tested; `EVAL_MATRIX.md` now appears to show many more tested rows.
- Older audit notes say custom metrics are not wired in; current `main.py` imports and calls `compute_custom_metrics()`.
- Older audit notes say RAGAS is hardcoded to faithfulness only; current evaluation code and result models are broader, but the exact active metric set should be checked in `benchmark/evaluation.py`.
- Paper results appear incomplete or placeholder-like and should not be used as final claims until reconciled with `results/`, MLflow, and `EVAL_MATRIX.md`.
- Some notes discuss FinQA/t2-ragbench failure modes while the current matrix is largely `squad`; keep dataset-specific conclusions separate.
- `info.md` is operational scratch material rather than durable design documentation.

Maintenance rule:

- When a stale claim is resolved, update or remove the stale source note and update this page.

Related notes:

- [[Research Notes]]
- [[Evaluation and Metrics]]
- [[Known Risks and Gaps]]
