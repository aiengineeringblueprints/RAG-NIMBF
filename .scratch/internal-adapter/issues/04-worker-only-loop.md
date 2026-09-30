# 04: Worker becomes the only orchestration loop; entry point a thin CLI

**What to build:** Exactly one orchestration loop exists, in the worker. The duplicate loop in the entry-point script — including the legacy .env-matrix entry — is deleted, taking the drifted corpus-path copy with it. `python main.py` requires an experiment manifest and delegates to the worker; without a manifest it fails with a clear pointer ("set BENCHMARK_CONFIG_FILE=experiments/<name>.yaml"). `python -m benchmark.worker plan/run` is unchanged. The worker loop is tested through its own interface with a fake tracking backend and temp run dirs, including resume via the existing checkpoint/progress stores.

**Blocked by:** 03 (Orchestrator drives every system through the adapter seam).

**Status:** ready-for-agent

- [ ] The legacy loop, the .env-matrix entry, and the duplicated run-dir/corpus-path helpers are deleted; the worker's versions are the only ones
- [ ] `python main.py` without a manifest exits with an actionable YAML-pointer error (tested)
- [ ] `python main.py` with a manifest and `python -m benchmark.worker run` both execute the worker loop
- [ ] Worker tests: one-cell manifest in → per-config artifacts, tracking parent run, report out; resume behavior covered
- [ ] README/AGENTS.md command documentation still accurate; env-var table updated for removed legacy variables
