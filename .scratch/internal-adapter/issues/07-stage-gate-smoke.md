# 07: Stage gate — full test-suite verification (no benchmark runs)

**What to build:** Proof that the refactor preserves behavior, via the automated test suite only — no real benchmark runs (agents never execute paid/slow benchmark runs; the user runs those themselves): worker-level tests with a fake tracking backend and temp run dirs covering the internal adapter and a stubbed external adapter through component injection, asserting structurally identical artifacts and distinct vector-store cache keys per dataset.

**Blocked by:** 04 (Worker becomes the only orchestration loop), 05 (Component-injection policy moves into the adapter infrastructure).

**Status:** ready-for-agent

- [ ] Full pytest passes with RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false
- [ ] Worker tests: internal adapter, stub dataset → run dir, tracking records, report and per-sample artifacts present and structurally unchanged (fake tracker, temp dirs)
- [ ] Worker tests: stubbed external adapter through component injection → identical structure; distinct cache keys per dataset asserted
- [ ] No test downloads models, calls paid APIs, or launches MLflow for real; any disposable artifacts land in temp dirs, not the repo
