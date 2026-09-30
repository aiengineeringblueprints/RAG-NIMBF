# 06: Paper trail — CONTEXT.md glossary and ADRs

**What to build:** The domain glossary (CONTEXT.md) is seeded with the vocabulary the codebase actually uses: adapter (black-box vs managed), ComponentBundle and its slots, benchmark stages, run/run_dir, Sample, critic LLM, checkpoint/progress stores, matrix/manifest. ADRs record the load-bearing decisions: (1) internal pipeline is a managed adapter — one lifecycle for all systems; (2) YAML-first only — legacy .env matrix removed, manifest required at the CLI; (3) the worker is the only orchestration loop; (4) rejected/deferred: config grouping, evaluation merge, token-stats unification, multi-hop consolidation, result-model builder — deferred to later stages, not rejected.

**Blocked by:** None (can start immediately; ADR content is decided, not discovered).

**Status:** ready-for-agent

- [ ] CONTEXT.md exists with glossary entries matching the terms above
- [ ] ADRs exist for the three accepted decisions and one deferred-batch record
- [ ] Future architecture reviews reading CONTEXT.md + ADRs would not re-propose the internal-pipeline seam or the second orchestration loop
