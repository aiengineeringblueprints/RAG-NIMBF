# 03: Orchestrator drives every system through the adapter seam

**What to build:** The run path stops special-casing the internal pipeline. The branch on adapter-absence inside the per-sample loop is removed: every system, internal or external, crosses the same adapter interface. Adapter-specific diagnostic aggregation (e.g. MCP connection/tool/generation timings read from stringly-typed dictionary keys) moves behind the adapter interface — the orchestrator consumes a uniform, adapter-agnostic summary. Tests for component injection run against the adapter interface and no longer import or patch the entry-point script.

**Blocked by:** 02 (Internal pipeline becomes a registered managed adapter).

**Status:** done

- [x] No `adapter is None` branching remains in the benchmark run path
- [x] The orchestrator reads no adapter-internal diagnostic keys; MCP timings still appear in stage timings and reports
- [x] Component-injection end-to-end tests drive the adapter seam directly (no entry-point imports)
- [x] Full pytest passes; one internal run and one stubbed external run produce structurally identical results
