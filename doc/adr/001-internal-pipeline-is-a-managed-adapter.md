# ADR-001: The internal pipeline is a managed adapter

**Status:** Accepted

## Context

The framework benchmarks both external RAG systems (HTTP, MCP, Python
plugins) and its own built-in pipeline (chunking → embedding → vector
store → retrieval → reranking → generation). Originally the internal
pipeline had a bespoke code path in the runner, external systems went
through adapters, and the two lifecycles diverged: different prepare/run
flow, different result shapes, different caching, and per-comparison
fairness hacks.

## Decision

The internal pipeline is registered as a **managed adapter** through the
same `register_rag_adapter` seam as every external system (issue #2). There
is exactly one adapter lifecycle for all systems. "Managed" means the
Framework builds the pipeline components and offers them to the adapter via
the immutable `ComponentBundle` (injection policy declared per adapter via
`RAG_ADAPTER_ACCEPTS`); the adapter still owns its own run.

## Consequences

- The runner contains no special-case branch for the internal pipeline.
- New systems integrate only through the adapter protocol; no custom logic
  in `main.py` or the runner.
- Internal-only knobs (e.g. `BENCHMARK_STAGE=index`) exist because the
  Framework manages that adapter's components; they do not apply to
  black-box adapters.
- Fairness validation for internal-vs-external comparisons rests on the
  shared bundle/registry, not on duplicated wiring.
- Future architecture reviews must not re-propose a separate seam for the
  internal pipeline or a second lifecycle.
