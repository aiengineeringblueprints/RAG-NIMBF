# ADR-004: Deferred cleanup batch — grouping, evaluation merge, token stats, multi-hop, result model

**Status:** Deferred (not rejected)

## Context

Architecture gap analyses (doc/01-architecture-gaps.md,
doc/02-feature-improvements.md) surfaced five candidate refactors. They
were reviewed together because none is load-bearing for the adapter seam,
the YAML-first config, or the single worker loop (ADRs 001–003).

## Decision

All five are **deferred to later stages, not rejected**:

1. **Config grouping** — restructuring `BenchmarkConfig` into nested
   sub-models (generation/retrieval/evaluation groups). Deferred: the flat
   model is verbose but every knob currently has an env fallback, a test,
   and a YAML field; grouping is churn without a blocking pain.
2. **Evaluation merge** — consolidating RAGAS evaluation, custom metrics,
   and gold-retrieval metrics behind a single evaluator interface.
   Deferred: the three currently have different input shapes and execution
   costs; merging before the metrics set stabilizes would couple them
   prematurely.
3. **Token-stats unification** — one token-accounting path for generator
   and critic LLM usage. Deferred: critic counting already piggybacks on
   `_TokenCountingChatModel`; unification should follow, not precede, the
   evaluation merge.
4. **Multi-hop consolidation** — folding the multi-hop retrieval work (see
   `doc/multihop-retrieval.md`) and `benchmark/dynamic_groundtruth` into the standard adapter/stage flow.
   Deferred: the multi-hop pipeline is still experimental; consolidating
   now would freeze a moving design.
5. **Result-model builder** — a typed result model replacing the ad-hoc
   result dicts assembled in the runner. Deferred: schema is still changing
   with each report consumer; premature typing would add friction.

## Consequences

- Future architecture reviews should treat these as known, scheduled work,
  not as new discoveries to re-litigate from scratch.
- Order matters: evaluation merge should precede token-stats unification;
  result-model builder should follow the metrics schema settling.
- None of these blocks the decided seams in ADRs 001–003.
