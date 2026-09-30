# Spec: Internal pipeline as a managed adapter; ExperimentWorker as the only orchestration loop

Label: `ready-for-agent`

## Problem Statement

Running a benchmark against the framework's own built-in RAG pipeline is architecturally different from running one against an external system. The adapter registry knows four external adapters (http, mcp, ragflow, optimaiserag), but asking for the internal one returns nothing, and the orchestrator special-cases that nothing by hand-wiring the entire pipeline — chunking, embedding, indexing, retrieval, reranking, generation — inline in a ~565-line function. Adapter-internal diagnostics (e.g. MCP stage timings) leak into that function through stringly-typed dictionary keys. The same orchestration loop (dataset load → run dir → reproducibility bundle → tracking parent run → per-config run → report → artifacts) is written twice — once in the script entry point, once in the worker — and the copies have already drifted on a corpus-path parameter. Separately, the component-injection path computes a weaker index cache key than the internal path, so two experiments over different datasets can silently share a vector-store collection. Tests for component injection can only run by importing the entry-point script and patching its module globals. Every change to how runs execute means editing a god-script that four other modules import from.

## Solution

The internal pipeline becomes a first-class managed adapter. Asking the registry for "internal" returns a real adapter implementing the same lifecycle interface as every external system: capabilities, prepare (build/lookup the index), retrieve, generate, cleanup. The orchestrator stops branching on adapter absence — every run, internal or external, crosses the same seam. Adapter-specific diagnostics are aggregated behind the adapter interface, not by the orchestrator reading the adapter's dictionaries.

The orchestration loop exists exactly once, in the worker. The script entry point becomes a thin CLI: it requires an experiment manifest (YAML-first); without one it fails with a clear pointer instead of silently falling back to the legacy environment-variable matrix, which is deleted. Component injection's cache key and the internal path's cache key become one shared function.

## User Stories

1. As a benchmark author, I want the internal pipeline to be registered as an adapter named "internal", so that experiment manifests treat it exactly like any external system.
2. As a benchmark author, I want every adapter (internal included) to expose the same managed lifecycle, so that I can run stage-level experiments (index-only, retrieve-only) uniformly across all systems.
3. As a benchmark author, I want the orchestrator to be agnostic of which adapter it is driving, so that adding a new system requires zero orchestrator changes.
4. As a framework maintainer, I want adapter-specific diagnostics aggregated behind the adapter interface, so that orchestrator code never reads adapter-internal dictionary keys.
5. As a framework maintainer, I want exactly one orchestration loop, so that a fix to run orchestration (reproducibility bundles, tracking hierarchy, resource monitoring) applies to every entry point at once.
6. As a framework maintainer, I want the corpus-path handling to exist in one place, so that the two loops can never disagree about it again.
7. As a researcher reproducing experiments, I want `python main.py` with a manifest to delegate to the worker, so that CLI-documented invocations keep working.
8. As a researcher reproducing experiments, I want an explicit error telling me to set the manifest variable when I run without one, so that I am never surprised by a silent legacy-matrix fallback.
9. As a researcher comparing systems, I want the internal pipeline's index cache key to include dataset name, dataset subset, sample size, corpus fingerprint, and embedding provider, so that two experiments can never silently share a vector-store collection.
10. As a benchmark author using component injection, I want the injected path to use the same cache key function as the internal path, so that cache behavior is identical regardless of how components are provided.
11. As a plugin author, I want the component-injection policy (deciding whether and what to inject) to live with the adapter infrastructure, so that the orchestrator stays thin and the policy is testable at its own seam.
12. As a framework maintainer, I want tests for component injection to exercise the adapter interface directly, so that they no longer need to import and patch the entry-point script.
13. As a framework maintainer, I want the orchestration loop tested through the worker's interface with a fake tracker and temporary directories, so that orchestration regressions are caught without paid LLM calls.
14. As an AI agent working in this codebase, I want the entry-point script to be thin, so that understanding how a run executes requires reading one deep module instead of a 1,200-line script.
15. As a benchmark author, I want existing experiment manifests and .env files to keep working unchanged, so that the refactor never breaks a recorded experiment.
16. As a framework maintainer, I want the legacy environment-matrix code path deleted, so that there is one configuration language (YAML-first) and no duplicated validation.

## Implementation Decisions

- **Internal adapter**: registered under the name "internal" in the existing adapter registry. It implements the full managed lifecycle interface (capabilities, prepare, retrieve, generate, cleanup) so all five adapters are lifecycle-uniform. Stage-level benchmarking works uniformly through this interface.
- **Diagnostic aggregation**: the per-adapter diagnostics contract moves behind the adapter interface. The orchestrator consumes a uniform, adapter-agnostic timing/metrics summary; MCP-specific stage names are the MCP adapter's private business.
- **One orchestration loop**: the worker owns the loop. The script entry point parses the manifest requirement, resolves the run directory, and delegates to the worker. The legacy loop and the legacy .env-matrix entry are deleted; the `main.py`-without-manifest case fails with an actionable message.
- **CLI surface unchanged**: `python main.py` (manifest required) and `python -m benchmark.worker plan/run` both keep working, both delegating to the worker.
- **One cache key**: a single shared index-cache-key function, including dataset name, subset, sample size, corpus fingerprint, and embedding provider, used by both the internal path and the component-injection path.
- **Injection policy relocated**: the decide-what-to-inject logic moves from the orchestrator into the adapter infrastructure beside the component factory. The frozen-config mutation fallback is removed as part of the move.
- **Config compatibility is absolute**: all existing flat YAML manifest keys and .env variables keep working unchanged. Internal grouping of the configuration object is deferred to the next stage and is out of scope here.
- **Evaluation, metrics merge, token-stats unification, multi-hop consolidation, result-model builder**: deferred to later stages (see Out of Scope).
- **Paper trail**: a domain glossary (CONTEXT.md) is seeded, and ADRs are recorded for: internal-is-a-managed-adapter; YAML-first-only with legacy matrix removal; worker-as-only-loop.

## Testing Decisions

- Tests exercise external behavior through the three seams only: the adapter interface, the worker's run interface, and the entry point's manifest requirement. No test patches entry-point module globals.
- The internal adapter is tested like the existing managed-adapter lifecycle tests: prepare against a tiny in-memory corpus, assert capabilities, retrieve, generate with a stubbed generator, cleanup.
- The worker is tested with a fake tracking backend and temp run dirs: one-cell manifest in, per-config artifacts out, resume behavior via the existing checkpoint/progress stores.
- The cache-key function is tested as a pure function: distinct datasets / fingerprints / embedding providers produce distinct keys; identical inputs produce identical keys.
- Prior art: the existing managed-adapter lifecycle tests, adapter-protocol tests, and orchestration matrix tests define the house style.
- The stage is verified by the full test suite (RAGAS and custom metrics disabled) with the worker driven by a fake tracking backend and temp run dirs. No real benchmark runs are executed by agents; the user validates with real experiments themselves.

## Out of Scope

- Grouping the configuration object into subsystem settings (stage 3) — but nothing here may make that harder.
- Merging evaluation modules and deduplicating IR metrics (stage 4).
- Token-stats value object unification; multi-hop concept consolidation; result-model builder (deferred).
- Any change to results/ directory formats, MLflow hierarchy semantics, or manifest YAML keys.
- The MLP fairness-validation logic beyond what the diagnostic-aggregation move touches.

## Further Notes

- The `BENCHMARK_STAGE=index` behavior is internal-adapter-specific today; it becomes uniformly expressible through the lifecycle interface, but no new stage values are added in this stage.
- Results directories remain append-only; the refactor must not require re-running past experiments to keep them interpretable.
- The report visualisation that motivated this work is archived at /tmp/architecture-review-20260914-083819.html (ephemeral).
