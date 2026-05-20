# Architecture Gaps

## 1. Missing Abstract Interfaces

No abstract base classes for core components. Direct implementation coupling makes swapping difficult.

**Affected modules:**
- `benchmark/retrieval.py` — no `VectorStore` interface; ChromaDB and LanceDB logic inline
- `benchmark/evaluation.py` — no `Evaluator` interface; RAGAS-specific code baked in
- `benchmark/generation.py` — no `Generator` interface; provider-specific logic in factory
- `benchmark/custom_metrics.py` — no `Metric` interface; all metrics in single file

**Fix:** Define ABCs (`BaseVectorStore`, `BaseEvaluator`, `BaseMetric`) with pluggable implementations.

## 2. Tight Coupling in main.py

`main.py:15-21` imports directly from all benchmark submodules. `run_single_benchmark()` is a 450-line monolith that orchestrates chunking, retrieval, generation, evaluation, and reporting in a single function.

**Fix:** Extract into a pipeline pattern with discrete stages:

```python
class PipelineStage(ABC):
    def run(self, context: PipelineContext) -> PipelineContext: ...

stages = [ChunkStage(), IndexStage(), RetrieveStage(), GenerateStage(), EvaluateStage(), ReportStage()]
```

## 3. Sequential Execution Only

`main.py:513` — runs are sequential by design (single GPU assumption). No support for:
- Multi-GPU parallel configs
- Remote/distributed benchmark execution
- Async I/O for network-bound operations (API calls, embedding)

**Fix:** Add optional `ParallelExecutor` with configurable worker count. Network-bound steps (generation, embedding) can overlap with local steps.

## 4. Configuration Scattered Across ENV Vars

`config.py` has 40+ env vars. No validation layer beyond basic type checks. Missing:
- Config file support (YAML/TOML)
- Config profiles (dev/staging/prod)
- Config diffing between runs

**Fix:** Add YAML config layer that env vars override. Support `--config benchmark.yaml` CLI arg.

## 5. Vector Store Cache Unbounded

`benchmark/retrieval.py` — ChromaDB collections grow indefinitely. No eviction, no TTL, no size limits. `.chroma/` directory currently has 50+ UUID collections.

**Fix:** Add cache eviction strategy (LRU by access time, or explicit `--clean-cache` command). Track collection metadata (creation date, last used, size).

## 6. Agentic Runner Duplicates Main Pipeline

`agentic/benchmark_runner.py` reimplements `run_single_benchmark()` with subtle differences:
- Missing shared-corpus loading
- Missing full reporting/tracing
- Missing custom metrics computation

**Fix:** Agentic runner should call `main.run_single_benchmark()` directly, not reimplement.

## 7. No Plugin/Extension System

Adding a new metric, dataset, or provider requires modifying core modules. No registration mechanism.

**Fix:** Add entry-point based plugin registry:

```python
# In setup.py or pyproject.toml
[project.entry-points."benchmark.metrics"]
bert_score = "benchmark.custom_metrics:BERTScoreMetric"
```

## 8. Result Format Not Versioned

JSON results in `results/runN/configs/*.json` have no schema version. Breaking changes to `BenchmarkResultExtended` silently invalidate old results.

**Fix:** Add `schema_version` field to result JSON. Add migration utilities for old schemas.

## 9. Runtime Artifacts Mixed with Source

`results/`, `mlruns/`, `.chroma/`, `.lancedb/` all live in repo root alongside source code. Risk of accidental commits, confusion in git status.

**Fix:** Move all runtime artifacts to a single `.bench/` directory (or `~/.cache/benchmark-framework/`). Add to `.gitignore`.

## 10. No Graceful Degradation

If MLflow is down, the entire benchmark fails. If RAGAS evaluation crashes mid-run, partial results are lost except for the per-config JSON writes.

**Fix:** Make tracking and evaluation optional/async. Continue benchmark even if non-critical services fail. Add resume-from-failure support.
