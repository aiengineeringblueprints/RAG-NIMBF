# CONTEXT.md — Domain glossary

The vocabulary this codebase actually uses. Future architecture work should
read this file plus `doc/adr/` before proposing changes; several seams have
already been decided (see ADRs 001–004).

## Glossary

### Adapter
The single integration seam for every RAG system under test. Adapters are
registered with `register_rag_adapter` in `benchmark/adapters/` and implement
one uniform lifecycle (prepare → run → teardown). Two flavors:

- **Black-box adapter** — the Framework treats the system as opaque (HTTP,
  MCP, Python-plugin adapters). It sends questions and receives answers plus
  retrieved contexts; it never touches the system's internals.
- **Managed adapter** — the Framework builds and owns the pipeline's
  components and drives the lifecycle itself. The internal pipeline is a
  managed adapter (ADR-001).

### ComponentBundle
`benchmark/adapters/components.py::ComponentBundle` — an immutable bag of
optional LangChain-compatible components the Framework builds from
configuration and offers to an adapter. Slots:

- `chunker` (LangChain `TextSplitter`)
- `embedder` (LangChain `Embeddings`)
- `retriever_factory` (callable `list[dict] -> retriever`)
- `reranker`
- `llm` (LangChain `BaseChatModel`)
- `prompt_template`

All slots default to `None`. `None` means "the Framework did not build this
slot" or "the adapter declined injection via `supports_components`".
Adapters receive the bundle, store it, read it — they never mutate the
Framework-built instance (it is frozen so it can be hashed/cached).
Injection policy is declared per adapter via `RAG_ADAPTER_ACCEPTS`.

### Benchmark stages
The phases of a single benchmark run (`benchmark/orchestration/runner.py`):
ingest/chunk/embed/index → retrieve → rerank → generate → evaluate. The
`BENCHMARK_STAGE` knob (currently internal-adapter only) cuts the run short
after a given stage (e.g. `index` to build and cache a corpus index).

### Run / run_dir
A **run** is one execution of a worker loop over an expanded config matrix.
Its **run_dir** (e.g. `results/runN`) is the append-only record of outputs:
per-question checkpoints, benchmark JSONs, reports, plots. Run dirs are
never edited by hand and are reused via `worker --run-dir` for resumption.

### BenchmarkSample
`benchmark/dataset.py::BenchmarkSample` — the normalized per-question unit
of a dataset: question, ground-truth answer, gold contexts/ids. Datasets
(HuggingFace or local JSONL/CSV) are loaded and normalized into samples by
`dataset_adapters.py` before entering the pipeline.

### Critic LLM
The judge model used during evaluation (`evaluation.py`, `custom_metrics.py`,
RAGAS metrics). Configured separately from the generator via
`EVAL_CRITIC_LLM` / `EVAL_CRITIC_EMBEDDING` and their own base-URL/key
settings in `config.py`. Falls back to the generator model when unset.
Slow and costly — disable via `RAGAS_ENABLED=false` for smoke tests.

### Checkpoint store / progress store
- **Checkpoint store** — `benchmark/checkpoint.py`: append-style JSON
  checkpoint keyed by question index; makes a run resumable per question.
- **Progress store** — the worker's resumable plan/state that tracks which
  matrix cells of a run are already complete (`python -m benchmark.worker
  plan` / `run` with `--run-dir`).

### Matrix / manifest
- **Manifest** — an `experiments/*.yaml` (or JSON) file declaring an
  experiment. The required entry point; `main.py` refuses to run without one.
- **Matrix** — the expansion of a manifest into a list of concrete
  `BenchmarkConfig` instances (`benchmark/orchestration/matrix.py`), one per
  combination of dataset/model/parameter values. The worker executes the
  matrix cell by cell, resumably.

### Parser adapter
A second, separate adapter family for document parsers (OCR / PDF→Markdown
systems), registered with `register_parser_adapter`. Contract: a document
(PDF or page images) in → a `ParseResult` out. Two flavors exist: HTTP
(OpenAI-compatible endpoints — the converged serving standard) and in-process
Python plugins (Docling, Marker, MinerU). Distinct from the RAG adapter
family (ADR-005); the two share serving patterns, not lifecycles.

### ParseResult
The output contract of a Parser adapter: one Markdown document per page.
Structured blocks (bbox, category, table HTML, formula LaTeX) are optional;
parsers that emit only Markdown still score on text and table metrics
(markdown tables are converted to HTML for TEDS, as OmniDocBench does).

### Parsing metrics
The metric family scoring a ParseResult against document ground truth:
CER/WER and normalized edit distance for text, TEDS for tables, computed
after GT↔prediction alignment (vendored OmniDocBench `quick_match`
semantics). The matching algorithm version is pinned in run metadata because
scores are not comparable across alignment versions. Layout mAP, reading
order, CDM (formulas), and charts are out of scope for v1.

### Corpus parser
The `corpus_parser` config knob that routes corpus construction of a RAG
experiment through a Parser adapter instead of direct text loading.
Chunking, retrieval, and evaluation run unchanged; parser identity and
version are pinned in run metadata. This is the seam that makes parsing-metrics
↔ RAG-outcomes correlation experiments possible in one provenance chain.

## Decided seams (do not re-propose)

| Decision | ADR |
| --- | --- |
| Internal pipeline is a managed adapter; one lifecycle for all systems | ADR-001 |
| YAML-first configuration only; legacy `.env` matrix removed; manifest required at the CLI | ADR-002 |
| The worker is the only orchestration loop | ADR-003 |
| Deferred batch: config grouping, evaluation merge, token-stats unification, multi-hop consolidation, result-model builder | ADR-004 |
| Parser adapters are a separate family from RAG adapters; parsing evaluation is its own module group | ADR-005 |
