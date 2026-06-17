# Component Injection for External RAG Adapters

**Date:** 2026-06-17
**Status:** Design
**Author:** Brainstorming session

## Problem

Framework already supports three RAG system modes via `RAG_SYSTEM_ADAPTER`:

- `internal` — full Framework pipeline, cartesian sweep over components
- `http` — black-box HTTP RAG endpoint
- `<plugin>` — Python-importable black-box adapter (`prepare()` + `answer()`)

None of these lets the user specify Framework components (chunker, embedder,
LLM, prompt, etc.) in `.env` and have an **external** RAG system consume them.
Either the Framework owns the whole pipeline (`internal`) or the external RAG
owns everything (`http`/plugin). Mixed ownership — e.g. Framework-supplied
LLM + external retriever — is not supported.

## Goal

User declares in `.env`:

1. Which external RAG system to benchmark (`RAG_SYSTEM_ADAPTER=enterprise_rag`)
2. Which components the Framework should build from `.env` and hand to the
   external RAG (`CHUNKING_STRATEGIES`, `EMBEDDING_MODELS`, `LLM_MODELS`, …)
3. Which component slots the external RAG is willing to accept
   (`RAG_ADAPTER_ACCEPTS=chunker,embedder,llm,prompt`)

Framework builds the requested components, hands them to the adapter, runs
the adapter's `prepare()` + per-sample `answer()`, then evaluates with the
existing RAGAS / custom-metric stack.

## Non-Goals

- Full bergen-style rewrite of the component registry. Existing
  `benchmark/chunking.py`, `benchmark/embedding.py`, `benchmark/providers.py`,
  `benchmark/retrieval.py`, `benchmark/reranker.py` are reused as-is.
- Component-level matrix sweep across an external RAG. The cartesian product
  over chunkers/embedders/LLMs still runs inside the Framework's `internal`
  pipeline. For external adapters, one `ComponentBundle` is constructed per
  benchmark run (single config row) and injected. Matrix sweeping external
  adapters is a follow-up, not in this spec.
- Supporting non-LangChain components. `ComponentBundle` slots use LangChain
  types (`BaseChatModel`, `Embeddings`, `TextSplitter`). External RAG systems
  using different runtimes must wrap them.

## Architecture

```
.env  ──> BenchmarkConfig  ──> build_components(config) ──> ComponentBundle
                                                                     │
                                                                     v
                                    RagSystemAdapter.set_components(bundle)
                                                                     │
                                                                     v
                                                 Adapter.prepare() + answer()
                                                                     │
                                                                     v
                                                 RagSystemOutput (answer, contexts,
                                                                  metadata, latency, …)
                                                                     │
                                                                     v
                                              Existing RAGAS + custom metric eval
```

The new pieces sit between config and the existing adapter `prepare()` call.
They are **opt-in**: adapters that do not implement `set_components()` keep
working unchanged.

## Components

### 1. `ComponentBundle` dataclass

New file: `benchmark/adapters/components.py`

```python
from dataclasses import dataclass
from typing import Any, Callable, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import TextSplitter


@dataclass
class ComponentBundle:
    """Framework-built components offered to an external RAG adapter.

    All fields are optional. Adapters pick what they support; the rest stays
    `None`. `None` means "Framework did not build this" or "adapter does not
    want it" (see `supports_components`).
    """
    chunker: Optional[TextSplitter] = None
    embedder: Optional[Embeddings] = None
    retriever_factory: Optional[Callable[[list[dict]], Any]] = None
    reranker: Optional[Any] = None
    llm: Optional[BaseChatModel] = None
    prompt_template: Optional[str] = None
```

`retriever_factory` is a callable taking the corpus records and returning a
LangChain `VectorStore` or `BaseRetriever`. This avoids building the retriever
before `prepare()` (which needs the corpus) and lets the adapter decide when
and how to instantiate it.

### 2. `build_components(config)` factory

New function in `benchmark/adapters/components.py`.

```python
def build_components(config) -> ComponentBundle:
    """Build a ComponentBundle from a BenchmarkConfig row, respecting
    `RAG_ADAPTER_ACCEPTS` capability flags."""
    accepts = _parse_accepts(getattr(config, "rag_adapter_accepts", ""))
    bundle = ComponentBundle()
    if "chunker" in accepts and config.chunking_strategy and config.chunk_size:
        bundle.chunker = build_chunker(config)
    if "embedder" in accepts and config.embedding_model:
        bundle.embedder = build_embedder(config)
    if "llm" in accepts and config.llm_model:
        bundle.llm = build_llm(config)
    if "reranker" in accepts and config.reranker_model:
        bundle.reranker = build_reranker(config)
    if "prompt" in accepts:
        bundle.prompt_template = resolve_prompt(config)
    if "retriever" in accepts and config.embedding_model:
        bundle.retriever_factory = _make_retriever_factory(config)
    return bundle
```

`build_chunker`, `build_embedder`, `build_llm`, `build_reranker` already
exist in `benchmark/chunking.py`, `benchmark/embedding.py`,
`benchmark/providers.py`, `benchmark/reranker.py`. They are refactored only
to the extent needed to be safely callable with a single config row. If those
helpers currently live inline in `run_single_benchmark()`, extract them to
module-level functions in their respective modules and re-import in
`main.py`.

`_make_retriever_factory(config)` returns a closure
`factory(corpus_records) -> BaseRetriever` that captures
`config.embedding_model`, `config.vector_db_backend`, `config.retrieval_top_k`
and defers vector-store construction until `prepare()` has the corpus.

`_parse_accepts` splits the comma-separated `RAG_ADAPTER_ACCEPTS` env var
into a normalized set. Empty value → empty set → no injection.

### 3. Protocol extension

Update `benchmark/adapters/base.py`:

```python
class RagSystemAdapter(Protocol):
    name: str

    def supports_components(self) -> dict[str, bool]:
        """Declare which ComponentBundle slots this adapter accepts.
        Default for legacy adapters: empty dict = accepts nothing."""
        ...

    def set_components(self, bundle: ComponentBundle) -> None:
        """Receive Framework-built components. Called once before prepare()."""
        ...

    def prepare(self, config, data, corpus=None) -> None: ...
    def answer(self, sample, config) -> RagSystemOutput: ...
```

The two new methods are **not** marked `@abstractmethod` (Protocol structural
typing). Adapters without them keep working; `main.py` uses `getattr` /
`hasattr` checks before calling.

`HttpRagAdapter` and `DemoPythonRagAdapter` do not implement them — they
remain pure black-box.

### 4. `main.py` wiring

In `run_single_benchmark()`, after `adapter = get_rag_adapter(config)`:

```python
if adapter is not None and hasattr(adapter, "supports_components"):
    capabilities = adapter.supports_components()
    config.rag_adapter_accepts = ",".join(
        k for k, v in capabilities.items() if v
    )
    bundle = build_components(config)
    if any(v is not None for v in dataclasses.astuple(bundle)):
        adapter.set_components(bundle)
elif adapter is not None and hasattr(adapter, "set_components"):
    # Adapter accepts injection but does not declare capabilities — trust .env
    bundle = build_components(config)
    if any(v is not None for v in dataclasses.astuple(bundle)):
        adapter.set_components(bundle)
```

Capability flags from the adapter override `RAG_ADAPTER_ACCEPTS` if both are
set — adapter's declaration wins (it knows what it can handle). If the
adapter implements `set_components` but not `supports_components`,
`RAG_ADAPTER_ACCEPTS` from `.env` is used as the sole source of truth.

### 5. New env vars

- `RAG_ADAPTER_ACCEPTS` — comma-separated subset of
  `{chunker, embedder, retriever, reranker, llm, prompt}`. Empty by default.

No other env vars change. `CHUNKING_STRATEGIES`, `EMBEDDING_MODELS`,
`LLM_MODELS`, etc. keep their existing meaning; in external-adapter mode
they are interpreted as "what to inject" rather than "what to run in the
internal pipeline".

### 6. Reference adapter (Enterprise_RAG_Blueprint)

New file: `examples/enterprise_rag_plugin/adapter.py`

```python
from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from benchmark.adapters.components import ComponentBundle


class EnterpriseRagAdapter:
    name = "enterprise_rag"

    def __init__(self, config):
        self.config = config
        self.bundle = ComponentBundle()
        self.chain = None
        self.retriever = None

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": True,
            "embedder": True,
            "retriever": False,   # Enterprise_RAG brings its own
            "reranker": True,
            "llm": True,
            "prompt": False,      # Enterprise_RAG brings its own
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle

    def prepare(self, config, data, corpus=None) -> None:
        from chain.retriever import EnterpriseRetriever
        from chain.load_chain import build_chain

        docs = corpus or data
        self.retriever = EnterpriseRetriever(corpus=docs)
        if self.bundle.chunker is not None:
            self.retriever.chunker = self.bundle.chunker
        if self.bundle.embedder is not None:
            self.retriever.embedder = self.bundle.embedder
        llm = self.bundle.llm
        self.chain = build_chain(llm=llm, retriever=self.retriever)

    def answer(self, sample, config) -> RagSystemOutput:
        import time
        t0 = time.perf_counter()
        result = self.chain.invoke(sample["question"])
        elapsed = time.perf_counter() - t0
        docs = result.get("source_documents", [])
        return RagSystemOutput(
            answer=result.get("answer", ""),
            contexts=[d.page_content for d in docs],
            metadata=[{"doc_id": d.metadata.get("doc_id")} for d in docs],
            total_seconds=elapsed,
        )


register_rag_adapter("enterprise_rag", lambda c: EnterpriseRagAdapter(c))
```

Exact import paths (`chain.retriever`, `chain.load_chain`) must be verified
against the actual Enterprise_RAG_Blueprint codebase during implementation.
The structure above is the contract; the imports are illustrative.

## Data Flow

1. `.env` is loaded → `BenchmarkConfig` row built as today
2. `get_rag_adapter(config)` returns the adapter (e.g. `EnterpriseRagAdapter`)
3. Framework calls `adapter.supports_components()` if present
4. `build_components(config)` builds the bundle (only slots the adapter
   accepts and `.env` requested)
5. `adapter.set_components(bundle)` hands over the bundle
6. `adapter.prepare(config, data, corpus)` runs (adapter uses injected
   components or its own where it declined injection)
7. Per-sample `adapter.answer(sample, config) -> RagSystemOutput`
8. Existing RAGAS + custom metrics evaluate the output
9. Reporting + MLflow logging unchanged

## Error Handling

- `set_components` called with empty bundle → adapter proceeds with its own
  components. No error.
- `build_components` fails to construct a slot (e.g. bad model name) →
  raise the underlying exception with context. Do not silently fall back.
- Adapter `supports_components()` returns a slot the user listed in
  `RAG_ADAPTER_ACCEPTS` but Framework has no config for → bundle slot stays
  `None`, no error logged. Adapter decides how to handle missing slot.
- Adapter implements `set_components` but not `supports_components` →
  `RAG_ADAPTER_ACCEPTS` from `.env` is the sole truth. Same flow otherwise.

## Testing

New tests in `tests/`:

1. `test_component_bundle.py` — `ComponentBundle` defaults, dataclass
   equality, optional slots.
2. `test_build_components.py` — `build_components(config)` with various
   `RAG_ADAPTER_ACCEPTS` subsets; asserts only requested slots populated.
3. `test_adapter_component_injection.py` — fake adapter implementing
   `supports_components` + `set_components`; verify `main.py` wiring calls
   both in correct order, skips when adapter lacks the methods.
4. `test_enterprise_rag_adapter.py` — smoke test using a mocked
   `EnterpriseRetriever` and `build_chain`. Verifies bundle slots get
   passed through, `answer()` returns a valid `RagSystemOutput`.
5. Existing `test_demo_python_rag_plugin.py` must still pass (no
   `set_components` on `DemoPythonRagAdapter`, framework must not break).

Coverage target: ≥85% for new files in `benchmark/adapters/`.

## Migration / Backwards Compatibility

- Existing adapters (`internal`, `http`, `demo_python`) unchanged in
  behavior. `internal` and `http` never implement the new methods; legacy
  plugin adapters do not need changes.
- No env var removed or renamed. New env var `RAG_ADAPTER_ACCEPTS` defaults
  to empty (injection disabled) → identical behavior to today.
- `benchmark/adapters/base.py` Protocol grows two methods but Protocol
  structural typing means existing classes do not need re-declaration.
- Extracting `build_chunker` / `build_embedder` / `build_llm` / `build_reranker`
  into module-level functions is a pure refactor — `main.py` calls them
  from new location, behavior identical.

## Risks

- **Refactor risk in `main.py`**: extracting component builders from
  `run_single_benchmark` may break the matrix sweep for `internal` mode.
  Mitigation: keep existing call sites identical, only add new module-level
  helpers, then have `main.py` call both.
- **LangChain version mismatches**: external RAG may pin a different
  `langchain_core` version. `ComponentBundle` types must match. Mitigation:
  pin `langchain_core` in Framework `requirements.txt`, document version
  constraint in adapter doc.
- **Enterprise_RAG_Blueprint API drift**: reference adapter depends on
  internal imports (`chain.retriever`, `chain.load_chain`). If that
  codebase changes, adapter breaks. Mitigation: pin a submodule commit or
  document the expected snapshot.

## Out of Scope (Follow-up Specs)

- Matrix sweeping component slots across external adapters (e.g.
  `LLM_MODELS=a,b,c` × external adapter produces 3 runs with 3 injected
  LLMs). Today: one bundle per run.
- Component injection over HTTP (`HttpRagAdapter` + components). Would
  require serializing LangChain objects — non-trivial. Defer.
- Auto-discovery of installed adapters (`RAG_SYSTEM=enterprise_rag`
  resolving without `RAG_ADAPTER_MODULES`). Future UX polish.
