# Component Injection for External RAG Adapters — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user specify Framework components (chunker, embedder, LLM, etc.) in `.env` and have an external RAG adapter consume them via dependency injection, evaluated by the existing RAGAS / custom-metric stack.

**Architecture:** A new `ComponentBundle` carries Framework-built components. `build_components(config)` constructs the bundle from `.env`/YAML using existing builders (`get_chunker`, `get_embedding_model`, `get_chat_model`, `get_reranker`, `get_template`). The `RagSystemAdapter` protocol gains two optional methods (`supports_components`, `set_components`). `main.py` calls them when present, before `prepare()`. Reference adapter for `Enterprise_RAG_Blueprint` ships under `examples/`.

**Tech Stack:** Python 3.11+, LangChain (`BaseChatModel`, `Embeddings`, `TextSplitter`), existing Framework modules.

**Spec:** `doc/superpowers/specs/2026-06-17-component-injection-adapter-design.md`

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `benchmark/adapters/components.py` | `ComponentBundle` dataclass, `build_components(config)`, `_parse_accepts`, `_make_retriever_factory` | Create |
| `benchmark/adapters/base.py` | Extend `RagSystemAdapter` Protocol with `supports_components` + `set_components` | Modify |
| `benchmark/adapters/__init__.py` | Re-export `ComponentBundle`, `build_components` | Modify |
| `main.py` | Wire `build_components` + `set_components` between adapter instantiation and `prepare()` | Modify (around lines 199-215) |
| `config.py` | Add `rag_adapter_accepts` field to `BenchmarkConfig` | Modify |
| `tests/test_component_bundle.py` | Unit tests for bundle + factory | Create |
| `tests/test_adapter_component_injection.py` | Tests for `main.py` wiring with fake adapter | Create |
| `tests/test_enterprise_rag_adapter.py` | Smoke test for reference adapter with mocked chain | Create |
| `examples/enterprise_rag_plugin/__init__.py` | Package marker | Create |
| `examples/enterprise_rag_plugin/adapter.py` | Reference adapter for `Enterprise_RAG_Blueprint` | Create |

---

## Task 1: Extend `BenchmarkConfig` with `rag_adapter_accepts`

**Files:**
- Modify: `config.py` (around line 76-98 where existing `rag_system_adapter` lives)
- Test: `tests/test_config.py` (existing — extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_config.py`:

```python
def test_rag_adapter_accepts_defaults_to_empty():
    cfg = BenchmarkConfig(
        llm_model="ollama:gemma3:4b",
        embedding_model="ollama:nomic-embed-text",
        chunk_size=512,
        chunk_overlap=64,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        dataset_name="squad",
    )
    assert cfg.rag_adapter_accepts == ""
```

If `BenchmarkConfig` is constructed elsewhere in the test file via a helper, use that helper instead — match the existing style.

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_config.py::test_rag_adapter_accepts_defaults_to_empty -v
```

Expected: FAIL with `AttributeError: 'BenchmarkConfig' object has no attribute 'rag_adapter_accepts'` or a dataclass `TypeError` for unexpected keyword.

- [ ] **Step 3: Add the field to `BenchmarkConfig`**

In `config.py`, locate the dataclass body where `rag_system_adapter` is declared (around line 98) and add directly below:

```python
    rag_adapter_accepts: str = ""  # comma-separated: chunker,embedder,retriever,reranker,llm,prompt
```

- [ ] **Step 4: Wire env var load**

Find the env-loading section of `config.py` (search for `rag_system_adapter =` to locate where existing `RAG_SYSTEM_ADAPTER` is read). Add right after:

```python
    rag_adapter_accepts=os.getenv("RAG_ADAPTER_ACCEPTS", ""),
```

Use the same indentation and construction style as the surrounding lines. If existing code constructs `BenchmarkConfig(**kwargs)` from a dict, add `rag_adapter_accepts` to that dict instead.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_config.py::test_rag_adapter_accepts_defaults_to_empty -v
```

Expected: PASS.

- [ ] **Step 6: Run full config test module**

```bash
python -m pytest tests/test_config.py -v
```

Expected: All PASS (existing tests still green).

- [ ] **Step 7: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(config): add rag_adapter_accepts field for component injection"
```

---

## Task 2: Create `ComponentBundle` dataclass

**Files:**
- Create: `benchmark/adapters/components.py`
- Test: `tests/test_component_bundle.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_component_bundle.py`:

```python
from benchmark.adapters.components import ComponentBundle


def test_bundle_defaults_all_none():
    b = ComponentBundle()
    assert b.chunker is None
    assert b.embedder is None
    assert b.retriever_factory is None
    assert b.reranker is None
    assert b.llm is None
    assert b.prompt_template is None


def test_bundle_accepts_fields():
    b = ComponentBundle(prompt_template="Answer: {context}")
    assert b.prompt_template == "Answer: {context}"
    assert b.llm is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark.adapters.components'`.

- [ ] **Step 3: Create the module**

Create `benchmark/adapters/components.py`:

```python
"""Framework-built components offered to external RAG adapters via injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class ComponentBundle:
    """Optional LangChain-compatible components built from .env.

    All fields default to None. Adapters pick the slots they support; the
    rest stay None. None means "Framework did not build this slot" or
    "adapter declined injection via supports_components".
    """
    chunker: Optional[Any] = None  # langchain TextSplitter
    embedder: Optional[Any] = None  # langchain Embeddings
    retriever_factory: Optional[Callable[[list[dict]], Any]] = None
    reranker: Optional[Any] = None
    llm: Optional[Any] = None  # langchain BaseChatModel
    prompt_template: Optional[str] = None
```

`Optional[Any]` instead of LangChain concrete types avoids a hard import
dependency at dataclass-definition time. The `TYPE_CHECKING` block below
adds precise types for type-checkers (omitted here for brevity, but the
real file should include it):

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.embeddings import Embeddings
    from langchain_text_splitters import TextSplitter
```

Then swap `Any` for the right types inside `TYPE_CHECKING`. Implementation
can choose either approach — the runtime contract is identical.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add benchmark/adapters/components.py tests/test_component_bundle.py
git commit -m "feat(adapters): add ComponentBundle dataclass"
```

---

## Task 3: Implement `_parse_accepts` helper

**Files:**
- Modify: `benchmark/adapters/components.py`
- Test: `tests/test_component_bundle.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_component_bundle.py`:

```python
import pytest
from benchmark.adapters.components import _parse_accepts, VALID_SLOTS


def test_parse_accepts_empty():
    assert _parse_accepts("") == set()


def test_parse_accepts_single():
    assert _parse_accepts("llm") == {"llm"}


def test_parse_accepts_multiple_with_spaces():
    assert _parse_accepts("llm, chunker , embedder") == {"llm", "chunker", "embedder"}


def test_parse_accepts_rejects_unknown():
    with pytest.raises(ValueError, match="unknown component slot"):
        _parse_accepts("llm,bogus")


def test_valid_slots_contains_expected_keys():
    assert VALID_SLOTS == frozenset(
        {"chunker", "embedder", "retriever", "reranker", "llm", "prompt"}
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: FAIL with `ImportError: cannot import name '_parse_accepts'`.

- [ ] **Step 3: Implement the helper**

Append to `benchmark/adapters/components.py`:

```python
VALID_SLOTS: frozenset[str] = frozenset(
    {"chunker", "embedder", "retriever", "reranker", "llm", "prompt"}
)


def _parse_accepts(raw: str) -> set[str]:
    """Parse comma-separated RAG_ADAPTER_ACCEPTS into a normalized set.

    Raises ValueError on unknown slot names to fail fast at config load.
    """
    if not raw or not raw.strip():
        return set()
    slots = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = slots - VALID_SLOTS
    if unknown:
        raise ValueError(
            f"unknown component slot(s): {sorted(unknown)}. "
            f"Valid: {sorted(VALID_SLOTS)}"
        )
    return slots
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: PASS (all 7 tests).

- [ ] **Step 5: Commit**

```bash
git add benchmark/adapters/components.py tests/test_component_bundle.py
git commit -m "feat(adapters): add _parse_accepts helper with validation"
```

---

## Task 4: Implement `build_components(config)`

**Files:**
- Modify: `benchmark/adapters/components.py`
- Test: `tests/test_component_bundle.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_component_bundle.py`:

```python
from unittest.mock import MagicMock, patch

from benchmark.adapters.components import build_components


def _fake_config(**overrides):
    base = dict(
        llm_model="ollama:gemma3:4b",
        embedding_model="ollama:nomic-embed-text",
        chunk_size=512,
        chunk_overlap=64,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        prompt_template="concise",
        reranker_model=None,
        vector_db_backend="chroma",
        rag_adapter_accepts="",
    )
    base.update(overrides)
    cfg = MagicMock()
    for k, v in base.items():
        setattr(cfg, k, v)
    return cfg


def test_build_components_empty_accepts_returns_empty_bundle():
    cfg = _fake_config()
    b = build_components(cfg)
    assert b.chunker is None
    assert b.llm is None
    assert b.prompt_template is None


def test_build_components_llm_only():
    cfg = _fake_config(rag_adapter_accepts="llm")
    with patch("benchmark.adapters.components.get_chat_model", return_value="FAKE_LLM") as m:
        b = build_components(cfg)
        m.assert_called_once()
    assert b.llm == "FAKE_LLM"
    assert b.chunker is None


def test_build_components_chunker_and_prompt():
    cfg = _fake_config(rag_adapter_accepts="chunker,prompt")
    with patch("benchmark.adapters.components.get_chunker", return_value="FAKE_CHUNKER") as mc, \
         patch("benchmark.adapters.components.get_template", return_value="FAKE_PROMPT") as mt:
        b = build_components(cfg)
        mc.assert_called_once_with("recursive", 512, 64)
        mt.assert_called_once_with("concise")
    assert b.chunker == "FAKE_CHUNKER"
    assert b.prompt_template == "FAKE_PROMPT"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: FAIL with `ImportError: cannot import name 'build_components'`.

- [ ] **Step 3: Implement `build_components`**

Append to `benchmark/adapters/components.py`:

```python
from benchmark.chunking import get_chunker
from benchmark.embedding import get_embedding_model
from benchmark.prompt_templates import get_template
from benchmark.providers import get_chat_model
from benchmark.reranker import get_reranker


def build_components(config) -> ComponentBundle:
    """Build a ComponentBundle honoringings RAG_ADAPTER_ACCEPTS.

    Slots the user did not list stay None. If config.rag_adapter_accepts is
    empty, returns an empty bundle.
    """
    accepts = _parse_accepts(getattr(config, "rag_adapter_accepts", ""))
    bundle = ComponentBundle()

    if "chunker" in accepts and getattr(config, "chunking_strategy", ""):
        bundle.chunker = get_chunker(
            config.chunking_strategy,
            int(config.chunk_size),
            int(config.chunk_overlap),
        )

    if "embedder" in accepts and getattr(config, "embedding_model", ""):
        bundle.embedder = get_embedding_model(config.embedding_model)

    if "llm" in accepts and getattr(config, "llm_model", ""):
        bundle.llm = get_chat_model(config.llm_model)

    if "reranker" in accepts and getattr(config, "reranker_model", None):
        bundle.reranker = get_reranker(config.reranker_model)

    if "prompt" in accepts and getattr(config, "prompt_template", ""):
        bundle.prompt_template = get_template(config.prompt_template)

    if "retriever" in accepts and getattr(config, "embedding_model", ""):
        bundle.retriever_factory = _make_retriever_factory(config)

    return bundle


def _make_retriever_factory(config):
    """Return a closure that builds a retriever once the corpus is available.

    The factory defers vector-store construction until prepare() has the
    corpus records. This matches the Framework's existing two-stage pattern
    (index → query) but lets the adapter decide when to call it.
    """
    def factory(corpus_records: list[dict]):
        # Import locally to avoid a heavy import at module load.
        from benchmark.retrieval import build_vector_store
        return build_vector_store(
            records=corpus_records,
            embedding_model=config.embedding_model,
            backend=config.vector_db_backend,
            chunk_size=int(config.chunk_size),
            chunk_overlap=int(config.chunk_overlap),
            chunking_strategy=config.chunking_strategy,
            top_k=int(config.retrieval_top_k),
        )
    return factory
```

**Verify before continuing:** if `get_chat_model` in `benchmark/providers.py` does not accept a single model-string argument, adjust the call to match its real signature. Check with:

```bash
grep -n "^def get_chat_model" benchmark/providers.py
```

If the signature differs (e.g. requires `base_url`, `api_key`), update the test patch target accordingly and the call in `build_components` to use the existing pattern from `main.py` lines 278-285. Same for `get_embedding_model`. Document the resolved signature in the commit message.

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest tests/test_component_bundle.py -v
```

Expected: PASS (all 10 tests).

- [ ] **Step 5: Commit**

```bash
git add benchmark/adapters/components.py tests/test_component_bundle.py
git commit -m "feat(adapters): implement build_components factory"
```

---

## Task 5: Extend `RagSystemAdapter` Protocol

**Files:**
- Modify: `benchmark/adapters/base.py`
- Modify: `benchmark/adapters/__init__.py`
- Test: `tests/test_adapter_protocol.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapter_protocol.py`:

```python
"""Protocol is structural — verify optional methods are truly optional."""
from benchmark.adapters.base import RagSystemAdapter
from benchmark.adapters import (
    ComponentBundle,
    build_components,
    RagSystemOutput,
)


def test_protocol_optional_methods_absent_on_legacy_adapter():
    class LegacyAdapter:
        name = "legacy"
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="x")

    inst = LegacyAdapter()
    # The two new methods are NOT required to exist:
    assert not hasattr(inst, "supports_components")
    assert not hasattr(inst, "set_components")


def test_protocol_accepts_full_implementation():
    class FullAdapter:
        name = "full"
        def supports_components(self):
            return {"llm": True, "chunker": False}
        def set_components(self, bundle: ComponentBundle):
            self.bundle = bundle
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="x")

    inst = FullAdapter()
    inst.set_components(ComponentBundle(llm="FAKE"))
    assert inst.bundle.llm == "FAKE"
    assert inst.supports_components() == {"llm": True, "chunker": False}


def test_re_exports():
    assert ComponentBundle is not None
    assert callable(build_components)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_adapter_protocol.py -v
```

Expected: FAIL on `test_re_exports` with `ImportError: cannot import name 'ComponentBundle'`.

- [ ] **Step 3: Extend the Protocol**

In `benchmark/adapters/base.py`, modify the `RagSystemAdapter` Protocol:

```python
from benchmark.adapters.components import ComponentBundle


class RagSystemAdapter(Protocol):
    """A black-box RAG system that can answer benchmark samples.

    Two optional methods support component injection. Adapters that do not
    implement them are treated as pure black-box — the Framework will not
    attempt to inject components.
    """

    name: str

    def supports_components(self) -> dict[str, bool]:
        """Declare which ComponentBundle slots this adapter accepts.
        Keys: any of {chunker, embedder, retriever, reranker, llm, prompt}.
        Value True means the adapter will use a Framework-provided slot if
        given one. Method is optional; absence = accepts nothing.
        """
        ...

    def set_components(self, bundle: ComponentBundle) -> None:
        """Receive Framework-built components. Called once before prepare().
        Optional; absence = pure black-box adapter.
        """
        ...

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        """Prepare the system before per-sample queries run."""

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        """Return a normalized answer and optional retrieval evidence."""
```

Note: `ComponentBundle` is imported at the top of `base.py`. To avoid a
circular import (`components.py` will need Protocol only under
`TYPE_CHECKING`), confirm `components.py` does NOT import from `base.py` at
runtime. It currently does not.

- [ ] **Step 4: Re-export from `__init__.py`**

In `benchmark/adapters/__init__.py`, add to imports and `__all__`:

```python
from benchmark.adapters.components import ComponentBundle, build_components
```

Extend `__all__` to include `"ComponentBundle"` and `"build_components"`.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_adapter_protocol.py -v
```

Expected: PASS (all 3 tests).

- [ ] **Step 6: Run full test suite to catch regressions**

```bash
python -m pytest tests/ -v
```

Expected: All existing tests still PASS. If something fails due to circular imports, move the `from benchmark.adapters.components import ...` line in `base.py` under `TYPE_CHECKING` and use a string annotation `"ComponentBundle"`.

- [ ] **Step 7: Commit**

```bash
git add benchmark/adapters/base.py benchmark/adapters/__init__.py tests/test_adapter_protocol.py
git commit -m "feat(adapters): extend RagSystemAdapter protocol with optional component injection"
```

---

## Task 6: Wire injection into `main.py`

**Files:**
- Modify: `main.py` (around lines 199-215)
- Test: `tests/test_adapter_component_injection.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_adapter_component_injection.py`:

```python
"""Integration-level test: main.py wires build_components → set_components."""
from unittest.mock import MagicMock, patch

from benchmark.adapters import ComponentBundle


def test_inject_components_called_when_adapter_supports(monkeypatch):
    captured_bundle = {}

    class FakeAdapter:
        name = "fake"
        def __init__(self, cfg): self.cfg = cfg
        def supports_components(self):
            return {"llm": True, "chunker": True}
        def set_components(self, bundle: ComponentBundle):
            captured_bundle["bundle"] = bundle
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            from benchmark.adapters.base import RagSystemOutput
            return RagSystemOutput(answer="ok")

    fake_bundle = ComponentBundle(llm="FAKE_LLM", chunker="FAKE_CHUNKER")

    with patch("main.get_rag_adapter", return_value=FakeAdapter(MagicMock())), \
         patch("main.build_components", return_value=fake_bundle) as bc_mock:
        import main
        cfg = MagicMock()
        cfg.benchmark_stage = "all"
        cfg.name = "test"
        cfg.retrieval_mode = "retrieval"
        cfg.rag_adapter_accepts = "llm,chunker"
        data = [{"question": "q", "ground_truth": "g", "context": "c"}]

        main.run_single_benchmark(cfg, data)

    assert "bundle" in captured_bundle
    assert captured_bundle["bundle"].llm == "FAKE_LLM"
    bc_mock.assert_called_once()


def test_set_components_not_called_when_adapter_lacks_method():
    class LegacyAdapter:
        name = "legacy"
        def __init__(self, cfg): pass
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            from benchmark.adapters.base import RagSystemOutput
            return RagSystemOutput(answer="ok")

    with patch("main.get_rag_adapter", return_value=LegacyAdapter()), \
         patch("main.build_components", return_value=ComponentBundle()) as bc_mock:
        import main
        cfg = MagicMock()
        cfg.benchmark_stage = "all"
        cfg.name = "test"
        cfg.retrieval_mode = "retrieval"
        cfg.rag_adapter_accepts = ""
        data = [{"question": "q", "ground_truth": "g", "context": "c"}]

        main.run_single_benchmark(cfg, data)

    # build_components should not even be called for legacy adapters
    bc_mock.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_adapter_component_injection.py -v
```

Expected: FAIL with `AttributeError: module 'main' has no attribute 'build_components'`.

- [ ] **Step 3: Add import to `main.py`**

At top of `main.py`, near the existing `from benchmark.adapters import get_rag_adapter` (line 36), extend:

```python
from benchmark.adapters import get_rag_adapter, build_components, ComponentBundle
```

- [ ] **Step 4: Insert injection logic**

In `run_single_benchmark()` between line 199 (`rag_adapter = get_rag_adapter(config)`) and line 210 (`# 1. Chunk + 2. Embed`), insert:

```python
    # Component injection: hand Framework-built components to external adapter
    if rag_adapter is not None and hasattr(rag_adapter, "set_components"):
        if hasattr(rag_adapter, "supports_components"):
            capabilities = rag_adapter.supports_components()
            accepted = ",".join(k for k, v in capabilities.items() if v)
            if accepted:
                config.rag_adapter_accepts = accepted
        bundle = build_components(config)
        if any(v is not None for v in (
            bundle.chunker, bundle.embedder, bundle.retriever_factory,
            bundle.reranker, bundle.llm, bundle.prompt_template,
        )):
            rag_adapter.set_components(bundle)
            console.print(
                f"  [dim]Injected components: "
                f"{', '.join(k for k, v in {
                    'chunker': bundle.chunker, 'embedder': bundle.embedder,
                    'retriever': bundle.retriever_factory, 'reranker': bundle.reranker,
                    'llm': bundle.llm, 'prompt': bundle.prompt_template,
                }.items() if v is not None)}[/dim]"
            )
```

Note: the test mocks `build_components` to return an empty bundle in the legacy case — the `any(...)` check ensures `set_components` is not called with a no-op bundle. The legacy test also patches `build_components` but asserts it is NOT called, because the wrapping `if ... hasattr(rag_adapter, 'set_components'):` is False for legacy adapters.

- [ ] **Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_adapter_component_injection.py -v
```

Expected: PASS (both tests).

If the legacy test fails because `build_components` was called, re-check the `hasattr` guard — legacy adapter does not implement `set_components`, so the entire block must be skipped.

- [ ] **Step 6: Run full test suite**

```bash
python -m pytest tests/ -v
```

Expected: All PASS. Existing `test_demo_python_rag_plugin.py` must still PASS (it uses an adapter without `set_components`).

- [ ] **Step 7: Commit**

```bash
git add main.py tests/test_adapter_component_injection.py
git commit -m "feat(main): wire component injection between adapter creation and prepare()"
```

---

## Task 7: Create reference adapter for `Enterprise_RAG_Blueprint`

**Files:**
- Create: `examples/enterprise_rag_plugin/__init__.py`
- Create: `examples/enterprise_rag_plugin/adapter.py`
- Test: `tests/test_enterprise_rag_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_enterprise_rag_adapter.py`:

```python
"""Smoke test for the Enterprise_RAG_Blueprint reference adapter.

Mocks the Enterprise_RAG chain imports so the test does not require the
full blueprint codebase. Verifies the adapter:
1. Declares expected capability flags.
2. Honors set_components by routing bundle slots into the chain.
3. Returns a valid RagSystemOutput with contexts + metadata.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def fake_enterprise_modules(monkeypatch):
    """Inject fake `chain.retriever` and `chain.load_chain` modules."""
    fake_chain_pkg = types.ModuleType("chain")
    fake_retriever_mod = types.ModuleType("chain.retriever")
    fake_load_mod = types.ModuleType("chain.load_chain")

    fake_retriever = MagicMock()
    fake_retriever_mod.EnterpriseRetriever = MagicMock(return_value=fake_retriever)

    fake_chain_obj = MagicMock()
    fake_chain_obj.invoke.return_value = {
        "answer": "42",
        "source_documents": [
            MagicMock(page_content="ctx-A", metadata={"doc_id": "doc-a"}),
            MagicMock(page_content="ctx-B", metadata={"doc_id": "doc-b"}),
        ],
    }
    fake_load_mod.build_chain = MagicMock(return_value=fake_chain_obj)

    fake_chain_pkg.retriever = fake_retriever_mod
    fake_chain_pkg.load_chain = fake_load_mod

    monkeypatch.setitem(sys.modules, "chain", fake_chain_pkg)
    monkeypatch.setitem(sys.modules, "chain.retriever", fake_retriever_mod)
    monkeypatch.setitem(sys.modules, "chain.load_chain", fake_load_mod)
    return {
        "retriever": fake_retriever,
        "chain": fake_chain_obj,
        "build_chain": fake_load_mod.build_chain,
        "EnterpriseRetriever": fake_retriever_mod.EnterpriseRetriever,
    }


def test_supports_components(fake_enterprise_modules):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter
    adapter = EnterpriseRagAdapter(MagicMock())
    caps = adapter.supports_components()
    assert caps["chunker"] is True
    assert caps["embedder"] is True
    assert caps["retriever"] is False
    assert caps["reranker"] is True
    assert caps["llm"] is True
    assert caps["prompt"] is False


def test_prepare_passes_injected_components(fake_enterprise_modules):
    from benchmark.adapters.components import ComponentBundle
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    adapter = EnterpriseRagAdapter(MagicMock())
    adapter.set_components(ComponentBundle(
        chunker="FAKE_CHUNKER",
        embedder="FAKE_EMBEDDER",
        llm="FAKE_LLM",
    ))
    adapter.prepare(MagicMock(), data=[], corpus=[{"context": "doc1"}])

    assert adapter.retriever.chunker == "FAKE_CHUNKER"
    assert adapter.retriever.embedder == "FAKE_EMBEDDER"
    fake_enterprise_modules["build_chain"].assert_called_once()
    _, kwargs = fake_enterprise_modules["build_chain"].call_args
    assert kwargs["llm"] == "FAKE_LLM"


def test_answer_returns_valid_output(fake_enterprise_modules):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    adapter = EnterpriseRagAdapter(MagicMock())
    adapter.prepare(MagicMock(), data=[], corpus=[{"context": "doc1"}])
    out = adapter.answer({"question": "what?"}, MagicMock())
    assert out.answer == "42"
    assert out.contexts == ["ctx-A", "ctx-B"]
    assert {"doc_id": "doc-a"} in out.metadata
    assert out.total_seconds >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_enterprise_rag_adapter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'examples.enterprise_rag_plugin'`.

- [ ] **Step 3: Create package marker**

Create `examples/enterprise_rag_plugin/__init__.py` (empty file):

```python
```

- [ ] **Step 4: Create the adapter**

Create `examples/enterprise_rag_plugin/adapter.py`:

```python
"""Reference adapter for Enterprise_RAG_Blueprint.

Demonstrates the component-injection contract: accepts Framework chunker,
embedder, reranker, and LLM, but keeps its own retriever and prompt.

Activation (bash):

    PYTHONPATH=examples \
    RAG_ADAPTER_MODULES=enterprise_rag_plugin.adapter \
    RAG_SYSTEM_ADAPTER=enterprise_rag \
    RAG_ADAPTER_ACCEPTS=chunker,embedder,llm,reranker \
    BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_eval.yaml \
    python main.py

NOTE: import paths `chain.retriever.EnterpriseRetriever` and
`chain.load_chain.build_chain` must match the actual Enterprise_RAG_Blueprint
codebase at integration time. Adjust the imports below if they differ.
"""

from __future__ import annotations

import time
from typing import Any

from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from benchmark.adapters.components import ComponentBundle


class EnterpriseRagAdapter:
    name = "enterprise_rag"

    def __init__(self, config: Any):
        self.config = config
        self.bundle: ComponentBundle = ComponentBundle()
        self.chain = None
        self.retriever = None

    def supports_components(self) -> dict[str, bool]:
        return {
            "chunker": True,
            "embedder": True,
            "retriever": False,
            "reranker": True,
            "llm": True,
            "prompt": False,
        }

    def set_components(self, bundle: ComponentBundle) -> None:
        self.bundle = bundle

    def prepare(
        self,
        config: Any,
        data: list[dict],
        corpus: list[dict] | None = None,
    ) -> None:
        from chain.retriever import EnterpriseRetriever
        from chain.load_chain import build_chain

        docs = corpus if corpus is not None else data
        self.retriever = EnterpriseRetriever(corpus=docs)

        if self.bundle.chunker is not None:
            self.retriever.chunker = self.bundle.chunker
        if self.bundle.embedder is not None:
            self.retriever.embedder = self.bundle.embedder
        if self.bundle.reranker is not None:
            self.retriever.reranker = self.bundle.reranker

        llm = self.bundle.llm
        self.chain = build_chain(llm=llm, retriever=self.retriever)

    def answer(self, sample: dict, config: Any) -> RagSystemOutput:
        if self.chain is None:
            raise RuntimeError("EnterpriseRagAdapter.prepare() was not called")
        t0 = time.perf_counter()
        result = self.chain.invoke(sample["question"])
        elapsed = time.perf_counter() - t0

        docs = result.get("source_documents", []) if isinstance(result, dict) else []
        return RagSystemOutput(
            answer=result.get("answer", "") if isinstance(result, dict) else str(result),
            contexts=[getattr(d, "page_content", str(d)) for d in docs],
            metadata=[
                getattr(d, "metadata", {}) or {"doc_id": None} for d in docs
            ],
            total_seconds=elapsed,
        )


register_rag_adapter("enterprise_rag", lambda c: EnterpriseRagAdapter(c))
```

- [ ] **Step 5: Run test to verify it passes**

```bash
PYTHONPATH=. python -m pytest tests/test_enterprise_rag_adapter.py -v
```

Expected: PASS (all 3 tests).

- [ ] **Step 6: Run full test suite**

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Expected: All PASS.

- [ ] **Step 7: Commit**

```bash
git add examples/enterprise_rag_plugin/ tests/test_enterprise_rag_adapter.py
git commit -m "feat(examples): add Enterprise_RAG_Blueprint reference adapter"
```

---

## Task 8: Add YAML experiment manifest + end-to-end smoke

**Files:**
- Create: `experiments/enterprise_rag_demo.yaml`

- [ ] **Step 1: Write the manifest**

Create `experiments/enterprise_rag_demo.yaml`:

```yaml
experiment_name: enterprise-rag-demo

dataset:
  name: squad
  sample_size: [5]

settings:
  rag_system_adapter: enterprise_rag
  rag_adapter_accepts: "chunker,embedder,llm"
  retrieval_top_k: 5
  eval_critic_llm: ollama:gemma3:12b
  eval_critic_embedding: nomic-embed-text:latest
  ragas_enabled: true
  custom_metrics_enabled: true
  vector_db_backend: chroma

matrix:
  llm_models: ["ollama:gemma3:4b"]
  embedding_models: ["ollama:nomic-embed-text:latest"]
  chunking_strategies: ["recursive"]
  chunk_sizes: [512]
  chunk_overlaps: [64]
  prompt_templates: ["concise"]
```

- [ ] **Step 2: Validate YAML parses**

```bash
python -c "import yaml; yaml.safe_load(open('experiments/enterprise_rag_demo.yaml'))"
```

Expected: no output (success).

- [ ] **Step 3: Document activation command in spec**

Append a "Quickstart" section to `doc/External_RAG_Python_Adapter.md` with the bash command shown in the adapter docstring (Task 7). This keeps discoverability — the existing doc is the canonical entry point.

- [ ] **Step 4: Commit**

```bash
git add experiments/enterprise_rag_demo.yaml doc/External_RAG_Python_Adapter.md
git commit -m "feat(experiments): add enterprise_rag_demo manifest and quickstart"
```

---

## Task 9: Update `doc/External_RAG_Python_Adapter.md`

**Files:**
- Modify: `doc/External_RAG_Python_Adapter.md`

- [ ] **Step 1: Append component-injection section**

Append to `doc/External_RAG_Python_Adapter.md`:

```markdown
---

## 8. Component Injection (optional)

If your RAG can accept Framework-built components (chunker, embedder, LLM,
etc.), implement two extra methods on your adapter class:

```python
def supports_components(self) -> dict[str, bool]:
    return {
        "chunker": True, "embedder": True, "retriever": False,
        "reranker": False, "llm": True, "prompt": False,
    }

def set_components(self, bundle: ComponentBundle) -> None:
    self.bundle = bundle
```

The Framework reads `supports_components()`, builds the requested slots from
`.env`/YAML, and calls `set_components()` once before `prepare()`. Slots
marked `False` are not built; slots marked `True` but unset in `.env` arrive
as `None`.

User-side `.env`:

```env
RAG_SYSTEM_ADAPTER=myrag
RAG_ADAPTER_MODULES=my_pkg.my_adapter
RAG_ADAPTER_ACCEPTS=chunker,embedder,llm
CHUNKING_STRATEGIES=recursive
EMBEDDING_MODELS=ollama:nomic-embed-text
LLM_MODELS=ollama:gemma3:4b
```

If your adapter only implements `set_components()` (no
`supports_components()`), `RAG_ADAPTER_ACCEPTS` from `.env` is the sole
source of truth.

Reference impl: `examples/enterprise_rag_plugin/adapter.py`.
```

- [ ] **Step 2: Commit**

```bash
git add doc/External_RAG_Python_Adapter.md
git commit -m "docs(external-rag): document component injection contract"
```

---

## Self-Review

**Spec coverage:**
- §1 ComponentBundle → Task 2
- §2 build_components → Task 4
- §3 Protocol extension → Task 5
- §4 main.py wiring → Task 6
- §5 New env vars → Task 1 (`RAG_ADAPTER_ACCEPTS`)
- §6 Reference adapter → Task 7
- §7 Data Flow → Task 6 + Task 7
- §8 Error Handling → `_parse_accepts` validation (Task 3), empty-bundle skip (Task 6)
- §9 Testing → all tasks have TDD tests
- §10 Migration/Backwards-compat → Task 5 keeps Protocol structural, Task 6 `hasattr` guards
- §11 Risks → Task 4 Step 3 footnote about verifying builder signatures; Task 7 NOTE about import paths

**Placeholder scan:** None found. Every step has code or commands. Two "verify signature" notes (Task 4 Step 3, Task 7 docstring) are explicit verification instructions, not placeholders.

**Type consistency:** `ComponentBundle` field names consistent across Tasks 2, 4, 5, 6, 7. `supports_components()` return shape consistent. `set_components(bundle)` signature consistent. `RagSystemOutput` already defined in `base.py`.

---

## Execution Handoff

Plan complete and saved to `doc/superpowers/plans/2026-06-17-component-injection-adapter.md`.

Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — batch execution with checkpoints in this session

Which approach?
