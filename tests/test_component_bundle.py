from unittest.mock import MagicMock, patch

import pytest

from benchmark.adapters.components import (
    ComponentBundle,
    VALID_SLOTS,
    _parse_accepts,
    build_components,
)


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


def test_bundle_equality_empty():
    assert ComponentBundle() == ComponentBundle()


def test_bundle_equality_with_fields():
    a = ComponentBundle(prompt_template="Answer: {context}")
    b = ComponentBundle(prompt_template="Answer: {context}")
    c = ComponentBundle(prompt_template="Different: {context}")
    assert a == b
    assert a != c


def test_bundle_hashable_when_frozen():
    # frozen=True enables __hash__; this raises TypeError if not frozen
    assert isinstance(hash(ComponentBundle()), int)


def test_bundle_retriever_factory_assignment():
    factory = lambda docs: docs  # noqa: E731
    b = ComponentBundle(retriever_factory=factory)
    assert b.retriever_factory is not None
    assert callable(b.retriever_factory)


# ---------------------------------------------------------------------------
# _parse_accepts / VALID_SLOTS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# build_components
# ---------------------------------------------------------------------------

def _fake_config(**overrides):
    base = dict(
        llm_model="ollama:gemma3:4b",
        llm_provider="ollama",
        embedding_model="ollama:nomic-embed-text",
        chunk_size=512,
        chunk_overlap=64,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        prompt_template="concise",
        reranker_model=None,
        vector_db_backend="chroma",
        rag_adapter_accepts="",
        max_new_tokens=256,
        lancedb_path=".lancedb",
    )
    base.update(overrides)
    cfg = MagicMock()
    for k, v in base.items():
        setattr(cfg, k, v)
    cfg.embedding_base_url.return_value = "http://localhost:11434"
    cfg.embedding_api_key.return_value = None
    cfg.embedding_provider = base["embedding_model"].split(":")[0]
    cfg.llm_base_url.return_value = "http://localhost:11434"
    cfg.llm_api_key.return_value = None
    return cfg


def test_build_components_empty_accepts_returns_empty_bundle():
    cfg = _fake_config()
    b = build_components(cfg)
    assert b.chunker is None
    assert b.llm is None
    assert b.prompt_template is None


def test_build_components_llm_only():
    cfg = _fake_config(rag_adapter_accepts="llm")
    with patch("benchmark.adapters.components.get_llm", return_value="FAKE_LLM") as m:
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


def test_build_components_retriever_factory_built():
    cfg = _fake_config(rag_adapter_accepts="retriever")
    b = build_components(cfg)
    assert callable(b.retriever_factory)
    assert b.llm is None
