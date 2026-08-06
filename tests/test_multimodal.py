"""Tests for benchmark.multimodal — registry, dispatch, optional-dep handling.

Heavy ML deps (docling, colpali-engine, faster-whisper, openai-whisper, torch)
are mocked throughout. These tests verify:
  - Registry contents and dispatch
  - corpus chunk shape produced by multimodal factories (mocked backends)
  - OptionalDependencyError raised when a backend isn't importable
  - ColPaliIndex max-sim scoring with synthetic embeddings (no torch needed
    for the math path when embeddings are passed as raw lists)
  - chunking.known_chunking_strategies / run_multimodal_ingestion dispatch
  - config validation accepts multimodal strategies and rejects mismatches
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_registered_chunker_names(self):
        from benchmark.multimodal import available_multimodal_chunkers

        names = available_multimodal_chunkers()
        assert "pdf_ocr_docling" in names
        assert "image_ocr_docling" in names
        assert "audio_whisper" in names

    def test_registered_embedder_names(self):
        from benchmark.multimodal import available_multimodal_embedders

        assert "colpali_visual" in available_multimodal_embedders()

    def test_is_chunker_multimodal_true(self):
        from benchmark.multimodal import is_chunker_multimodal

        assert is_chunker_multimodal("pdf_ocr_docling") is True
        assert is_chunker_multimodal("audio_whisper") is True

    def test_is_chunker_multimodal_false_for_text_strategies(self):
        from benchmark.multimodal import is_chunker_multimodal

        assert is_chunker_multimodal("recursive") is False
        assert is_chunker_multimodal("semantic") is False
        assert is_chunker_multimodal("nonexistent") is False

    def test_is_embedder_multimodal(self):
        from benchmark.multimodal import is_embedder_multimodal

        assert is_embedder_multimodal("colpali_visual") is True
        assert is_embedder_multimodal("nomic-embed-text:latest") is False


# ---------------------------------------------------------------------------
# OptionalDependencyError
# ---------------------------------------------------------------------------


class TestOptionalDependencyError:
    def test_error_carries_install_hint(self):
        from benchmark.multimodal.registry import OptionalDependencyError

        err = OptionalDependencyError(
            feature="pdf_ocr_docling", package="docling", extra="docling"
        )
        assert "docling" in str(err)
        assert "pip install docling" in str(err)
        assert err.feature == "pdf_ocr_docling"
        assert err.install_target == "docling"


# ---------------------------------------------------------------------------
# Docling OCR — mocked backend
# ---------------------------------------------------------------------------


def _fake_docling_module():
    """Build a fake `docling` package surface that the wrapper will accept."""
    docling = types.ModuleType("docling")
    document_converter = types.ModuleType("docling.document_converter")
    base_models = types.ModuleType("docling.datamodel.base_models")
    pipeline_options = types.ModuleType("docling.datamodel.pipeline_options")

    # Fake classes — the wrapper just constructs and calls them.
    class _PdfPipelineOptions:
        def __init__(self, do_ocr=True, do_table_structure=True):
            self.do_ocr = do_ocr
            self.do_table_structure = do_table_structure

    class _PdfFormatOption:
        def __init__(self, pipeline_options=None):
            self.pipeline_options = pipeline_options

    class _InputFormat:
        PDF = "pdf"

    class _ConversionResult:
        def __init__(self, document):
            self.document = document

    class _Document:
        def __init__(self, page_texts):
            self._page_texts = page_texts

        def iterate_items(self):
            class _Item:
                def __init__(self, text, page_no):
                    self.text = text
                    self.prov = [types.SimpleNamespace(page_no=page_no)]

            class _Level0:
                pass

            for page_no, text in self._page_texts:
                yield _Item(text, page_no), 0

        def export_to_markdown(self):
            return "\n".join(t for _, t in self._page_texts)

    class _DocumentConverter:
        def __init__(self, format_options=None):
            self.format_options = format_options

        def convert(self, source):
            # Return two pages of fake text per call.
            return _ConversionResult(
                _Document([(1, "Page one text."), (2, "Page two text.")])
            )

    document_converter.DocumentConverter = _DocumentConverter
    document_converter.PdfFormatOption = _PdfFormatOption
    document_converter.ConversionResult = _ConversionResult
    base_models.InputFormat = _InputFormat
    pipeline_options.PdfPipelineOptions = _PdfPipelineOptions

    datamodel = types.ModuleType("docling.datamodel")
    datamodel.base_models = base_models
    datamodel.pipeline_options = pipeline_options

    docling.document_converter = document_converter
    docling.datamodel = datamodel
    return docling


class TestDoclingOcr:
    def test_ingest_pdf_walks_directory(self, tmp_path, monkeypatch):
        # Create a fake PDF file (contents don't matter; converter is mocked).
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "b.pdf").write_bytes(b"%PDF-1.4 fake")
        # Non-PDF file ignored.
        (tmp_path / "notes.txt").write_text("ignore me")

        fake = _fake_docling_module()
        monkeypatch.setitem(sys.modules, "docling", fake)
        monkeypatch.setitem(sys.modules, "docling.document_converter", fake.document_converter)
        monkeypatch.setitem(sys.modules, "docling.datamodel", fake.datamodel)
        monkeypatch.setitem(sys.modules, "docling.datamodel.base_models", fake.datamodel.base_models)
        monkeypatch.setitem(sys.modules, "docling.datamodel.pipeline_options", fake.datamodel.pipeline_options)

        from benchmark.multimodal.docling_ocr import ingest_pdf_via_docling

        result = ingest_pdf_via_docling(tmp_path)
        # 2 PDFs x 2 pages each = 4 corpus dicts.
        assert len(result) == 4
        for entry in result:
            assert "context" in entry and "metadata" in entry
            assert entry["metadata"]["corpus_type"] == "pdf"
            assert "doc_id" in entry["metadata"]
            assert "source_name" in entry["metadata"]
        # Page numbering resets per source PDF.
        page_numbers = [entry["metadata"]["page_no"] for entry in result]
        assert page_numbers == [1, 2, 1, 2]

    def test_ingest_pdf_missing_dep_raises(self, tmp_path, monkeypatch):
        (tmp_path / "a.pdf").write_bytes(b"%PDF-1.4 fake")
        # Force ImportError for docling.
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name.startswith("docling"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        # Purge any cached docling modules.
        for mod in list(sys.modules):
            if mod.startswith("docling"):
                del sys.modules[mod]

        from benchmark.multimodal.docling_ocr import ingest_pdf_via_docling
        from benchmark.multimodal.registry import OptionalDependencyError

        with pytest.raises(OptionalDependencyError) as exc_info:
            ingest_pdf_via_docling(tmp_path)
        assert "docling" in str(exc_info.value)

    def test_ingest_pdf_empty_directory_raises(self, tmp_path):
        from benchmark.multimodal.docling_ocr import ingest_pdf_via_docling

        with pytest.raises(ValueError, match="No PDF documents found"):
            ingest_pdf_via_docling(tmp_path)

    def test_ingest_pdf_non_directory_raises(self, tmp_path):
        from benchmark.multimodal.docling_ocr import ingest_pdf_via_docling

        not_a_dir = tmp_path / "file.pdf"
        not_a_dir.write_bytes(b"x")
        with pytest.raises(ValueError, match="not a directory"):
            ingest_pdf_via_docling(not_a_dir)


# ---------------------------------------------------------------------------
# Whisper ASR — mocked backend
# ---------------------------------------------------------------------------


class TestWhisperAsr:
    def _build_corpus_dir(self, tmp_path):
        (tmp_path / "a.mp3").write_bytes(b"fake audio")
        (tmp_path / "b.wav").write_bytes(b"fake audio")
        (tmp_path / "ignore.txt").write_text("not audio")
        return tmp_path

    def test_ingest_faster_whisper(self, tmp_path, monkeypatch):
        corpus_dir = self._build_corpus_dir(tmp_path)

        fake_module = types.ModuleType("faster_whisper")

        class _FakeWhisperModel:
            def __init__(self, *args, **kwargs):
                pass

            def transcribe(self, path, language=None, beam_size=5):
                class _Info:
                    language = "en"
                    language_probability = 0.99
                    duration = 1.0
                    duration_after_vad = 1.0

                class _Segment:
                    def __init__(self, text):
                        self.text = text
                        self.start = 0.0
                        self.end = 1.0

                return iter([_Segment("hello world")]), _Info()

        fake_module.WhisperModel = _FakeWhisperModel
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_module)

        from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper

        result = ingest_audio_via_whisper(corpus_dir, backend="faster-whisper")
        assert len(result) == 2
        for entry in result:
            assert entry["context"] == "hello world"
            assert entry["metadata"]["corpus_type"] == "audio"
            assert entry["metadata"]["whisper_backend"] == "faster-whisper"

    def test_ingest_openai_whisper(self, tmp_path, monkeypatch):
        corpus_dir = self._build_corpus_dir(tmp_path)

        fake_module = types.ModuleType("whisper")

        class _FakeModel:
            def transcribe(self, path, language=None):
                return {"text": "transcribed text"}

            @staticmethod
            def load_model(*args, **kwargs):
                return _FakeModel()

        fake_module.load_model = _FakeModel.load_model
        fake_module.WhisperModel = _FakeModel
        monkeypatch.setitem(sys.modules, "whisper", fake_module)

        from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper

        result = ingest_audio_via_whisper(corpus_dir, backend="openai-whisper")
        assert len(result) == 2
        assert all(entry["context"] == "transcribed text" for entry in result)

    def test_ingest_unknown_backend_raises(self, tmp_path, monkeypatch):
        corpus_dir = self._build_corpus_dir(tmp_path)
        # Stub both backends to bypass import gates.
        monkeypatch.setitem(sys.modules, "faster_whisper", types.ModuleType("faster_whisper"))
        monkeypatch.setitem(sys.modules, "whisper", types.ModuleType("whisper"))

        from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper

        with pytest.raises(ValueError, match="Unknown Whisper backend"):
            ingest_audio_via_whisper(corpus_dir, backend="deepspeech")

    def test_ingest_missing_dep_raises(self, tmp_path, monkeypatch):
        corpus_dir = self._build_corpus_dir(tmp_path)
        import builtins

        real_import = builtins.__import__

        def _fail(name, *args, **kwargs):
            if name in ("faster_whisper", "whisper"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail)
        for mod in list(sys.modules):
            if mod in ("faster_whisper", "whisper"):
                del sys.modules[mod]

        from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper
        from benchmark.multimodal.registry import OptionalDependencyError

        with pytest.raises(OptionalDependencyError):
            ingest_audio_via_whisper(corpus_dir)

    def test_ingest_empty_directory_raises(self, tmp_path):
        from benchmark.multimodal.whisper_asr import ingest_audio_via_whisper

        with pytest.raises(ValueError, match="No audio documents found"):
            ingest_audio_via_whisper(tmp_path)


# ---------------------------------------------------------------------------
# ColPali embedder — registration + max-sim index (no torch needed)
# ---------------------------------------------------------------------------


class TestColPaliIndex:
    def test_save_and_load_round_trip(self, tmp_path):
        from benchmark.multimodal.colpali_embed import ColPaliIndex, MultiVectorChunk

        chunk = MultiVectorChunk(
            doc_id="doc_p1",
            page_no=1,
            source_name="doc.pdf",
            source_path="doc.pdf",
            token_vectors=[[0.1, 0.2], [0.3, 0.4]],
            page_text="hello",
        )
        index = ColPaliIndex(chunks=[chunk])
        path = tmp_path / "index.json"
        index.save(path)

        loaded = ColPaliIndex.load(path)
        assert len(loaded.chunks) == 1
        assert loaded.chunks[0].doc_id == "doc_p1"
        assert loaded.chunks[0].token_vectors == [[0.1, 0.2], [0.3, 0.4]]

    def test_retrieve_empty_index(self, monkeypatch):
        # Stub torch so the import gate inside retrieve passes.
        _install_fake_torch(monkeypatch)

        from benchmark.multimodal.colpali_embed import ColPaliIndex

        index = ColPaliIndex(chunks=[])
        assert index.retrieve([[0.1, 0.2]], top_k=5) == []

    def test_retrieve_ranks_by_max_sim(self, monkeypatch):
        """Two chunks; the query is more similar to chunk B than chunk A."""
        _install_fake_torch(monkeypatch)

        from benchmark.multimodal.colpali_embed import ColPaliIndex, MultiVectorChunk

        # Each chunk has 1 token vector of dim 2. Query is close to B, far from A.
        chunk_a = MultiVectorChunk(
            doc_id="a_p1", page_no=1, source_name="a.pdf", source_path="a.pdf",
            token_vectors=[[0.0, 1.0]],
        )
        chunk_b = MultiVectorChunk(
            doc_id="b_p1", page_no=1, source_name="b.pdf", source_path="b.pdf",
            token_vectors=[[1.0, 0.0]],
        )
        index = ColPaliIndex(chunks=[chunk_a, chunk_b])

        results = index.retrieve([[1.0, 0.0]], top_k=2)
        assert len(results) == 2
        top_chunk, top_score = results[0]
        assert top_chunk.doc_id == "b_p1"
        assert top_score > results[1][1]


class _ListTensor:
    """Minimal torch.Tensor stand-in supporting the ops retrieve() uses.

    Implements: numel(), t(), matmul(), max(dim=1).values, sum().item(),
    tolist(), dtype, cpu().
    """

    def __init__(self, data):
        # data is list[list[float]] (2D).
        self.data = data

    @property
    def dtype(self):
        return "float32"

    def cpu(self):
        return self

    def tolist(self):
        return self.data

    def numel(self):
        return sum(len(row) for row in self.data) if self.data else 0

    def t(self):
        if not self.data:
            return _ListTensor([])
        rows = len(self.data)
        cols = len(self.data[0]) if rows else 0
        transposed = [[self.data[r][c] for r in range(rows)] for c in range(cols)]
        return _ListTensor(transposed)

    def matmul(self, other):
        a, b = self.data, other.data
        rows = len(a)
        inner = len(a[0]) if rows else 0
        cols = len(b[0])
        out = [
            [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
            for i in range(rows)
        ]
        return _ListTensor(out)

    def max(self, dim=None):
        if dim is None:
            flat = [v for row in self.data for v in row]
            return types.SimpleNamespace(values=max(flat))
        # max along the column axis (dim=1) — values is a 1-D tensor of row maxes
        out = [max(row) for row in self.data]
        return types.SimpleNamespace(values=_ListTensor([out]))

    def sum(self):
        flat = [v for row in self.data for v in row]
        return types.SimpleNamespace(item=lambda: sum(flat))


def _install_fake_torch(monkeypatch):
    """Register a fake ``torch`` module backed by :class:`_ListTensor`."""
    fake_torch = types.ModuleType("torch")
    fake_torch.Tensor = _ListTensor
    fake_torch.float32 = "float32"
    fake_torch.tensor = lambda x, dtype=None: (
        x if isinstance(x, _ListTensor) else _ListTensor(x)
    )
    fake_torch.matmul = lambda a, b: a.matmul(b)
    fake_torch.no_grad = lambda: MagicMock()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)


# ---------------------------------------------------------------------------
# Chunking integration — known strategies + multimodal dispatch
# ---------------------------------------------------------------------------


class TestChunkingIntegration:
    def test_known_chunking_strategies_includes_multimodal(self):
        from benchmark.chunking import known_chunking_strategies

        strategies = known_chunking_strategies()
        assert "recursive" in strategies
        assert "semantic" in strategies
        assert "pdf_ocr_docling" in strategies
        assert "audio_whisper" in strategies

    def test_get_chunker_rejects_multimodal_with_clear_error(self):
        from benchmark.chunking import get_chunker

        with pytest.raises(ValueError, match="multi-modal ingestion path"):
            get_chunker("pdf_ocr_docling", 1000, 200)

    def test_get_chunker_unknown_lists_multimodal(self):
        from benchmark.chunking import get_chunker

        with pytest.raises(ValueError, match="pdf_ocr_docling"):
            get_chunker("not_a_strategy", 1000, 200)

    def test_run_multimodal_ingestion_dispatches(self, tmp_path, monkeypatch):
        # Register a fake multimodal chunker under a test-only name.
        from benchmark.multimodal.registry import register_multimodal_chunker

        def _fake(corpus_path, **kwargs):
            return [{"context": "fake", "metadata": {"corpus_type": "test"}}]

        register_multimodal_chunker("__test_mm__", _fake)
        try:
            from benchmark.chunking import run_multimodal_ingestion

            result = run_multimodal_ingestion(
                "__test_mm__", corpus_path=str(tmp_path), settings={"k": 1}
            )
            assert result == [{"context": "fake", "metadata": {"corpus_type": "test"}}]
        finally:
            from benchmark.multimodal.registry import MULTIMODAL_CHUNKERS

            MULTIMODAL_CHUNKERS.pop("__test_mm__", None)

    def test_run_multimodal_ingestion_unknown_strategy(self):
        from benchmark.chunking import run_multimodal_ingestion

        with pytest.raises(ValueError, match="not a registered multi-modal chunker"):
            run_multimodal_ingestion("nope", corpus_path=".")


# ---------------------------------------------------------------------------
# Config integration — multimodal strategies accept None chunk_size/overlap
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Config-validation behaviour for multi-modal strategies.

    These tests construct BenchmarkConfig directly (bypassing get_env_combinations,
    which imports optional deps not present in every test environment). They
    exercise only the validate_benchmark_config() code path.
    """

    def _make_config(self, **overrides):
        from dataclasses import replace

        base = _minimal_base_config()
        if overrides:
            return replace(base, **overrides)
        return base

    def test_validate_accepts_multimodal_strategy_without_size_overlap(self):
        from config import validate_benchmark_config

        cfg = self._make_config(
            chunking_strategy="pdf_ocr_docling",
            chunk_size=None,
            chunk_overlap=None,
            corpus_type="pdf",
            multimodal_backend="docling",
        )
        validated = validate_benchmark_config(cfg)
        assert validated.chunking_strategy == "pdf_ocr_docling"

    def test_validate_rejects_corpus_type_mismatch(self):
        from config import validate_benchmark_config

        cfg = self._make_config(
            chunking_strategy="audio_whisper",
            chunk_size=None,
            chunk_overlap=None,
            corpus_type="pdf",  # mismatched
        )
        with pytest.raises(ValueError, match="requires corpus_type='audio'"):
            validate_benchmark_config(cfg)

    def test_validate_rejects_invalid_corpus_type(self):
        from config import validate_benchmark_config

        cfg = self._make_config(corpus_type="video")
        with pytest.raises(ValueError, match="corpus_type must be one of"):
            validate_benchmark_config(cfg)


# ---------------------------------------------------------------------------
# YAML experiment spec — chunking: block is honoured
# ---------------------------------------------------------------------------


class TestYAMLExperimentSpec:
    def test_chunking_block_applies_strategy(self, tmp_path):
        from benchmark.orchestration.matrix import (
            load_experiment_spec,
            build_configs_from_spec,
        )
        import yaml as _yaml  # only used to write; PyYAML is a project dep

        spec_path = tmp_path / "spec.yaml"
        spec_path.write_text(
            _yaml.safe_dump(
                {
                    "experiment_name": "mm-test",
                    "dataset": {
                        "name": "jsonl",
                        "path": "./questions.jsonl",
                        "corpus_path": "./pdfs",
                        "corpus_type": "pdf",
                    },
                    "chunking": {"strategy": "pdf_ocr_docling"},
                    "settings": {
                        "multimodal_backend": "docling",
                        "ragas_enabled": False,
                        "custom_metrics_enabled": False,
                    },
                }
            )
        )

        spec = load_experiment_spec(spec_path)
        assert spec.chunking == {"strategy": "pdf_ocr_docling"}

        # Build a base config directly to avoid pulling optional deps through
        # get_env_combinations (which imports benchmark.retrieval).
        base_cfg = _minimal_base_config()
        configs = build_configs_from_spec(spec, base_configs=[base_cfg])
        assert len(configs) >= 1
        cfg = configs[0]
        assert cfg.chunking_strategy == "pdf_ocr_docling"
        assert cfg.corpus_type == "pdf"
        assert cfg.chunk_size is None
        assert cfg.chunk_overlap is None


def _minimal_base_config():
    """Construct a minimal valid BenchmarkConfig without going through env load."""
    from config import BenchmarkConfig

    return BenchmarkConfig(
        llm_model="gpt-4o-mini",
        llm_provider="openai",
        embedding_model="nomic-embed-text:latest",
        chunk_size=1000,
        chunk_overlap=200,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        max_new_tokens=256,
        ollama_base_url="http://localhost:11434",
        ollama_api_key=None,
        openai_compat_base_url=None,
        openai_compat_api_key=None,
        llm_ollama_base_url=None,
        llm_ollama_api_key=None,
        llm_openai_compat_base_url=None,
        llm_openai_compat_api_key=None,
        eval_critic_ollama_base_url=None,
        eval_critic_ollama_api_key=None,
        eval_critic_openai_compat_base_url=None,
        eval_critic_openai_compat_api_key=None,
        embedding_ollama_base_url=None,
        embedding_ollama_api_key=None,
        eval_critic_max_tokens=4096,
        dataset_name="jsonl",
        dataset_subset="",
        dataset_sample_size=5,
        eval_critic_llm="gpt-4o-mini",
        eval_critic_embedding="nomic-embed-text:latest",
    )
