# G — Multi-modal RAG Ingestion (Tier-3 Prototype)

Reference: **RAGPerf §3.3.1 / §4.1** (arXiv:2603.10765v1).
This note documents the multi-modal ingestion paths added to the benchmarking
framework: what's wired, what's stubbed, how to install the optional deps,
and how this integrates with the rest of the system (TRACe / workload agents
live in separate worktrees).

## TL;DR

| Path | Strategy name | Status | Optional dep |
|------|---------------|--------|--------------|
| PDF OCR | `pdf_ocr_docling` | **Working** (with `docling` installed) | `pip install docling` |
| Image OCR | `image_ocr_docling` | **Working** (with `docling` installed) | `pip install docling` |
| Audio ASR | `audio_whisper` | **Working** (with `faster-whisper` or `openai-whisper`) | `pip install faster-whisper` |
| PDF/Image Visual | `colpali_visual` (embedder) | **Interface complete; backend stubbed** — `build_colpali_visual_embedder` raises `OptionalDependencyError` until `colpali-engine` is installed. Retrieval math (`ColPaliIndex.retrieve`) is fully implemented and unit-tested with synthetic embeddings. | `pip install colpali-engine torch pdf2image` |

The text pipeline is **unchanged**. Multi-modal chunkers produce ordinary
`{context, metadata}` dicts that flow through the existing chunk → embed →
retrieve → generate path. ColPali is the exception: it stores multi-vector
embeddings in a dedicated JSON-blob index, not the Chroma text collection.

## Architecture

```
                      ┌─────────────────────────────────────────────┐
                      │  corpus_path (PDF / image / audio dir)      │
                      └─────────────────────┬───────────────────────┘
                                            │
                ┌───────────────────────────┼───────────────────────────┐
                │                           │                           │
       pdf_ocr_docling /            audio_whisper                colpali_visual
       image_ocr_docling            (Whisper ASR)                  (embedder)
                │                           │                           │
       ┌────────▼────────┐         ┌────────▼────────┐         ┌────────▼────────┐
       │ Docling         │         │ faster-whisper  │         │ ColQwen2 model  │
       │ DocumentConverter│        │ or openai-whisper│        │ multi-vector    │
       │ → per-page text │         │ → transcript    │         │ embeddings      │
       └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
                │                           │                           │
                │   {context, metadata}     │                           │ {token_vectors}
                │   (one per page/file)     │                           │ (one per page)
                ▼                           ▼                           ▼
       ┌───────────────────────────────────────────┐         ┌──────────────────┐
       │ Markdown / Recursive text splitter        │         │ ColPaliIndex     │
       │ (re-chunk to target chunk_size)           │         │ (JSON blob +     │
       └────────────────────┬──────────────────────┘         │  ColBERT max-sim)│
                            │                                └────────┬─────────┘
                            ▼                                         │
                    ┌─────────────────────────────────┐               │
                    │  Standard text pipeline:        │               │
                    │  embed (Ollama/HF) → Chroma →   │               │
                    │  retrieve → rerank → generate   │               │
                    └─────────────────────────────────┘               │
                                                                      │
                            (visual retrieval bypasses Chroma) ◄──────┘
```

Key design decisions:

1. **Lazy imports everywhere.** The `benchmark.multimodal` package can be
   imported with zero heavy ML deps installed. Each factory function
   (`ingest_pdf_via_docling`, `ingest_audio_via_whisper`,
   `build_colpali_visual_embedder`) imports its backend on first call and
   raises `OptionalDependencyError` with a `pip install ...` hint otherwise.

2. **Multi-modal is a corpus-loader concern, not a chunker concern.** The
   text pipeline's `get_chunker` raises a clear `ValueError` if handed a
   multi-modal strategy name. Dispatch happens in `main.py`'s
   `_build_internal_retrieval_index`, which calls `run_multimodal_ingestion`
   to produce corpus dicts and then re-chunks them with a text splitter.

3. **ColPali stores outside Chroma.** ColPali's per-page multi-vector
   embeddings don't fit Chroma's single-vector schema. They live in a
   `ColPaliIndex` (JSON file at `.colpali_index.json` by default) and are
   retrieved via brute-force ColBERT max-sim. This is **prototype-scale**
   (hundreds to low thousands of pages). For production, swap in Vespa /
   Milvus / Qdrant multi-vector behind the `ColPaliIndex` interface.

4. **YAML gets a first-class `chunking:` block.** Previously only the
   `matrix:` block let you pick a chunking strategy. The new top-level
   `chunking: { strategy: ..., size: ..., overlap: ... }` makes single-config
   multi-modal experiments readable. See `experiments/multimodal-pdf-smoke.yaml`.

## Optional-dependency install

Install only what you need for the path you're benchmarking:

```bash
# PDF / image OCR path
pip install docling

# Audio ASR path (pick one)
pip install faster-whisper        # recommended (CTranslate2, lower memory)
# OR
pip install openai-whisper

# ColPali visual path (heaviest — pulls torch + transformers + ColQwen2 weights)
pip install colpali-engine torch pdf2image
# pdf2image requires poppler-system-utils at the OS level; alternatively
# install PyMuPDF (fitz) and ColPali will fall back to it for PDF rasterisation.
```

Verify install:

```bash
python -c "from benchmark.multimodal import available_multimodal_chunkers; \
           print(available_multimodal_chunkers())"
# ('audio_whisper', 'image_ocr_docling', 'pdf_ocr_docling')
```

## YAML examples

Three smoke-test manifests ship in `experiments/`:

- `multimodal-pdf-smoke.yaml` — Docling OCR path
- `multimodal-image-smoke.yaml` — Docling OCR path (images)
- `multimodal-audio-smoke.yaml` — Whisper ASR path

Run one:

```bash
BENCHMARK_CONFIG_FILE=experiments/multimodal-pdf-smoke.yaml python main.py
```

Each manifest expects:
- `dataset.path` — a `questions.jsonl` with the QA pairs (standard text dataset format)
- `dataset.corpus_path` — directory of source files (PDFs / images / audio)
- `dataset.corpus_type` — `pdf` | `image` | `audio`
- `chunking.strategy` — the multi-modal chunker name
- `settings.multimodal_backend` — `docling` | `faster-whisper` | `openai-whisper` | `colpali`

## What's working vs stubbed

### Working (end-to-end, with deps installed)
- Registry / dispatch / config validation for all three chunker strategies
- Docling PDF OCR (verified via mocked-backend unit tests)
- Docling image OCR (verified via mocked-backend unit tests)
- Whisper ASR with both `faster-whisper` and `openai-whisper` backends
  (verified via mocked-backend unit tests)
- ColPali `MultiVectorChunk` JSON (de)serialisation
- ColPali `ColPaliIndex.retrieve` ColBERT max-sim scoring (unit-tested with
  synthetic embeddings and a torch stand-in; real torch path is identical
  but unverified in CI due to torch's install weight)

### Stubbed / interface-only
- `ColPaliVisualEmbedder.embed_documents` for **live** PDFs/images — the
  interface is complete and matches the current `colpali-engine` ColQwen2
  API (`ColQwen2.from_pretrained`, `ColQwen2Processor.process_images` /
  `process_queries` / `score_multi_vector`). It will work once
  `colpali-engine` + `torch` are installed, but we have not run it against
  a real model download in CI. Treat the first end-to-end ColPali run as
  the integration smoke test.
- **Main-pipeline ColPali retrieval**: when `embedding_model: colpali_visual`,
  `main.py` does NOT yet route retrieval to `ColPaliIndex` — it still tries
  to build a Chroma collection. Wiring this is deliberately deferred because
  it requires a separate retrieve path (no Chroma), which in turn requires
  changes to `_cache_key`, `_run_single_benchmark_impl`, and the retrieval
  metric scorers. For this Tier-3 prototype the **OCR path is the
  recommended path**; ColPali retrieval is interface-ready for the next tier.

### Unsupported combinations (will raise clear errors)
- `chunking_strategy: pdf_ocr_docling` + `corpus_type: audio` → ValueError
  at config validation (`requires corpus_type='pdf'`)
- `chunking_strategy: audio_whisper` + missing audio files → ValueError
  ("No audio documents found")
- Any multi-modal strategy without a `corpus_path` → ValueError at runtime
- `embedding_model: colpali_visual` without `colpali-engine` →
  `OptionalDependencyError` at first embed call

## Integration points (separate worktrees)

This work is one of several parallel RAGPerf-aligned worktrees. Hand-off
notes for the others:

- **TRACe eval harness** (separate worktree): TRACe consumes
  `(retrieved_chunks, generated_answer, ground_truth)` tuples. The
  multi-modal path produces these in the same shape as the text pipeline,
  so TRACe needs **no changes** to score multi-modal runs. The only
  multi-modal-specific signal TRACe may want is `metadata.corpus_type`
  for slicing reports by ingestion path — already populated in chunk
  metadata by every multi-modal factory.

- **Workload agents** (separate worktree): agents that drive the benchmark
  (e.g. concurrent-stress harness) treat `main.py` as a black box. The
  multi-modal dispatch is transparent to them. The only new failure mode
  is `OptionalDependencyError` — agents should surface the install hint
  verbatim rather than retry.

- **Eval matrix** (`EVAL_MATRIX.md`): a new column for `corpus_type`
  (text / pdf / image / audio) should be added when this lands on main.
  The current matrix assumes text-only.

## Current limitations

1. **ColPali is brute-force.** `ColPaliIndex.retrieve` is O(q·d·n) per query.
   Fine for prototype corpora (≤ a few thousand pages). Swap for a real
   multi-vector ANN before benchmarking at scale.
2. **No mixed-corpus support.** A single experiment config targets one
   `corpus_type`. Benchmarking a joint text+PDF corpus requires two configs
   and post-hoc report merging.
3. **Docling image OCR fallback.** Raw image OCR relies on Docling's
   image-format support; if that's absent, the wrapper logs a warning and
   skips the file rather than failing hard. Production code should call
   `docling_core` directly.
4. **Whisper backend selection.** We default to `faster-whisper` + `cpu` +
   `int8` for installability. Switch to `cuda` + `float16` + `large-v3`
   for production-quality ASR.
5. **No GPU-aware defaults.** All device/compute-type settings are explicit
   in `BenchmarkConfig` so users must opt into GPU. The framework never
   silently grabs a GPU.
6. **ColPali model weights** download from HuggingFace Hub on first use
   (~2 GB for `vidore/colqwen2-v1.0`). The first run will be slow; cache
   the weights in CI for reproducibility.

## Test status

```
tests/test_multimodal.py — 27 passed
```

Covers:
- Registry contents and dispatch
- `OptionalDependencyError` raises + install hints
- Docling PDF walk (mocked backend) + per-page chunk shape
- Whisper ASR with both backends (mocked)
- ColPali `ColPaliIndex` save/load round-trip
- ColPali `ColPaliIndex.retrieve` max-sim ranking (synthetic embeddings,
  torch stand-in — verifies the math without requiring torch)
- `known_chunking_strategies` includes multi-modal names
- `get_chunker` rejects multi-modal names with a clear error
- `run_multimodal_ingestion` dispatch + unknown-strategy error
- Config validation: multi-modal accepts None chunk_size/overlap, rejects
  corpus_type mismatches, rejects invalid corpus_type values
- YAML `chunking:` block flows through `build_configs_from_spec`

Heavy deps (`docling`, `colpali-engine`, `faster-whisper`, `openai-whisper`,
`torch`) are **mocked throughout** — tests run on a base install.

## Files

New:
- `benchmark/multimodal/__init__.py`
- `benchmark/multimodal/registry.py`
- `benchmark/multimodal/docling_ocr.py`
- `benchmark/multimodal/whisper_asr.py`
- `benchmark/multimodal/colpali_embed.py`
- `experiments/multimodal-pdf-smoke.yaml`
- `experiments/multimodal-image-smoke.yaml`
- `experiments/multimodal-audio-smoke.yaml`
- `tests/test_multimodal.py`
- `NOTES_G_Multimodal.md` (this file)

Modified (small, surgical edits):
- `benchmark/chunking.py` — multimodal-name surfacing + dispatch helper
- `benchmark/embedding.py` — multimodal-name surfacing
- `benchmark/orchestration/matrix.py` — `chunking:` block + corpus_type
- `config.py` — `corpus_type` / `multimodal_backend` / Whisper / ColPali
  fields, multimodal-aware validation
- `main.py` — `_build_internal_retrieval_index` dispatches to
  `run_multimodal_ingestion` for multi-modal strategies
