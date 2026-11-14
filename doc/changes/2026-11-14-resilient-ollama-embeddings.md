# Resilient Ollama embeddings: guard against embedding-runner crashes

Date: 2026-11-14

## Summary

Ollama's internal `llama-server` embedding runner crashes (`connection reset
by peer` on its internal `/tokenize` port, surfaced as a 400 `ResponseError`)
when it receives an empty string, and can trip on oversized inputs. This
killed both `semantic_bppercentile95` configs of run7 twice (Nov 13 and Nov
14) while all recursive configs passed — the langchain `SemanticChunker`
sends sentence-level embed calls that can include empty/whitespace-only
strings. `benchmark/embedding.py` now wraps `OllamaEmbeddings` with input
sanitization and bounded retries.

## Changes

- `benchmark/embedding.py`:
  - New `ResilientOllamaEmbeddings` (subclass of `OllamaEmbeddings`, built
    lazily via `_ollama_embeddings_class()` so `isinstance` checks elsewhere
    stay valid):
    - `_sanitize()`: empty/whitespace-only strings become a single-space
      placeholder; texts are capped at 6000 chars (safe margin below
      nomic-embed-text's 2048-token context).
    - `_embed_with_retry()`: retries `ollama.ResponseError` and
      `httpx.HTTPError` up to 5 times with exponential backoff (2s base),
      surviving transient runner restarts; re-raises after exhaustion.
  - `get_embedding_model()` returns the resilient class for the `ollama`
    provider; API unchanged otherwise.
- `tests/test_embedding.py`: `TestResilientOllamaEmbeddings` covers
  sanitization (empty, whitespace, truncation, passthrough), batch/query
  sanitization, retry-on-connection-reset, retry-on-httpx error, and
  exhaustion raising.

## Behaviour changes and migration

- No manifest/config changes. Empty corpus strings now embed as `" "` (a
  near-zero vector) instead of crashing the run; texts longer than 6000
  chars are embedded truncated. Recursive/semantic chunk outputs are far
  below the cap, so results for healthy configs are unchanged.
- Note: the random port in past error messages (42597, 45887) is Ollama's
  internal per-model `llama-server` port, **not** the worker's
  `OLLAMA_BASE_URL` — the worker always talks to the server port (11434).

## Verification

- `pytest tests/test_embedding.py tests/test_adapter_component_injection.py`
  → 31 passed; `tests/test_config.py` → 125 passed (local).
- Same suite on spark (`141.39.193.218`) → 17 passed after syncing the two
  files. The two failed semantic configs of `results/Enterprise-RAG/run7`
  can be retried by resuming the worker (`--run-dir` resume retries
  `failed` entries) once the current k12 pass finishes.
