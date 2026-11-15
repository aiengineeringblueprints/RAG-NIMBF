# PR #33 follow-up documentation

Date: 2026-11-15

## Summary
Brings the durable guides in line with the code merged in PR #33 (resilient
Ollama embeddings, final gold-doc nDCG fix). The run7 results section still
described the interim nDCG fix and quoted values computed with it, and the new
`EMBEDDING_MAX_TEXT_CHARS` variable was only documented in `.env.example`.

## Changes
- `doc/enterprise-rag-blueprint-results.md` (run7 section): the nDCG@10
  column is now labelled as raw and invalid. The fix description now matches
  the merged code: original ranks are kept, each gold document is credited
  once, and chunks without a doc id stay in place. The interim recomputation
  (0.711–0.739) is withdrawn because it used dedupe-then-renumber, which
  inflates scores. A re-score on the DGX Spark is still pending. Also notes
  that hit@k / recall@k only change if run7 chunks lacked doc ids.
- `doc/env-vars.md`: new **Embeddings** section with `EMBEDDING_MODELS`,
  `EMBEDDING_OLLAMA_BASE_URL`, `EMBEDDING_OLLAMA_API_KEY` (all three were
  read by `config.py` but not documented before) and `EMBEDDING_MAX_TEXT_CHARS`,
  plus a short note on the embedding hardening.
- `doc/enterprise-rag-blueprint-setup.md`: two troubleshooting rows: the
  embedding-runner crash (`connection reset by peer` / 500 while embedding)
  and the truncation warning.

## Behaviour changes and migration
None (documentation only).

## Verification
Documentation only; no code changed. Checked variable names and defaults
against `benchmark/embedding.py` and `config.py`.

## Follow-ups
- Re-score run7 nDCG@10 (and check for missing doc ids) on the DGX Spark, then
  replace the invalid column in `doc/enterprise-rag-blueprint-results.md`.
- Issue #34: promote `EMBEDDING_MAX_TEXT_CHARS` to a `BenchmarkConfig` field.
- The docstring of `_rebuild_resilient_ollama` and the comment in
  `ResilientOllamaEmbeddings.__reduce__` in `benchmark/embedding.py` still say
  the factory registers the class in the module; that registration was
  removed. Pickling works because pickle calls the module-level function.
