# README documentation gaps

Date: 2026-11-12

## Summary
The README trailed the code: multi-hop retrieval, HyDE and several retrieval,
generation, reranker, and vector-store knobs existed in `config.py` but were
absent from the env-var table, and the new LLM answer judge had no user-facing
README section.

## Changes
README.md only (no code changes):
- Added retrieval rows: `RETRIEVAL_TOP_K`, `RETRIEVAL_FETCH_K`,
  `RETRIEVAL_CANDIDATE_K`, `RETRIEVAL_VECTOR_SIMILARITY_WEIGHT`,
  `RETRIEVAL_MMR_LAMBDA`, `RETRIEVAL_SIMILARITY_THRESHOLD`,
  `RETRIEVAL_USE_HYDE`, `RETRIEVAL_MULTIHOP`, `RETRIEVAL_MULTIHOP_ROUNDS`.
- Added pipeline rows: `VECTOR_DB_BACKEND`, `LANCEDB_PATH`,
  `GENERATION_TEMPERATURE`, `GENERATION_TOP_P`, `MAX_NEW_TOKENS`,
  `RERANKER_MODELS`, `RERANKER_TOP_K`, `LLM_ANSWER_STRIP_MODE`,
  `LLM_ANSWER_VALUE_FALLBACK`.
- Added the section "Re-Score Finished Runs With the LLM Answer Judge"
  documenting `scripts/judge_answers.py`, its cache/resume behavior, and the
  `EVAL_CRITIC_LLM`/`EVAL_CRITIC_MAX_WORKERS` usage.

## Migration
None. Documentation only.

## Verification
Manual cross-check of `os.getenv` calls in `config.py`, `benchmark/`, and
`benchmark/adapters/` against the README env-var table.

## Addendum (same session)
- Added a "Publication" section at the bottom of the README linking the
  RAG-NIMBF paper via DOI (https://doi.org/10.48550/arXiv.2610.27816).
