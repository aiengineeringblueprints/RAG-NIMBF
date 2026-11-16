# New metric: `relevant_sentence_recall` from RAGBench sentence labels

Date: 2026-11-16 · Issue: [#17](https://github.com/aiengineeringblueprints/RAG-NIMBF/issues/17)

## Summary

New retrieval metric `relevant_sentence_recall`: the share of the sentences
RAGBench labels as relevant (`all_relevant_sentence_keys`, pointing into
`documents_sentences`) that appear in the contexts a RAG system retrieved. It
measures passage-level retrieval, which is the main cause of refusals on
TechQA (see `doc/enterprise-rag-techqa-refusal-analysis.md`, section 2.2).

## Changes

- `benchmark/gold_retrieval_metrics.py`: new `resolve_relevant_sentences` and
  `compute_relevant_sentence_recall`. Primary matching normalises text to
  lowercase Unicode word tokens with whitespace collapsed, preserving word
  boundaries (no false positives at word edges) and is applied per chunk and
  across the space-joined chunks. Sentences split across a chunk boundary —
  even mid-word or with overlapping chunks — are repaired via compact
  (boundary-free) concatenations of adjacent chunk pairs, with and without
  longest-overlap trimming. Empty or label-only sentences never match;
  malformed label values (`None`, nested lists) are ignored instead of being
  stringified into scored sentences.
- `benchmark/dataset_adapters.py`: the `ragbench` adapter now carries
  `all_relevant_sentence_keys`, `documents_sentences` and `adherence_score`
  as sample metadata; new `derive_answerable_flag` sets
  `metadata["answerable"]` (True iff the sample has at least one relevant
  sentence).
- `benchmark/dataset.py`: derives the `answerable` flag at load time.
- `benchmark/orchestration/runner.py`: `_with_relevant_sentence_recall`
  merges the metric into the custom-metrics result whenever the labels are
  present. Datasets without sentence labels skip the metric silently (it is
  simply absent from the output).
- `tests/test_relevant_sentence_recall.py`: 22 tests for label resolution,
  matching robustness (case/whitespace, chunk-boundary and overlapping-chunk
  splits, word-boundary false positives, empty/Unicode sentences, malformed
  labels), exclusion rules, adapter metadata and runner wiring.

## Semantics

- Score per sample: retrieved relevant sentences / total relevant sentences.
- Samples **without** any relevant sentence (unanswerable per RAGBench) and
  samples without the label fields get no score (excluded from the mean),
  not 0.
- Reported as `relevant_sentence_recall` in the result JSON, reports/CSVs and
  MLflow (`custom_relevant_sentence_recall`) via the existing generic
  custom-metrics plumbing.

## Verification

Recomputing on the run6 samples (`results/Enterprise-RAG/run6`, config
`recursive_cs512_co64_..._k10`) reproduces the manual numbers from
`doc/enterprise-rag-techqa-refusal-analysis.md` section 2.2 within a few
points:

| | n | Metric | Manual |
|---|---|---|---|
| Answered | 313 | 44.3 % | 48 % |
| Refused | 139 | 17.6 % | 19 % |

(Differences come from the refusal-detection heuristic used for the split.)
Full test suite: 882 passed.

## Migration

No action needed. Existing runs are unaffected; new runs on RAGBench-based
datasets get the metric automatically, other datasets unchanged.
