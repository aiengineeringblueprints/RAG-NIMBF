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
  lowercase Unicode word tokens with whitespace collapsed and compares
  boundary-padded word sequences **within each chunk** (no false positives
  at word edges: "Yes." never matches "yesterday", "The rapist …" never
  matches "Therapist …"). Label-like sentences — fewer than 3 normalised
  words AND fewer than 20 compact characters ("Yes.", "Results.",
  "Table 1.", "Yes it is.") — are excluded from scoring entirely: they
  would match almost any chunk, while real short content sentences
  ("Revenue grew 5%.") stay scoreable.
  Sentences split across a chunk boundary — even mid-word, across
  overlapping chunks, or with the fragments retrieved at non-adjacent ranks
  in either order — are repaired by a direct two-chunk reconstruction check
  (`_split_repair_match`): one chunk must end with a proper prefix of the
  sentence while a different chunk starts with the corresponding suffix
  (`q <= p` allows overlapping fragments), with both fragments at least 4
  compact characters so tiny fragments cannot combine arbitrary chunks. No
  per-pair join strings are built, so the cost is independent of k².
  Duplicate keys and key variants ("3a", "3a.", repeated keys) are
  deduplicated before scoring; empty, label-like and malformed labels
  (`None`, nested lists, JSON strings) never reach the score.
- `benchmark/dataset_adapters.py`: the `ragbench` adapter now carries
  `all_relevant_sentence_keys`, `documents_sentences` and `adherence_score`
  as sample metadata; `derive_answerable_flag` sets
  `metadata["answerable"]` (parsing JSON-string values, treating blank
  CSV cells as empty, applied to the normalised copy — the caller's dict is
  not mutated); `strip_sentence_label_metadata` keeps the heavy label
  fields out of corpus and chunk metadata.
- `benchmark/chunking.py`: `chunk_documents` strips the sentence-label
  fields at the chunking boundary, so no load path (shared corpus or
  per-sample like `ragbench`) leaks the full document text into the vector
  store or `retrieval_metadata`.
- `benchmark/dataset.py`: `_chroma_safe_metadata` drops empty lists and
  `None`-only lists, removes `None` entries from partially-null lists and
  JSON-serialises genuinely mixed-type lists (Chroma 1.5.9 rejects all of
  these; dropping instead of stringifying keeps one type per key across a
  collection — `meta.get(key)` returns `None` for dropped keys);
  `normalize_sample` derives the `answerable` flag on the copy for every
  load path.
- `benchmark/orchestration/runner.py`: `_with_relevant_sentence_recall`
  merges the metric into the custom-metrics result whenever the labels are
  present — computed on all metric paths including
  `CUSTOM_METRICS_ENABLED=false`. Datasets without sentence labels skip the
  metric silently. The samples handed to adapters and checkpoints have the
  label fields stripped, so the system under test never sees the sentence
  labels (the separate, pre-existing `ground_truth` exposure of the HTTP
  adapter is tracked in issue #39). A shared
  `_merge_custom_metric_results` helper backs both this merge and the
  gold-doc retrieval metrics merge.
- `tests/test_relevant_sentence_recall.py`: 43 tests for label resolution,
  matching robustness (case/whitespace, chunk-boundary/overlapping/reverse-
  order splits, word-boundary false positives, label-like short sentences
  incl. all reviewer repros, short content sentences staying scoreable,
  empty/Unicode sentences, malformed and JSON-string labels, blank CSV
  cells, duplicate keys), exclusion rules, adapter metadata,
  corpus/chunk-metadata filtering, Chroma-safe list handling and runner
  wiring.

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
| Answered | 313 | 43.6 % | 48 % |
| Refused | 139 | 17.4 % | 19 % |

(Differences come from the refusal-detection heuristic used for the split
and from label-like sentences ("Yes.", "Results.", short headings) being
excluded from scoring entirely.)
Full test suite: 903 passed.

## Migration

No action needed. Existing runs are unaffected; new runs on RAGBench-based
datasets get the metric automatically, other datasets unchanged.
