# Dataset Layer

Sources:

- [benchmark/dataset.py](../benchmark/dataset.py)
- [benchmark/dataset_adapters.py](../benchmark/dataset_adapters.py)

Dataset adapters normalize Hugging Face datasets into records consumed by the rest of the framework:

```text
{
  "question": ...,
  "ground_truth": ...,
  "context": ...,
  "metadata": ...
}
```

Adapter registry and contracts:

- `benchmark/dataset_adapters.py` owns `REGISTRY`, `register()`, and `get_adapter()`. Config validation and dataset loading both use that registry, so a new dataset name must be registered before it can run.
- `DatasetAdapter` records the Hugging Face ID, question and ground-truth column names, context builder, preferred split, metadata keys, subset requirement, optional ground-truth transform, and whether the dataset has a shared corpus.
- `load_benchmark_data()` selects the adapter, loads the preferred split, applies deterministic sample shuffling, converts ground truth to text, builds context, and copies declared metadata keys only.
- `load_corpus_and_questions()` is the shared-corpus path. It deduplicates contexts for adapters with `has_shared_corpus` and adds stable `metadata.doc_id` / `metadata.gold_doc_id` pairs where labels are available.

Sample validation surface:

- `BenchmarkSample`, `normalize_sample()`, and `normalize_samples()` in `benchmark/dataset.py` define the typed/validated sample contract while preserving the public plain-dict flow. Every sample must expose `question`, `ground_truth`, `context`, and `metadata`.
- Normalization coerces question and ground truth to strings, keeps string context as text, converts list context items to strings, treats `metadata=None` as `{}`, and rejects non-dict metadata with a source-indexed error message.
- Unknown dataset names fail early through `get_adapter()` / config registry checks. Missing or malformed dataset columns fail while the adapter reads row keys or builds the normalized sample.
- Empty contexts are not globally rejected. This is intentional for datasets such as `ragperf-wikipedia-nq`, where questions are evaluated against a separate shared corpus and retrieval ground truth is unavailable.
- Retrieval-label validation is metadata-based: gold-document metrics use `metadata.gold_doc_id` when present and skip samples without it.

Built-in adapters:

- `t2-ragbench`: Hugging Face `G4KMU/t2-ragbench`, requires subset, default target is FinQA in config.
- `ragbench`: Hugging Face `rungalileo/ragbench`, requires subset.
- `squad`: Hugging Face `rajpurkar/squad`, validation split, shared corpus mode.
- `ragas-wikiqa`: Hugging Face `vibrantlabsai/ragas-wikiqa`, train split.
- `ragperf-wikipedia-nq`: RAGPerf-style setup with `wikimedia/wikipedia`
  (`20231101.en`) as shared corpus and `sentence-transformers/natural-questions`
  train split as questions/ground-truth answers.

Shared corpus behavior:

- Some datasets, currently `squad` and `ragperf-wikipedia-nq`, can deduplicate or load a shared corpus and benchmark questions separately.
- `main.py` chooses `load_corpus_and_questions()` when the adapter advertises `has_shared_corpus`.
- Deduplicated shared-corpus datasets assign stable `metadata.doc_id` values to corpus documents and matching `metadata.gold_doc_id` values to each question. For SQuAD, this turns the original context paragraph into the gold retrieval document.
- `ragperf-wikipedia-nq` has answer ground truth, but no gold Wikipedia document or chunk IDs. Retrieval precision/recall should be treated as proxy/judge-based rather than labeled retrieval evaluation.

Extension pattern:

1. Add a `DatasetAdapter`.
2. Implement a context builder and optional ground-truth transformer.
3. Register the adapter in `REGISTRY`.
4. Add or adjust tests in [[Testing and Coverage]].
5. Update this note and [[Configuration Reference]].

Related notes:

- [[Benchmark Pipeline]]
- [[Evaluation and Metrics]]
