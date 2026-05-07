# Dataset Layer

Sources:

- [benchmark/dataset.py](../benchmark/dataset.py)
- [benchmark/dataset_adapters.py](../benchmark/dataset_adapters.py)

Dataset adapters normalize Hugging Face datasets into records consumed by the rest of the framework:

```text
{
  "id": ...,
  "question": ...,
  "ground_truth": ...,
  "context": ...,
  "metadata": ...
}
```

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
