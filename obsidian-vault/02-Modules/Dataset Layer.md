# Dataset Layer

Sources:

- [benchmark/dataset.py](../benchmark/dataset.py)
- [benchmark/dataset_adapters.py](../benchmark/dataset_adapters.py)

The dataset layer normalizes built-in Hugging Face adapters, arbitrary Hugging
Face datasets, JSONL files, CSV files, and split corpus/question JSONL files
into records consumed by the rest of the framework:

```text
{
  "id": ...,
  "question": ...,
  "ground_truth": ...,
  "context": ...,
  "metadata": ...
}
```

Canonical types in `benchmark.dataset`:

- `BenchmarkSample`: question, optional ground truth, optional context, optional
  relevant context IDs, and metadata.
- `CorpusDocument`: searchable document text with optional ID and metadata.
- `DatasetMapping`: maps arbitrary source columns into the canonical schema.

Custom source loaders:

- `samples_from_records()` and `corpus_from_records()` normalize in-memory rows.
- `load_jsonl_dataset()` reads one JSON object per line.
- `load_csv_dataset()` reads CSV rows with a mapping.
- `load_huggingface_dataset()` loads any HF dataset when mapped.
- `load_corpus_and_questions_from_jsonl()` loads separate corpus and question
  files for retrieval benchmarks with optional `relevant_context_ids`.
- `load_dataset_for_config()` is the single config-driven entrypoint used by
  `main.py`.

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

Custom data pattern:

1. Prefer JSONL for durable eval sets.
2. Use `DATASET_SOURCE=jsonl` or `DATASET_SOURCE=csv` with `DATASET_PATH`.
3. Use `DATASET_MAPPING` JSON when columns differ from the canonical names.
4. Use `DATASET_SOURCE=corpus_jsonl` with `DATASET_CORPUS_PATH` and
   `DATASET_QUESTIONS_PATH` when the searchable corpus is separate from the
   benchmark questions.

Related notes:

- [[Benchmark Pipeline]]
- [[Evaluation and Metrics]]
