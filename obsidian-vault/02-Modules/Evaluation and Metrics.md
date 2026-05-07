# Evaluation and Metrics

Sources:

- [benchmark/evaluation.py](../benchmark/evaluation.py)
- [benchmark/custom_metrics.py](../benchmark/custom_metrics.py)
- [benchmark/metrics.py](../benchmark/metrics.py)
- [benchmark/reporting/models.py](../benchmark/reporting/models.py)

RAGAS metrics represented in result models:

- `faithfulness`
- `answer_relevancy`
- `answer_correctness`
- `context_precision`
- `context_recall`
- `semantic_similarity`

Current `evaluate_results()` default metric set should be verified in source before making claims; recent inspection found the active default focused on faithfulness, context recall, and semantic similarity, while result models still have fields for the broader set.

`evaluate_results()` wraps the RAGAS call and returns `EvaluationResult` with metric means, per-sample scores, error state, and valid sample counts.

Custom metrics:

- Retrieval-style metrics: hit@k, nDCG@k, recall@k.
- Context relevance.
- Answer lexical metrics: ROUGE-L, BLEU, METEOR.
- BERTScore precision, recall, F1 when enabled.
- Refusal detection and lightweight relevance determination.

Custom metrics are wired into `main.py` after RAGAS evaluation. Older audit notes that say custom metrics are not integrated are stale.

Result structures:

- `PerSampleResult`
- `StatSummary`
- `BenchmarkResultExtended`
- `BenchmarkRun`

Performance metrics:

- Time to first token.
- Tokens per second.
- Optional GPU utilization and memory usage.

Related notes:

- [[Benchmark Pipeline]]
- [[Reporting and Tracking]]
- [[Research Notes]]
- [[Known Stale Docs]]
