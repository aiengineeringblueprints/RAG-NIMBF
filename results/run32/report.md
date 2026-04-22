# RAG Benchmark Report

**Date:** 20260422_161941
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 8

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs1000_co200_nomic... | 1.054 | 0.0 | 0.2 | 933.1 |
| semantic_cs1000_co200_nomic... | 1.232 | 0.0 | 0.4 | 1260.2 |
| semantic_cs1000_co100_nomic... | 0.954 | 0.0 | 0.5 | 908.0 |
| semantic_cs1000_co100_nomic... | 1.203 | 0.0 | 0.7 | 1239.1 |
| semantic_cs500_co200_nomic-... | 1.053 | 0.0 | 0.7 | 966.8 |
| semantic_cs500_co200_nomic-... | 1.345 | 0.0 | 0.1 | 1297.7 |
| semantic_cs500_co100_nomic-... | 0.937 | 0.0 | 0.4 | 931.2 |
| semantic_cs500_co100_nomic-... | 1.248 | 0.0 | 0.1 | 1248.6 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs1000_co200_nomic... | 0.858 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co200_nomic... | 0.873 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co100_nomic... | 0.858 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co100_nomic... | 0.867 | N/A | N/A | N/A | N/A |
| semantic_cs500_co200_nomic-... | 0.853 | N/A | N/A | N/A | N/A |
| semantic_cs500_co200_nomic-... | 0.870 | N/A | N/A | N/A | N/A |
| semantic_cs500_co100_nomic-... | 0.855 | N/A | N/A | N/A | N/A |
| semantic_cs500_co100_nomic-... | 0.871 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| semantic_cs1000_co200_nomic... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.324 | 0.573 | 0.064 | 0.602 | 0.909 | 0.919 | 0.914 |
| semantic_cs1000_co200_nomic... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.150 | 0.226 | 0.031 | 0.426 | 0.930 | 0.952 | 0.941 |
| semantic_cs1000_co100_nomic... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.324 | 0.573 | 0.064 | 0.602 | 0.909 | 0.919 | 0.914 |
| semantic_cs1000_co100_nomic... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.150 | 0.226 | 0.031 | 0.426 | 0.930 | 0.952 | 0.941 |
| semantic_cs500_co200_nomic-... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.324 | 0.573 | 0.064 | 0.602 | 0.909 | 0.919 | 0.914 |
| semantic_cs500_co200_nomic-... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.150 | 0.226 | 0.031 | 0.426 | 0.930 | 0.952 | 0.941 |
| semantic_cs500_co100_nomic-... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.324 | 0.573 | 0.064 | 0.602 | 0.909 | 0.919 | 0.914 |
| semantic_cs500_co100_nomic-... | 0.740 | 0.740 | 0.222 | 0.860 | 0.703 | 0.400 | 0.900 | 0.719 | 0.542 | 0.596 | 0.474 | 0.150 | 0.226 | 0.031 | 0.426 | 0.930 | 0.952 | 0.941 |

## Insights

- Best overall: semantic_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed (composite score: 0.569)
-   Best at: faithfulness
- Highest faithfulness: semantic_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed (0.873)
- Fastest TTFT: semantic_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise (0.94s) — range: 0.94s to 1.34s (1.4x spread)

## Detailed Statistics

### semantic_cs1000_co200_nomic...

- TTFT: 1.054s +/- 0.560s (range: 0.563-6.084)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.858 +/- 0.336 (range: 0.000-1.000)

### semantic_cs1000_co200_nomic...

- TTFT: 1.232s +/- 0.383s (range: 0.759-2.869)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.873 +/- 0.276 (range: 0.000-1.000)

### semantic_cs1000_co100_nomic...

- TTFT: 0.954s +/- 0.313s (range: 0.554-3.245)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.858 +/- 0.333 (range: 0.000-1.000)

### semantic_cs1000_co100_nomic...

- TTFT: 1.203s +/- 0.332s (range: 0.755-2.028)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.867 +/- 0.278 (range: 0.000-1.000)

### semantic_cs500_co200_nomic-...

- TTFT: 1.053s +/- 0.847s (range: 0.548-5.849)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.853 +/- 0.336 (range: 0.000-1.000)

### semantic_cs500_co200_nomic-...

- TTFT: 1.345s +/- 0.912s (range: 0.706-6.321)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.870 +/- 0.276 (range: 0.000-1.000)

### semantic_cs500_co100_nomic-...

- TTFT: 0.937s +/- 0.176s (range: 0.559-1.479)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.855 +/- 0.336 (range: 0.000-1.000)

### semantic_cs500_co100_nomic-...

- TTFT: 1.248s +/- 0.351s (range: 0.759-2.198)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.871 +/- 0.278 (range: 0.000-1.000)
