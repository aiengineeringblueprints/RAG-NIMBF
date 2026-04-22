# RAG Benchmark Report

**Date:** 20260422_224257
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 8

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| semantic_cs1000_co100_nomic... | 0.655 | 0.0 | 0.3 | 822.0 |
| semantic_cs1000_co100_nomic... | 0.961 | 0.0 | 0.1 | 1204.2 |
| semantic_cs1000_co200_nomic... | 0.711 | 0.0 | 0.3 | 832.7 |
| semantic_cs1000_co200_nomic... | 1.011 | 0.0 | 0.6 | 1213.1 |
| semantic_cs500_co100_nomic-... | 0.736 | 0.0 | 0.4 | 847.9 |
| semantic_cs500_co100_nomic-... | 1.083 | 0.0 | 0.3 | 1235.8 |
| semantic_cs500_co200_nomic-... | 0.698 | 0.0 | 0.2 | 874.1 |
| semantic_cs500_co200_nomic-... | 1.038 | 0.0 | 0.1 | 1206.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| semantic_cs1000_co100_nomic... | 0.799 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co100_nomic... | 0.868 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co200_nomic... | 0.829 | N/A | N/A | N/A | N/A |
| semantic_cs1000_co200_nomic... | 0.866 | N/A | N/A | N/A | N/A |
| semantic_cs500_co100_nomic-... | 0.819 | N/A | N/A | N/A | N/A |
| semantic_cs500_co100_nomic-... | 0.865 | N/A | N/A | N/A | N/A |
| semantic_cs500_co200_nomic-... | 0.809 | N/A | N/A | N/A | N/A |
| semantic_cs500_co200_nomic-... | 0.846 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| semantic_cs1000_co100_nomic... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.362 | 0.571 | 0.067 | 0.565 | 0.862 | 0.870 | 0.866 |
| semantic_cs1000_co100_nomic... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.152 | 0.219 | 0.028 | 0.410 | 0.910 | 0.933 | 0.921 |
| semantic_cs1000_co200_nomic... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.362 | 0.571 | 0.067 | 0.565 | 0.862 | 0.870 | 0.866 |
| semantic_cs1000_co200_nomic... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.152 | 0.219 | 0.028 | 0.410 | 0.910 | 0.933 | 0.921 |
| semantic_cs500_co100_nomic-... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.362 | 0.571 | 0.067 | 0.565 | 0.862 | 0.870 | 0.866 |
| semantic_cs500_co100_nomic-... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.152 | 0.219 | 0.028 | 0.410 | 0.910 | 0.933 | 0.921 |
| semantic_cs500_co200_nomic-... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.362 | 0.571 | 0.067 | 0.565 | 0.862 | 0.870 | 0.866 |
| semantic_cs500_co200_nomic-... | 0.740 | 0.740 | 0.410 | 0.860 | 0.809 | 0.860 | 0.860 | 0.809 | 0.860 | 0.670 | 0.474 | 0.152 | 0.219 | 0.028 | 0.410 | 0.910 | 0.933 | 0.921 |

## Insights

- Best overall: semantic_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (composite score: 0.571)
-   Best at: faithfulness
- Highest faithfulness: semantic_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (0.868)
- Fastest TTFT: semantic_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (0.66s) — range: 0.66s to 1.08s (1.7x spread)

## Detailed Statistics

### semantic_cs1000_co100_nomic...

- TTFT: 0.655s +/- 0.180s (range: 0.331-1.174)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.799 +/- 0.394 (range: 0.000-1.000)

### semantic_cs1000_co100_nomic...

- TTFT: 0.961s +/- 0.342s (range: 0.485-1.706)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.868 +/- 0.264 (range: 0.000-1.000)

### semantic_cs1000_co200_nomic...

- TTFT: 0.711s +/- 0.204s (range: 0.430-1.563)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.829 +/- 0.369 (range: 0.000-1.000)

### semantic_cs1000_co200_nomic...

- TTFT: 1.011s +/- 0.373s (range: 0.549-2.209)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.866 +/- 0.264 (range: 0.000-1.000)

### semantic_cs500_co100_nomic-...

- TTFT: 0.736s +/- 0.228s (range: 0.415-1.807)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.819 +/- 0.378 (range: 0.000-1.000)

### semantic_cs500_co100_nomic-...

- TTFT: 1.083s +/- 0.408s (range: 0.463-2.809)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.865 +/- 0.266 (range: 0.000-1.000)

### semantic_cs500_co200_nomic-...

- TTFT: 0.698s +/- 0.208s (range: 0.364-1.481)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.809 +/- 0.386 (range: 0.000-1.000)

### semantic_cs500_co200_nomic-...

- TTFT: 1.038s +/- 0.337s (range: 0.628-1.886)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.846 +/- 0.283 (range: 0.000-1.000)
