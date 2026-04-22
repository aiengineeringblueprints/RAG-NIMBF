# RAG Benchmark Report

**Date:** 20260422_134625
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 8

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 1.225 | 0.0 | 0.7 | 1048.8 |
| recursive_cs1000_co200_nomi... | 1.527 | 0.0 | 0.0 | 1379.2 |
| recursive_cs1000_co100_nomi... | 1.219 | 0.0 | 0.1 | 988.7 |
| recursive_cs1000_co100_nomi... | 1.528 | 0.0 | 0.0 | 1398.8 |
| recursive_cs500_co200_nomic... | 0.948 | 0.0 | 0.2 | 884.4 |
| recursive_cs500_co200_nomic... | 1.286 | 0.0 | 0.1 | 1309.2 |
| recursive_cs500_co100_nomic... | 0.901 | 0.0 | 0.2 | 964.7 |
| recursive_cs500_co100_nomic... | 1.277 | 0.0 | 0.3 | 1309.0 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.889 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co200_nomi... | 0.921 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.898 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.876 | N/A | N/A | N/A | N/A |
| recursive_cs500_co200_nomic... | 0.842 | N/A | N/A | N/A | N/A |
| recursive_cs500_co200_nomic... | 0.875 | N/A | N/A | N/A | N/A |
| recursive_cs500_co100_nomic... | 0.872 | N/A | N/A | N/A | N/A |
| recursive_cs500_co100_nomic... | 0.904 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.222 | 0.820 | 0.680 | 0.397 | 0.850 | 0.678 | 0.508 | 0.574 | 0.474 | 0.314 | 0.546 | 0.071 | 0.586 | 0.917 | 0.928 | 0.923 |
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.222 | 0.820 | 0.680 | 0.397 | 0.850 | 0.678 | 0.508 | 0.574 | 0.474 | 0.159 | 0.214 | 0.026 | 0.409 | 0.929 | 0.952 | 0.940 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.205 | 0.820 | 0.658 | 0.386 | 0.850 | 0.672 | 0.519 | 0.574 | 0.474 | 0.304 | 0.517 | 0.071 | 0.568 | 0.926 | 0.938 | 0.932 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.205 | 0.820 | 0.658 | 0.386 | 0.850 | 0.672 | 0.519 | 0.574 | 0.474 | 0.155 | 0.214 | 0.028 | 0.406 | 0.928 | 0.952 | 0.940 |
| recursive_cs500_co200_nomic... | 0.720 | 0.720 | 0.202 | 0.850 | 0.723 | 0.413 | 0.890 | 0.725 | 0.553 | 0.605 | 0.474 | 0.329 | 0.570 | 0.079 | 0.590 | 0.899 | 0.909 | 0.904 |
| recursive_cs500_co200_nomic... | 0.720 | 0.720 | 0.202 | 0.850 | 0.723 | 0.413 | 0.890 | 0.725 | 0.553 | 0.605 | 0.474 | 0.155 | 0.216 | 0.030 | 0.418 | 0.910 | 0.932 | 0.921 |
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.197 | 0.860 | 0.718 | 0.409 | 0.900 | 0.728 | 0.553 | 0.600 | 0.474 | 0.344 | 0.595 | 0.096 | 0.602 | 0.891 | 0.899 | 0.895 |
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.197 | 0.860 | 0.718 | 0.409 | 0.900 | 0.728 | 0.553 | 0.600 | 0.474 | 0.153 | 0.211 | 0.030 | 0.402 | 0.890 | 0.913 | 0.902 |

## Insights

- Best overall: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed (composite score: 0.500)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed (0.921)
- Fastest TTFT: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise (0.90s) — range: 0.90s to 1.53s (1.7x spread)

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 1.225s +/- 0.194s (range: 0.901-2.050)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.889 +/- 0.302 (range: 0.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 1.527s +/- 0.373s (range: 0.979-2.902)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.921 +/- 0.217 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 1.219s +/- 0.211s (range: 0.760-2.128)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.898 +/- 0.289 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 1.528s +/- 0.500s (range: 1.040-4.537)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.876 +/- 0.261 (range: 0.000-1.000)

### recursive_cs500_co200_nomic...

- TTFT: 0.948s +/- 0.152s (range: 0.675-1.379)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.842 +/- 0.353 (range: 0.000-1.000)

### recursive_cs500_co200_nomic...

- TTFT: 1.286s +/- 0.312s (range: 0.808-1.993)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.875 +/- 0.254 (range: 0.000-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 0.901s +/- 0.162s (range: 0.557-1.378)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.872 +/- 0.329 (range: 0.000-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 1.277s +/- 0.385s (range: 0.761-2.842)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.904 +/- 0.215 (range: 0.000-1.000)
