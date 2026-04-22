# RAG Benchmark Report

**Date:** 20260422_110753
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 4

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 0.708 | 0.0 | 0.3 | 877.6 |
| recursive_cs1000_co200_nomi... | 1.113 | 0.0 | 2.6 | 1297.2 |
| recursive_cs500_co200_nomic... | 0.695 | 0.0 | 4.2 | 904.9 |
| recursive_cs500_co200_nomic... | 0.956 | 0.0 | 5.3 | 1176.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.847 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co200_nomi... | 0.874 | N/A | N/A | N/A | N/A |
| recursive_cs500_co200_nomic... | 0.867 | N/A | N/A | N/A | N/A |
| recursive_cs500_co200_nomic... | 0.863 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.367 | 0.592 | 0.069 | 0.583 | 0.852 | 0.860 | 0.856 |
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.156 | 0.212 | 0.030 | 0.397 | 0.881 | 0.903 | 0.892 |
| recursive_cs500_co200_nomic... | 0.720 | 0.720 | 0.352 | 0.850 | 0.793 | 0.850 | 0.850 | 0.793 | 0.850 | 0.678 | 0.474 | 0.369 | 0.592 | 0.076 | 0.561 | 0.872 | 0.880 | 0.876 |
| recursive_cs500_co200_nomic... | 0.720 | 0.720 | 0.352 | 0.850 | 0.793 | 0.850 | 0.850 | 0.793 | 0.850 | 0.678 | 0.474 | 0.152 | 0.222 | 0.033 | 0.399 | 0.863 | 0.883 | 0.873 |

## Insights

- Best overall: recursive_cs500_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (composite score: 0.615)
-   Best at: ttft
- Highest faithfulness: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (0.874)
- Fastest TTFT: recursive_cs500_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (0.70s) — range: 0.70s to 1.11s (1.6x spread)

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 0.708s +/- 0.151s (range: 0.426-1.186)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.847 +/- 0.352 (range: 0.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 1.113s +/- 0.427s (range: 0.587-3.462)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.874 +/- 0.275 (range: 0.000-1.000)

### recursive_cs500_co200_nomic...

- TTFT: 0.695s +/- 0.172s (range: 0.388-1.218)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.867 +/- 0.323 (range: 0.000-1.000)

### recursive_cs500_co200_nomic...

- TTFT: 0.956s +/- 0.301s (range: 0.512-1.722)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.863 +/- 0.295 (range: 0.000-1.000)
