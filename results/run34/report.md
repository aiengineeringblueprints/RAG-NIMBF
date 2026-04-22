# RAG Benchmark Report

**Date:** 20260422_202142
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 4

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.713 | 0.0 | 0.1 | 806.8 |
| recursive_cs1000_co100_nomi... | 1.068 | 0.0 | 0.0 | 1225.0 |
| recursive_cs1000_co200_nomi... | 0.711 | 0.0 | 0.7 | 826.4 |
| recursive_cs1000_co200_nomi... | 1.149 | 0.0 | 0.3 | 1167.1 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.843 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.865 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co200_nomi... | 0.840 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co200_nomi... | 0.863 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.474 | 0.374 | 0.605 | 0.084 | 0.585 | 0.852 | 0.861 | 0.856 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.474 | 0.155 | 0.213 | 0.031 | 0.396 | 0.900 | 0.923 | 0.911 |
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.367 | 0.592 | 0.069 | 0.583 | 0.852 | 0.860 | 0.856 |
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.156 | 0.212 | 0.030 | 0.397 | 0.881 | 0.903 | 0.892 |

## Insights

- Best overall: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (composite score: 0.546)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (0.865)
- Fastest TTFT: recursive_cs1000_co200_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (0.71s) — range: 0.71s to 1.15s (1.6x spread)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.713s +/- 0.175s (range: 0.444-1.422)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.843 +/- 0.352 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 1.068s +/- 0.365s (range: 0.541-2.349)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.865 +/- 0.291 (range: 0.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 0.711s +/- 0.150s (range: 0.441-1.276)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.840 +/- 0.362 (range: 0.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 1.149s +/- 0.498s (range: 0.558-3.012)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.863 +/- 0.289 (range: 0.000-1.000)
