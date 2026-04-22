# RAG Benchmark Report

**Date:** 20260422_095422
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.722 | 0.0 | 0.3 | 805.9 |
| recursive_cs1000_co100_nomi... | 1.007 | 0.0 | 0.1 | 1222.9 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.827 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.865 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.474 | 0.374 | 0.605 | 0.084 | 0.585 | 0.852 | 0.861 | 0.856 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.474 | 0.155 | 0.213 | 0.031 | 0.396 | 0.900 | 0.923 | 0.911 |

## Insights

- Best overall: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (composite score: 0.500)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (0.865)
- Fastest TTFT: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (0.72s) — range: 0.72s to 1.01s (1.4x spread)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.722s +/- 0.198s (range: 0.448-1.968)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.827 +/- 0.371 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 1.007s +/- 0.346s (range: 0.509-1.785)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.865 +/- 0.292 (range: 0.000-1.000)
