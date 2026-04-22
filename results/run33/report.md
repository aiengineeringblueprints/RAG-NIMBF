# RAG Benchmark Report

**Date:** 20260422_183718
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs500_co100_nomic... | 0.632 | 0.0 | 0.4 | 810.3 |
| recursive_cs500_co100_nomic... | 0.944 | 0.0 | 0.1 | 1114.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs500_co100_nomic... | 0.842 | N/A | N/A | N/A | N/A |
| recursive_cs500_co100_nomic... | 0.848 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.368 | 0.860 | 0.805 | 0.860 | 0.860 | 0.805 | 0.860 | 0.672 | 0.474 | 0.360 | 0.607 | 0.067 | 0.579 | 0.892 | 0.900 | 0.896 |
| recursive_cs500_co100_nomic... | 0.740 | 0.740 | 0.368 | 0.860 | 0.805 | 0.860 | 0.860 | 0.805 | 0.860 | 0.672 | 0.474 | 0.150 | 0.215 | 0.031 | 0.395 | 0.882 | 0.903 | 0.892 |

## Insights

- Best overall: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (composite score: 0.500)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed_mmr-l0.5 (0.848)
- Fastest TTFT: recursive_cs500_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_concise_mmr-l0.5 (0.63s) — range: 0.63s to 0.94s (1.5x spread)

## Detailed Statistics

### recursive_cs500_co100_nomic...

- TTFT: 0.632s +/- 0.179s (range: 0.339-1.246)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.842 +/- 0.353 (range: 0.000-1.000)

### recursive_cs500_co100_nomic...

- TTFT: 0.944s +/- 0.325s (range: 0.476-1.788)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.848 +/- 0.318 (range: 0.000-1.000)
