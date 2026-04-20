# RAG Benchmark Report

**Date:** 20260420_195107
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co200_nomi... | 0.494 | 0.0 | 0.3 | 852.4 |
| recursive_cs1000_co200_nomi... | 0.803 | 0.0 | 0.1 | 1561.6 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co200_nomi... | 0.885 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co200_nomi... | 0.896 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.258 | 0.319 | 0.051 | 0.468 | 0.823 | 0.837 | 0.830 |
| recursive_cs1000_co200_nomi... | 0.700 | 0.700 | 0.387 | 0.820 | 0.772 | 0.820 | 0.820 | 0.772 | 0.820 | 0.645 | 0.474 | 0.201 | 0.118 | 0.019 | 0.280 | 0.845 | 0.873 | 0.859 |

## Insights

- Best overall: recursive_cs1000_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (composite score: 0.500)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs1000_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (0.896)
- Fastest TTFT: recursive_cs1000_co200_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise_mmr-l0.5 (0.49s) — range: 0.49s to 0.80s (1.6x spread)

## Detailed Statistics

### recursive_cs1000_co200_nomi...

- TTFT: 0.494s +/- 0.329s (range: 0.279-2.749)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.885 +/- 0.317 (range: 0.000-1.000)

### recursive_cs1000_co200_nomi...

- TTFT: 0.803s +/- 0.757s (range: 0.316-3.316)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.896 +/- 0.206 (range: 0.000-1.000)
