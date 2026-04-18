# RAG Benchmark Report

**Date:** 20260418_094438
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.982 | 0.0 | 0.3 | 1001.9 |
| recursive_cs1000_co100_nomi... | 1.019 | 0.0 | 0.0 | 1610.4 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.910 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.940 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.205 | 0.820 | 0.658 | 0.386 | 0.850 | 0.672 | 0.519 | 0.574 | 0.352 | 0.051 | 0.518 | 0.872 | 0.887 | 0.879 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.205 | 0.820 | 0.658 | 0.386 | 0.850 | 0.672 | 0.519 | 0.574 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Insights

- Best overall: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed (composite score: 0.500)
-   Best at: faithfulness
- Highest faithfulness: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed (0.940)
- Fastest TTFT: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise (0.98s) — range: 0.98s to 1.02s (1.0x spread)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.982s +/- 0.701s (range: 0.520-3.630)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.910 +/- 0.288 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 1.019s +/- 0.661s (range: 0.569-3.750)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.940 +/- 0.154 (range: 0.000-1.000)
