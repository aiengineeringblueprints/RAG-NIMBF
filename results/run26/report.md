# RAG Benchmark Report

**Date:** 20260418_104722
**Dataset:** squad/FinQA (100 samples)
**Configurations:** 2

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 0.974 | 0.0 | 0.2 | 886.5 |
| recursive_cs1000_co100_nomi... | 0.922 | 0.0 | 0.0 | 1540.4 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.885 | N/A | N/A | N/A | N/A |
| recursive_cs1000_co100_nomi... | 0.920 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.328 | 0.051 | 0.472 | 0.833 | 0.847 | 0.840 |
| recursive_cs1000_co100_nomi... | 0.670 | 0.670 | 0.372 | 0.820 | 0.763 | 0.820 | 0.820 | 0.763 | 0.820 | 0.645 | 0.120 | 0.020 | 0.283 | 0.836 | 0.863 | 0.849 |

## Insights

- Best overall: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (composite score: 0.750)
-   Best at: faithfulness, ttft
- Highest faithfulness: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (0.920)
- Fastest TTFT: recursive_cs1000_co100_nomic-embed-text:latest_Qwen3.5-397B-A17B_detailed_mmr-l0.5 (0.92s) — range: 0.92s to 0.97s (1.1x spread)

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 0.974s +/- 0.877s (range: 0.357-3.324)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.885 +/- 0.317 (range: 0.000-1.000)

### recursive_cs1000_co100_nomi...

- TTFT: 0.922s +/- 0.829s (range: 0.275-3.261)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.920 +/- 0.188 (range: 0.000-1.000)
