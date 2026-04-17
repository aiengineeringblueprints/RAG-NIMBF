# RAG Benchmark Report

**Date:** 20260417_093459
**Dataset:** ragas-wikiqa/FinQA (100 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1000_co100_nomi... | 1.791 | 0.0 | 0.0 | 2788.3 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1000_co100_nomi... | 0.813 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1000_co100_nomi... | 1.000 | 1.000 | 0.125 | 1.000 | 1.000 | 0.375 | 1.000 | 1.000 | 0.625 | 0.623 | 0.183 | 0.052 | 0.326 | 0.803 | 0.815 | 0.809 |

## Insights

- Single configuration: recursive_cs1000_co100_nomic-embed-text:latest_Qwen/Qwen3-32B-AWQ_detailed
-   Processed 100 questions across 1117 chunks
-   TTFT: 1.791s (std 0.367s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.813
-   Total time: 2788.3s

## Detailed Statistics

### recursive_cs1000_co100_nomi...

- TTFT: 1.791s +/- 0.367s (range: 0.867-2.518)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.813 +/- 0.270 (range: 0.000-1.000)
