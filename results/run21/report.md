# RAG Benchmark Report

**Date:** 20260417_200245
**Dataset:** squad/FinQA (25 samples)
**Configurations:** 1

## Performance Summary

| Config | TTFT (s) | Tok/s | GPU % | Total (s) |
|--------|----------|-------|-------|-----------|
| recursive_cs1500_co350_nomi... | 1.651 | 0.0 | 0.0 | 295.4 |

## RAGAS Scores

| Config | Faithfulness | Answer Rel. | Answer Corr. | Ctx Precision | Ctx Recall |
|--------|-------------|-------------|--------------|---------------|------------|
| recursive_cs1500_co350_nomi... | 0.880 | N/A | N/A | N/A | N/A |

## Custom Metrics (IR + NLG)

| Config | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | rouge_l | bleu | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
|--------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|----------|
| recursive_cs1500_co350_nomi... | 0.760 | 0.760 | 0.290 | 0.800 | 0.690 | 0.426 | 0.840 | 0.717 | 0.584 | 0.522 | 0.383 | 0.067 | 0.544 | 0.814 | 0.828 | 0.821 |

## Insights

- Single configuration: recursive_cs1500_co350_nomic-embed-text:latest_Qwen3.5-397B-A17B_concise
-   Processed 25 questions across 26 chunks
-   TTFT: 1.651s (std 1.055s)
-   Throughput: 0.0 tok/s (std 0.0)
-   Faithfulness: 0.880
-   Total time: 295.4s

## Detailed Statistics

### recursive_cs1500_co350_nomi...

- TTFT: 1.651s +/- 1.055s (range: 0.737-3.722)
- Throughput: 0.0 tok/s +/- 0.0
- Faithfulness: 0.880 +/- 0.299 (range: 0.000-1.000)
