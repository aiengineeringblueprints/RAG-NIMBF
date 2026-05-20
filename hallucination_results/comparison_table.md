# Hallucination Detection — Method Comparison

## ragas-wikiqa/FinQA

| Method | N | Mean Score | AUROC | Pearson r | Spearman r | RMSE vs RAGAS | Hal. % | Sig. |
|--------|---|------------|-------|-----------|------------|---------------|--------|------|
| ragas_faithfulness | 739 | 0.878 | 1.000 | 1.000** | 1.000 | 0.000 | 10.1% | ** |

## squad/FinQA

| Method | N | Mean Score | AUROC | Pearson r | Spearman r | RMSE vs RAGAS | Hal. % | Sig. |
|--------|---|------------|-------|-----------|------------|---------------|--------|------|
| ragas_faithfulness | 9922 | 0.855 | 1.000 | 1.000** | 1.000 | 0.000 | 12.1% | ** |

## t2-ragbench/FinQA

| Method | N | Mean Score | AUROC | Pearson r | Spearman r | RMSE vs RAGAS | Hal. % | Sig. |
|--------|---|------------|-------|-----------|------------|---------------|--------|------|
| ragas_faithfulness | 131 | 0.505 | 1.000 | 1.000** | 1.000 | 0.000 | 49.6% | ** |

## unknown/FinQA

| Method | N | Mean Score | AUROC | Pearson r | Spearman r | RMSE vs RAGAS | Hal. % | Sig. |
|--------|---|------------|-------|-----------|------------|---------------|--------|------|
| ragas_faithfulness | 99 | 0.232 | 1.000 | 1.000** | 1.000 | 0.000 | 75.8% | ** |

## Aggregate (Mean Across Datasets)

| Method | Datasets | Mean Score | Mean AUROC | Mean Pearson r | Mean RMSE |
|--------|----------|------------|-----------|----------------|-----------|
| ragas_faithfulness | 4 | 0.617 | 1.000 | 1.000 | 0.000 |
