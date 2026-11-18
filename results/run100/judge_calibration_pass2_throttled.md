Config `recursive_cs512_co64_nomic-embed-text:latest_qwen3.8-27b_concise_k4`, MLflow run `ac134264f76b4961bb1b6f877de6ce52`, 50 samples. Δ = judge − direct over paired samples; failure = no score for a sample.

| Direct RAGAS | MLflow scorer | Relation | Paired | Direct mean | Judge mean | Mean Δ | Mean \|Δ\| | Pearson | Spearman | Direct fail | Judge fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| faithfulness | ragas_faithfulness | same RAGAS metric via MLflow | 50/50 | 0.875 | 0.835 | -0.040 | 0.040 | 0.846 | 0.856 | 0% | 0% |
| context_precision | ragas_context_precision | same RAGAS metric via MLflow | 50/50 | 0.947 | 0.938 | -0.009 | 0.052 | 0.778 | 0.333 | 0% | 0% |
| context_recall | ragas_context_recall | same RAGAS metric via MLflow | 50/50 | 0.960 | 1.000 | +0.040 | 0.040 | – | – | 0% | 0% |
| answer_correctness | answer_correctness | custom claim-F1 judge (#45) | 50/50 | 0.819 | 0.873 | +0.054 | 0.076 | 0.937 | 0.599 | 0% | 0% |
| answer_correctness | ragas_factual_correctness | claim F1, no similarity term | 50/50 | 0.819 | 0.300 | -0.519 | 0.521 | 0.414 | 0.253 | 0% | 0% |
| faithfulness | retrieval_groundedness | binary proxy | 50/50 | 0.875 | 0.880 | +0.005 | 0.045 | 0.805 | 0.755 | 0% | 0% |
| context_precision | retrieval_relevance | unweighted relevant-chunk share | 50/50 | 0.947 | 0.345 | -0.602 | 0.602 | 0.089 | 0.428 | 0% | 0% |
| context_recall | retrieval_sufficiency | binary proxy | 50/50 | 0.960 | 0.960 | +0.000 | 0.040 | 0.479 | 0.479 | 0% | 0% |
| answer_correctness | correctness | binary, recall-only | 50/50 | 0.819 | 0.880 | +0.061 | 0.080 | 0.933 | 0.577 | 0% | 0% |
| answer_relevancy | relevance_to_query | binary proxy | 50/50 | 0.561 | 0.880 | +0.319 | 0.319 | 0.688 | 0.526 | 0% | 0% |

| Runtime | Seconds | Per sample |
|---|---|---|
| Direct RAGAS (`stage_ragas_eval_seconds`) | 788 | 15.769 |
| MLflow scorers (`mlflow_judge_seconds`) | 1929 | 38.579 |
