Config `recursive_cs512_co64_nomic-embed-text:latest_qwen3.8-27b_concise_k4`, MLflow run `ac134264f76b4961bb1b6f877de6ce52`, 50 samples. Δ = judge − direct over paired samples; failure = no score for a sample.

| Direct RAGAS | MLflow scorer | Relation | Paired | Direct mean | Judge mean | Mean Δ | Mean \|Δ\| | Pearson | Spearman | Direct fail | Judge fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| context_recall | ragas_context_recall | same RAGAS metric via MLflow | 6/50 | 0.833 | 1.000 | +0.167 | 0.167 | – | – | 0% | 88% |
| answer_correctness | answer_correctness | custom claim-F1 judge (#45) | 26/50 | 0.802 | 0.846 | +0.044 | 0.069 | 0.965 | 0.650 | 0% | 48% |
| faithfulness | retrieval_groundedness | binary proxy | 31/50 | 0.871 | 0.871 | +0.000 | 0.065 | 0.713 | 0.713 | 0% | 38% |
| context_recall | retrieval_sufficiency | binary proxy | 26/50 | 0.962 | 1.000 | +0.038 | 0.038 | – | – | 0% | 48% |
| answer_correctness | correctness | binary, recall-only | 28/50 | 0.807 | 0.857 | +0.050 | 0.073 | 0.965 | 0.617 | 0% | 44% |
| answer_relevancy | relevance_to_query | binary proxy | 26/50 | 0.612 | 0.885 | +0.273 | 0.273 | 0.834 | 0.554 | 0% | 48% |

| Runtime | Seconds | Per sample |
|---|---|---|
| Direct RAGAS (`stage_ragas_eval_seconds`) | 788 | 15.769 |
| MLflow scorers (`mlflow_judge_seconds`) | 314 | 6.283 |
