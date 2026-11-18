# 14 — Calibration: direct RAGAS vs. MLflow scorers

Issue #46 (parent #40). Date: 2026-11-18. Run: `results/run100` on the DGX Spark.

**Decision: side by side. Direct RAGAS stays the default evaluator; the
MLflow scorers stay opt-in (`mlflow_scorers`).** No MLflow scorer meets the
replacement criterion, and the scorers are 2.4× slower per sample even when
throttled. The default evaluator does not change, so no ADR.

## 1. Setup

Manifest: [`experiments/ragas-mlflow-calibration.yaml`](../experiments/ragas-mlflow-calibration.yaml).

| Item | Value |
|---|---|
| Dataset | SQuAD, fixed 50-question slice (`shuffle(seed=42)`) |
| Pipeline | internal, recursive 512/64, `nomic-embed-text`, similarity top-k 4, template `concise` |
| Generator | `openai:qwen3.8-27b` |
| Critic (both paths) | `ollama:gemma3:12b`, temperature 0, max_tokens 4096 |
| Direct RAGAS | faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall; `eval_critic_max_workers: 8` |
| MLflow scorers | `ragas_faithfulness`, `ragas_context_precision`, `ragas_context_recall`, `ragas_factual_correctness`, `answer_correctness` (#45), `retrieval_groundedness`, `retrieval_relevance`, `retrieval_sufficiency`, `correctness`, `relevance_to_query` |
| MLflow run | `ac134264f76b4961bb1b6f877de6ce52` (Spark `sqlite:///mlflow.db`, experiment `ragas-mlflow-calibration`) |

Both evaluators judge the same 50 generated answers. Direct RAGAS scores
live in the config JSON (`per_sample[i].ragas_scores`); the MLflow scorers
write assessments on the run's traces, joined to samples via
`benchmark.sample_index`. Report:

```bash
python -m benchmark.reporting.judge_calibration \
  results/run100/configs/recursive_cs512_co64_nomic-embed-text_latest_qwen3.8-27b_concise_k4.json \
  --run-id ac134264f76b4961bb1b6f877de6ce52 --tracking-uri sqlite:///mlflow.db \
  --out results/run100/judge_calibration_pass2_throttled.md
```

Δ = MLflow score − direct score on samples where both produced a value.
Binary built-ins (yes/no) are mapped to 1/0. Failure = no score for a sample;
the report does not separate parse errors from timeouts (pass 1 failures
were timeouts per the worker log; pass 2 had none).

## 2. Pass 1: MLflow defaults (unusable)

The worker ran the scorers with MLflow's default concurrency: 10 rows × 10
scorers in parallel (`MLFLOW_GENAI_EVAL_MAX_WORKERS` /
`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS`, both 10), 60 s LLM timeout and
300 s scorer timeout. Up to ~100 concurrent judge calls flooded the single
Ollama instance:

- `ragas_faithfulness`, `ragas_context_precision`,
  `ragas_factual_correctness`, `retrieval_relevance`: **100 % failure**.
- All other scorers: 38–88 % failure.
- Direct RAGAS (8 workers): 0 % failure on every metric.

The 314 s judge wall time of pass 1 only reflects how fast calls timed out.
Saved as `results/run100/judge_calibration_pass1_default_concurrency.md`
(generated before pairs with 100 % judge failure were reported, so those
four are missing from that table).

## 3. Pass 2: throttled (the comparison)

Same traces re-scored by `results/run100/rescore_pass2.py` with
`MLFLOW_GENAI_EVAL_MAX_WORKERS=8` (= direct RAGAS workers),
`MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=1`, `MLFLOW_GENAI_EVAL_LLM_TIMEOUT=300`,
`MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT=1800`. The report reads the newest
assessment per trace, so pass 2 supersedes pass 1. **0 % failure on both
sides for every pair.**

| Direct RAGAS | MLflow scorer | Relation | Direct mean | MLflow mean | Mean Δ | Mean \|Δ\| | Pearson | Spearman |
|---|---|---|---|---|---|---|---|---|
| faithfulness | ragas_faithfulness | same RAGAS metric | 0.875 | 0.835 | −0.040 | 0.040 | 0.846 | 0.856 |
| context_precision | ragas_context_precision | same RAGAS metric | 0.947 | 0.938 | −0.009 | 0.052 | 0.778 | 0.333 |
| context_recall | ragas_context_recall | same RAGAS metric | 0.960 | 1.000 | +0.040 | 0.040 | – | – |
| answer_correctness | answer_correctness | custom claim-F1 judge (#45) | 0.819 | 0.873 | +0.054 | 0.076 | 0.937 | 0.599 |
| answer_correctness | ragas_factual_correctness | claim F1, no similarity term | 0.819 | 0.300 | −0.519 | 0.521 | 0.414 | 0.253 |
| faithfulness | retrieval_groundedness | binary proxy | 0.875 | 0.880 | +0.005 | 0.045 | 0.805 | 0.755 |
| context_precision | retrieval_relevance | unweighted relevant-chunk share | 0.947 | 0.345 | −0.602 | 0.602 | 0.089 | 0.428 |
| context_recall | retrieval_sufficiency | binary proxy | 0.960 | 0.960 | 0.000 | 0.040 | 0.479 | 0.479 |
| answer_correctness | correctness | binary, recall-only | 0.819 | 0.880 | +0.061 | 0.080 | 0.933 | 0.577 |
| answer_relevancy | relevance_to_query | binary proxy | 0.561 | 0.880 | +0.319 | 0.319 | 0.688 | 0.526 |

`–`: correlation undefined because one side is constant
(`ragas_context_recall` scored 1.0 on all 50 samples).

### Runtime

| Path | Wall time | Per sample |
|---|---|---|
| Direct RAGAS (`stage_ragas_eval_seconds`, 5 metrics) | 788 s | 15.8 s |
| MLflow scorers, pass 2 (`mlflow_judge_seconds`, 10 scorers) | 1929 s | 38.6 s |
| MLflow scorers, pass 1 (defaults, mostly failed) | 314 s | 6.3 s |

The MLflow path runs twice as many judges, so per judge it is roughly on
par; the point is that it cannot run faster against one Ollama critic
without failing.

## 4. Reading the results

- **Same-math RAGAS wrappers** agree on the mean (|Δ| ≈ 0.04–0.05) but not
  on the ranking. On this easy SQuAD slice most scores sit at the ceiling,
  so a few disagreements dominate the rank correlation
  (`context_precision`: Spearman 0.33). Ceiling effects make the
  correlations here a lower bound on what a harder dataset would show.
- **Custom `answer_correctness` (#45)** tracks direct RAGAS linearly
  (Pearson 0.94) with a +0.05 bias; ranking only moderate (Spearman 0.60).
- **`ragas_factual_correctness` is not a substitute for
  `answer_correctness`** (Δ −0.52, Spearman 0.25): it is claim F1 without the
  semantic-similarity term that RAGAS `answer_correctness` blends in.
- **`retrieval_relevance` is not a substitute for `context_precision`**
  (Δ −0.60, Pearson 0.09): it is the unweighted share of relevant chunks,
  while RAGAS context precision is rank-weighted (average precision over
  relevant positions), so one relevant chunk at rank 1 already scores 1.0.
  With top-k 4 and one gold passage, an unweighted share near 1/4 is
  expected.
- **Binary proxies**: `retrieval_groundedness` is the closest of all pairs
  (|Δ| 0.045, Spearman 0.755). `retrieval_sufficiency` matches on the mean
  (|Δ| 0.040) but ranks weakly (0.48); `correctness` misses the mean bound
  (|Δ| 0.080).
  `relevance_to_query` says yes far more often than RAGAS answer relevancy
  (+0.32).

## 5. Decision

Replacement criterion (doc 13 §6.5, research for #40, not yet merged): mean |Δ| < 0.05 **and** rank
correlation > 0.9 for every replaced metric, `answer_correctness` covered,
and the critic parses reliably.

| Criterion | Result |
|---|---|
| mean \|Δ\| < 0.05 | met by `ragas_faithfulness`, `ragas_context_recall`, `retrieval_groundedness`, `retrieval_sufficiency` |
| Spearman > 0.9 | **met by no pair** |
| `answer_correctness` covered | custom judge close on mean (0.076), Spearman 0.60 — not met |
| Critic parses reliably | yes when throttled (0 % failure); no with MLflow defaults |

→ **Side by side.** Direct RAGAS remains the default and the number to
compare across runs. MLflow scorers stay opt-in for trace-level inspection
in the MLflow UI. No hybrid: no single metric is interchangeable enough to
move it to the MLflow path.

## 6. Operating notes

- Running MLflow scorers against one local critic needs throttling:
  `MLFLOW_GENAI_EVAL_MAX_WORKERS=8 MLFLOW_GENAI_EVAL_MAX_SCORER_WORKERS=1`,
  plus `MLFLOW_GENAI_EVAL_LLM_TIMEOUT=300` and
  `MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT=1800`. Otherwise most scores are missing.
- Do not read `ragas_factual_correctness` or `retrieval_relevance` as the
  direct RAGAS `answer_correctness` / `context_precision`.
- Re-scoring a finished run: `score_run_traces(run_id, names, endpoint)`
  (see `results/run100/rescore_pass2.py`); the report picks the newest
  assessment.

## 7. Limits

One dataset (SQuAD, easy, ceiling-heavy), one config, one critic, 50
samples. A harder dataset (e.g. RAGBench TechQA) could separate the
evaluators better; rerun the manifest with another `dataset` if the
question comes back.
