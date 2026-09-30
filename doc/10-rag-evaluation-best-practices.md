# RAG Evaluation & Benchmarking — Current Best Practices (2025–2026)

Date: 2026-09-14. Research-only note; no code changes.
Method: primary sources only — official docs, source code, specs, first-party papers (arXiv). Every factual claim is cited inline. Where frameworks disagree on a metric definition, the source code/doc that owns the definition is cited.

---

## TL;DR

1. **"Faithfulness" is not one metric.** RAGAS counts claims that *can be inferred from* the retrieved context ([RAGAS docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)); DeepEval counts claims that *do not contradict* the context — a strictly more lenient definition ([DeepEval docs](https://docs.confident-ai.com/docs/metrics-faithfulness)). Cross-framework scores are **not** comparable; a benchmark must own and document its definitions.
2. **Gold-ID retrieval metrics (hit@k, nDCG, MRR, recall@k) are the only judge-free layer** and the de-facto standard is the BEIR protocol: NDCG@k/MAP@k/Recall@k/P@k with k=[1,3,5,10,100,1000], MRR via custom evaluation ([BEIR repo](https://github.com/beir-cellar/beir)). Prefer these for system-to-system sweeps; use LLM judges on final candidates.
3. **LLM-as-judge needs its own QA loop**: frozen dev/validation/held-out golden sets, paired comparisons on identical rows, documented full judge config, and calibration against human labels — TruLens now ships a whole protocol for this ([TruLens Judge Alignment](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)); the underlying biases (position, verbosity, self-enhancement) are documented in [MT-Bench, arXiv:2306.05685](https://arxiv.org/abs/2306.05685).
4. **Statistical practice is shifting from single averages to effect sizes and paired tests** ([BEIR SIGIR 2024 paper, arXiv:2306.07471](https://arxiv.org/abs/2306.07471)); RAGBench found a fine-tuned RoBERTa classifier **outperforms LLM-judge RAG evaluators** ([arXiv:2407.11005](https://arxiv.org/abs/2407.11005)) — do not treat judge scores as ground truth.
5. **Synthetic test sets are mainstream but supplementary**: RAGAS builds them from a knowledge graph with scenario-based query synthesis ([docs](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)); LangSmith explicitly recommends hand-curated 10–20 examples first, synthetic data only to extend them ([LangSmith eval concepts](https://docs.smith.langchain.com/evaluation)).
6. **Platform shift**: MLflow 3 deprecated `mlflow.evaluate` in favor of `mlflow.genai.evaluate` + scorers/judges, with latency/tokens coming from traces ([migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)); Phoenix runs evals on trace spans and logs annotations back ([Phoenix evals](https://arize.com/docs/phoenix/evaluation)); TruLens is OpenTelemetry-native with a dedicated MCP span type ([TruLens README](https://github.com/truera/trulens)).
7. For this repo: the biggest alignment gaps are (a) pinning/migrating the **legacy RAGAS metric API** (deprecated in 0.4, removed in 1.0), (b) adding **paired significance tests** on per-sample scores already collected, and (c) formalizing a **judge-calibration golden set**.

---

## 1. Standard metrics and how each framework defines/computes them

### 1.1 RAGAS (owner of the most-copied definitions)

| Metric | Definition (from docs/source) | Source |
|---|---|---|
| Faithfulness | Decompose `response` into claims; score = (claims supported by `retrieved_contexts`) / (total claims). "Supported" means *can be inferred from* the context. Optional HHEM-2.1-Open (Vectara T5 classifier) replaces the LLM in the verification step. | [Faithfulness page](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) |
| Context Precision | Mean of precision@k over ranks, weighted by a relevance indicator `v_k` obtained by LLM-judging each chunk against a `reference` (AP-like). Variants: `LLMContextPrecisionWithoutReference` (judges vs response), `LLMContextPrecisionWithReference` (judges vs reference), `NonLLMContextPrecisionWithReference` (Levenshtein via rapidfuzz), `IDBasedContextPrecision` = \|retrieved∩reference\|/\|retrieved\|. | [Context Precision page](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) |
| Context Utilization | Same AP-style formula as Context Precision but relevance is judged against the **generated response** instead of a reference (reference-free setting). | [Context Precision page, "Context Utilization" section](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/) |
| Context Recall | Claims in `reference` attributable to `retrieved_contexts` / total reference claims. Non-LLM variant: \|relevant retrieved\|/\|reference contexts\|; ID variant: \|reference∩retrieved\|/\|reference\|. | [Context Recall page](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/) |
| Answer/Response Relevancy | Generate N (default 3, `strictness`) reverse-engineered questions from the response; score = mean cosine similarity between their embeddings and the question embedding. Docs explicitly note the score is **not guaranteed to be in [0,1]** because cosine similarity can be negative. | [Answer Relevancy page](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/) |

Caveats surfaced by the docs themselves: the legacy metrics API (`ragas.metrics.Faithfulness`, `SingleTurnSample`, …) is **deprecated in 0.4 and removed in 1.0**, replaced by the collections-based API (`ragas.metrics.collections`) — every metric page carries this banner. RAGAS also ships agent metrics (tool-call accuracy, tool-call F1, agent goal accuracy, topic adherence) and a knowledge-graph testset generator (see §2.2) ([metrics index](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)). Original framework paper: [Ragas, arXiv:2309.15217](https://arxiv.org/abs/2309.15217) (reference-free evaluation motivation).

### 1.2 DeepEval

- **Faithfulness**: extract claims from `actual_output`, then a claim is truthful "if it **does not contradict** any facts presented in the `retrieval_context`" — categorically weaker than RAGAS's positive-inference requirement. Knobs: `threshold` (default 0.5), `strict_mode` (binary), `truths_extraction_limit`, `penalize_ambiguous_claims`, custom `evaluation_template`; the metric is "self-explaining" (returns a reason). ([docs](https://docs.confident-ai.com/docs/metrics-faithfulness); prompt template in [source](https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/faithfulness/template.py))
- Their FAQ positions faithfulness as a **generator** signal and Contextual Recall/Precision as **retrieval** signals ([same page, FAQs](https://docs.confident-ai.com/docs/metrics-faithfulness)).
- DeepEval also supports **component-level evaluation on traces** (`@observe` + `update_current_span`), i.e. metrics attached to retrieval/generation spans ([same page, "Within components"]).

### 1.3 TruLens — the RAG Triad

Three judges, each along one edge of the RAG loop: **context relevance** (is each retrieved chunk relevant to the query), **groundedness** (split response into claims, independently search the retrieved context for supporting evidence), **answer relevance** (does the response address the input). Satisfying all three bounds hallucination "up to the limit of its knowledge base". ([RAG Triad page](https://www.trulens.org/getting_started/core_concepts/rag_triad/)). Note the word choice: TruLens "groundedness" ≈ RAGAS "faithfulness". TruLens vendors report groundedness F1 = 0.81 on LLM-AggreFact against human holdout labels ([TruLens README](https://github.com/truera/trulens) — vendor-reported, see §"Unverified").

### 1.4 MLflow

- Legacy `mlflow.evaluate` shipped `precision_at_k`, `recall_at_k`, `ndcg_at_k` (retrieval), plus LLM judges `answer_similarity`, `answer_correctness`, `answer_relevance`, `relevance`, `faithfulness`. **Deprecated as of MLflow 3.0**; removal expected around 3.7.0. ([migration guide in mlflow source](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx))
- MLflow 3 suite: `mlflow.genai.evaluate(predict_fn=..., data=[{inputs, outputs, expectations}], scorers=[...])`; built-in judges, custom judges via `make_judge(name, instructions, feedback_value_type)`, code scorers via `@scorer`; **latency and token counts are recorded by traces**, no longer separate metrics ([same migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)). Every prediction gets a trace for root-cause analysis ([eval-monitor index](https://github.com/mlflow/mlflow/tree/master/docs/docs/genai/eval-monitor)).

### 1.5 Classic IR metrics (BEIR convention)

BEIR evaluates with **NDCG@k, MAP@k, Recall@k, Precision@k at k=[1,3,5,10,100,1000]** (pytrec_eval under the hood) and MRR via `retriever.evaluate_custom(..., metric="mrr")`; 17 preprocessed datasets, TREC runfiles saved alongside JSON results for reranking/comparison. ([BEIR repo + quick example](https://github.com/beir-cellar/beir)). These are deterministic, judge-free, and the only layer where framework-agnostic comparability is guaranteed.

### 1.6 Cross-framework takeaway

The same *name* can mean different computations: RAGAS faithfulness (positive entailment) vs DeepEval faithfulness (no contradiction) vs TruLens groundedness (per-claim evidence search) vs MLflow faithfulness (judge prompt of the day). A benchmark that wants external credibility must (a) pin exact definitions + judge model + prompts per version, (b) always co-report the judge-free ID-based layer, and (c) never average LLM-judge scores across different judge configs.

---

## 2. Benchmark design best practices

### 2.1 Golden datasets

- LangSmith: **start with 5–10 (ideally 10–20) manually curated examples** that define "good"; extend with historical traces (negative feedback, heuristics, LLM-flagged) and synthetic data as a supplement. Datasets support **splits** (train/validation/test or category-based) and automatic **versioning** so CI can pin a version. ([Evaluation concepts](https://docs.smith.langchain.com/evaluation))
- TruLens: for judge calibration specifically, golden sets must include clear successes, clear failures, borderline cases, and production slices, with **frozen development/validation/held-out splits** created once with a fixed seed and never recomputed ([Judge Alignment](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)).
- CRAG shows what an end-to-end answer-level protocol looks like: human-defined ratings *perfect / acceptable / missing / incorrect*, automated as rule-based matching + LLM assessment scoring **correct = +1, missing = 0, incorrect = −1** — i.e., penalizing confident wrong answers rather than just detecting refusals ([CRAG repo](https://github.com/facebookresearch/CRAG), [paper arXiv:2406.04744](https://arxiv.org/abs/2406.04744)).

### 2.2 Synthetic test set generation (RAGAS testset generator; "evolution" patterns)

- Current RAGAS generation is **knowledge-graph based**: documents → hierarchical chunk nodes (custom splitters) → LLM/rule **extractors** (NER, keyphrases) → **relationship builders** (e.g., Jaccard over entities) → transforms build the KG; queries are then synthesized from graph traversal. A "scenario" is a combination of **nodes × query length × query style × persona**, and query types are the cross-product of **single-hop vs multi-hop** and **specific vs abstract** ([RAGAS testset generation docs](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)).
- The earlier "question evolution" generation (simple/multi-context/reasoning/conditional evolutions) from RAGAS ≤0.1 is documented in the original paper ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)); the exact evolution taxonomy could not be re-verified against current docs (see §"Could not verify").
- LangSmith's guidance: synthetic generation "works best when starting with several high-quality, hand-crafted examples as templates" — i.e., seed-and-extend, not generate-and-trust ([Evaluation concepts](https://docs.smith.langchain.com/evaluation)).
- "RAG Toast" appears to be the name of a demo app in a TruLens test-set-generation write-up, not a framework concept; no primary source could be located (see §"Could not verify").

### 2.3 Multi-hop evaluation

- RAGAS defines multi-hop queries as requiring "information from two or more sources" and manufactures them from related KG node pairs ([testset docs](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)).
- Established multi-hop datasets remain HotpotQA (BEIR variant: 7,405 test queries, 2.0 relevant docs/query; gold evidence as title+sentence pairs — [BEIR dataset table](https://github.com/beir-cellar/beir), [hotpotqa.github.io](https://hotpotqa.github.io)) and, in KILT format, HotpotQA/NQ/TriviaQA with **paragraph-level provenance** as first-class evaluation targets ([KILT repo](https://github.com/facebookresearch/KILT)).
- NVIDIA's production blueprint treats multi-hop as an **agentic** mode (plan-and-execute retrieval with sub-tasks and optional verification) alongside the standard chain ([NVIDIA RAG Blueprint](https://github.com/NVIDIA-AI-Blueprints/rag)) — implying multi-hop eval sets should be reported separately from single-hop, not averaged.

### 2.4 Judge LLM calibration and position bias

Primary protocol (TruLens "LLM Judge Alignment"): treat the judge as a measured system — define target, build representative golden set, freeze splits, run baseline, diagnose (absolute error, ranking, calibration, threshold, score distribution), change **one** dimension at a time, compare **paired** results on the same rows, confirm on held-out data; record the complete judge configuration (provider, model, temperature, parser, examples), and "do not define success with the judge under test" ([page](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)). Backing research:

- MT-Bench: GPT-4-class judges reach >80% agreement with humans (≈ human-human level), but exhibit **position, verbosity and self-enhancement biases** with limited reasoning ability; mitigations include swapping response positions in pairwise comparison and few-shot prompting ([arXiv:2306.05685](https://arxiv.org/abs/2306.05685)).
- Alignment varies substantially by model, dataset, property and annotator expertise across 20 NLP tasks ([Bavaresco et al., arXiv:2406.18403](https://arxiv.org/abs/2406.18403), cited via [TruLens alignment page](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)).
- Judges that do well on ordinary preference data can fail on pairs requiring factual/logical/mathematical correctness ([JudgeBench, arXiv:2410.12784](https://arxiv.org/abs/2410.12784), cited via TruLens).
- Explicit evaluation criteria and evaluation steps in the judge prompt improve human correlation ([G-Eval, arXiv:2303.16634](https://arxiv.org/abs/2303.16634)); task-specific criteria can be calibrated from human-labeled examples ([AutoCalibrate, arXiv:2309.13308](https://arxiv.org/abs/2309.13308)) — both cited via the TruLens alignment page.
- Noisy/biased single judges can be replaced by a **jury ensemble** of diverse models (TruLens `Jury`, [reference](https://www.trulens.org/reference/trulens/feedback/jury/), linked from the alignment page).
- RAGBench's empirical finding is the sharpest caveat: LLM-based RAG evaluators "struggle to compete with a finetuned RoBERTa model" on the RAG-evaluation task ([arXiv:2407.11005](https://arxiv.org/abs/2407.11005)).

### 2.5 Statistical significance and fair comparison

- BEIR's SIGIR 2024 refresh argues that collapsing heterogeneous datasets into a single average "is difficult to interpret" and provides **meta-analyses based on effect sizes** plus reproducible reference implementations ([Resources for Brewing BEIR, arXiv:2306.07471 / DOI 10.1145/3626772.3657862](https://dl.acm.org/doi/10.1145/3626772.3657862)).
- Paired design is the default for A/B: TruLens mandates "run both variants over the same rows" ([alignment page](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)); LangSmith structures this as multiple experiments on the same dataset with side-by-side comparison, and regression tests that "assert new versions must outperform baseline versions on relevant metrics" ([Evaluation concepts](https://docs.smith.langchain.com/evaluation)).
- Fairness controls when comparing systems: identical corpus, dataset, retrieval budget (top-k), generator and judge across all systems; identical metric set. (This is the convention CRAG encodes by running all systems through the same `local_evaluation.py` path — [CRAG repo](https://github.com/facebookresearch/CRAG) — and it is what this repo's MCP fairness flag already enforces.)

---

## 3. Benchmark datasets/suites — status and access

| Suite | What it is | Access (first-party) | Status (as of research date) |
|---|---|---|---|
| **BEIR** | 15+/17 heterogeneous zero-shot IR datasets (MSMARCO, NQ, HotpotQA, FiQA, FEVER, SciFact, …) | `pip install beir`; zip downloads from UKP server; also on [HF: BeIR](https://huggingface.co/BeIR); metrics NDCG/MAP/Recall/P@k + MRR ([repo](https://github.com/beir-cellar/beir)) | Maintained; recent examples include LoRA+vLLM and Cohere API retrieval; leaderboard wiki updated by the SIGIR'24 "Brewing BEIR" effort |
| **MTEB / MMTEB** | Massive (multilingual, now multimodal) embedding benchmark incl. retrieval tasks; benchmark variants like `MTEB(eng, v2)` | `pip install mteb`, CLI `mteb run -m <model> -t <task>`; leaderboard on [HF Spaces](https://huggingface.co/spaces/mteb/leaderboard) ([repo](https://github.com/embeddings-benchmark/mteb)) | Very active; rebranded "Multimodal toolbox for evaluating embeddings and retrieval systems"; papers [arXiv:2210.07316](https://arxiv.org/abs/2210.07316), [arXiv:2502.13595](https://arxiv.org/abs/2502.13595) |
| **RAGBench + TRACe** | 100k RAG examples across 5 industry domains with *explainable* (actionable) TRACe labels (utilization/relevance/adherence-style labels) | HF dataset [`rungalileo/ragbench`](https://huggingface.co/datasets/rungalileo/ragbench) ([paper](https://arxiv.org/abs/2407.11005), v2 Jan 2025) | Paper revised 2025; dataset hosted on HF; TRACe labels make it usable for training/validating your own evaluators |
| **KILT** | 5 knowledge-intensive task types (fact checking, entity linking, slot filling, ODQA, dialogue) with **provenance** evaluation against a single Wikipedia knowledge source | JSONL downloads from `dl.fbaipublicfiles.com/KILT/…`, 34.76 GiB knowledge source, HF `datasets` mirror, EvalAI leaderboard ([repo](https://github.com/facebookresearch/KILT); [paper](https://arxiv.org/abs/2009.02252)) | Benchmark stable but tooling frozen (README still says conda python=3.7); use the data, not the library |
| **CRAG** | Factual QA benchmark: 5 domains, 8 question categories, entity popularity + temporal dynamism axes; **mock web/KG search APIs**; +1/0/−1 auto-eval | GitHub repo with `local_evaluation.py` and mock APIs ([repo](https://github.com/facebookresearch/CRAG); [paper](https://arxiv.org/abs/2406.04744)) | Challenge-era repo (migrated from KDD Cup 2024 AiCrowd); dataset still the reference for end-to-end RAG scoring; license CC BY-NC 4.0 |
| **HotpotQA** | Multi-hop QA with supporting-fact supervision | [hotpotqa.github.io](https://hotpotqa.github.io); BEIR-formatted `hotpotqa.zip`; KILT-formatted JSONLs | Stable; gold evidence (`supporting_facts`) remains the standard substrate for multi-hop retrieval metrics |

Practical note: for a RAG benchmark, the combinations that matter are **BEIR (single-hop retrieval, judge-free)** + **HotpotQA/MuSiQue/2Wiki-style multi-hop with supporting-facts** + **RAGBench/TRACe (domain-flavored, judge calibration)**; KILT/CRAG add provenance and end-to-end answer scoring protocols respectively.

---

## 4. Emerging trends (2025–2026)

### 4.1 Agentic RAG evaluation
NVIDIA's RAG Blueprint ships an **agentic mode** (LangGraph plan-and-execute: scope discovery, parallel sub-tasks, synthesis, optional verification) next to the classic chain, plus a separate latency/throughput benchmark track ([repo](https://github.com/NVIDIA-AI-Blueprints/rag)). Implication: benchmarks increasingly need to score *reasoning/plan quality*, not just final answers.

### 4.2 Tool-call evaluation
RAGAS exposes agent metrics: **tool call accuracy, tool call F1, agent goal accuracy, topic adherence** ([metrics index](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)). TruLens ships seven agentic evaluators — LogicalConsistency, ExecutionEfficiency, PlanAdherence, PlanQuality, **ToolSelection, ToolCalling, ToolQuality** — and a dedicated **MCP span type** capturing tool name, arguments, output and latency ([README](https://github.com/truera/trulens)).

### 4.3 Tracing-based and online evaluation ("OpenInference-style")
- Phoenix: export trace spans → map span attributes to evaluator inputs → run evals (built-in templates: relevance, correctness, faithfulness, summarization, toxicity) → **log results back as span annotations**; judge-model config is decoupled from evaluator config ([Phoenix evals guide](https://arize.com/docs/phoenix/evaluation)). (OpenInference is Arize's open tracing-semantics project behind this: [github.com/Arize-ai/openinference](https://github.com/Arize-ai/openinference) — not fetched this session, see §"Could not verify".)
- LangSmith splits evaluation into **offline** (datasets + reference outputs: benchmarking, regression, unit tests) and **online** (production runs/threads: monitoring, anomaly detection), with rules attaching evaluators to tracing projects at a sampling rate with spend limits ([Evaluation concepts](https://docs.smith.langchain.com/evaluation)).
- MLflow 3 records a trace per evaluated prediction and derives latency/token metrics from it ([migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)).
- TruLens is OpenTelemetry-native — spans portable to any OTLP backend — with both inline (guardrail-style) and batch (Run API) evaluation ([README](https://github.com/truera/trulens)). It even documents evaluating MLflow traces directly ([component guide](https://www.trulens.org/component_guides/evaluation/mlflow/)).

### 4.4 LLM-judge best practices (rubrics, pairwise)
Consolidated practice across sources: explicit rubric/criteria + evaluation steps in the prompt ([G-Eval](https://arxiv.org/abs/2303.16634)); pairwise comparison when absolute scoring is hard (summarization example in [LangSmith concepts](https://docs.smith.langchain.com/evaluation), [MT-Bench](https://arxiv.org/abs/2306.05685)); **position swapping** in pairwise judging ([MT-Bench](https://arxiv.org/abs/2306.05685)); self-explaining scores (DeepEval `include_reason`, [docs](https://docs.confident-ai.com/docs/metrics-faithfulness)); few-shot examples for weaker judges ([LangSmith](https://docs.smith.langchain.com/evaluation)); ensemble juries ([TruLens `Jury`](https://www.trulens.org/reference/trulens/feedback/jury/)); calibration against human golden sets with frozen splits ([TruLens alignment](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)).

### 4.5 Cost/latency tradeoffs
- Latency and token counts are now first-class trace attributes rather than metric functions (MLflow 3, [migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)); TruLens records per-step latency/cost in the trace ([README](https://github.com/truera/trulens)).
- Cheap replacements for judge calls: RAGAS non-LLM context precision/recall (Levenshtein/rapidfuzz) and ID-based metrics ([docs](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)); Vectara HHEM-2.1-Open classifier for faithfulness verification ([RAGAS faithfulness page](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)); fine-tuned small classifiers beat LLM judges on RAG-eval tasks ([RAGBench](https://arxiv.org/abs/2407.11005)).
- Online evaluation needs cost governance: LangSmith attaches sampling and spend limits per evaluator attachment ([concepts](https://docs.smith.langchain.com/evaluation)).

---

## 5. Implications for this repo

Findings mapped to `benchmark/`, `config.py`, `evaluation.py`, `tracking.py`/MLflow:

1. **RAGAS legacy API is a ticking clock** — `benchmark/evaluation.py:8-16` imports `from ragas.metrics._faithfulness import faithfulness` etc. (legacy API). RAGAS docs state this API is deprecated in 0.4 and removed in 1.0 ([source](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/)). Action: pin the RAGAS version in requirements now; plan a migration to the collections API (`ragas.metrics.collections`); keep metric definitions unchanged by adding a regression test comparing old/new scores on a fixed fixture.
2. **Document judge-dependence per run** — since faithfulness semantics differ across frameworks ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/) vs [DeepEval](https://docs.confident-ai.com/docs/metrics-faithfulness)), `tracking.py:_make_tags()` (benchmark/tracking.py:99) should also tag `critic_llm_model`, critic prompt/version and RAGAS version, so MLflow runs are comparable. `evaluation.py` already sets the `evaluation.critic_model` span attribute (benchmark/evaluation.py:65-67) — lift the same info into MLflow run tags.
3. **Lead with judge-free metrics; use judges as a second layer** — `gold_retrieval_metrics.py` (hit@k, nDCG, MRR-style) matches the BEIR protocol and RAGAS ID-based metrics ([BEIR](https://github.com/beir-cellar/beir), [RAGAS ID-based](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/)). Recommendation: make the gold-ID layer the headline number in reports for sweeps, and gate expensive RAGAS judge metrics behind final-candidate runs (the small default metric set + `RAGAS_ENABLED` smoke flag in `evaluation.py:141-158` is exactly the right instinct).
4. **Add paired significance testing** — per-sample scores already exist (`EvaluationResult.per_sample_scores`, evaluation.py:187-201) and `scripts/compare_runs.py` already pairs per-question scores. Following BEIR's SIGIR'24 guidance (effect sizes over single averages, [arXiv:2306.07471](https://dl.acm.org/doi/10.1145/3626772.3657862)) and TruLens's paired-rows rule ([alignment page](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)): add paired bootstrap or Wilcoxon signed-rank + effect size to `tools/ci_plots.py` (it already accepts a `paired` dict) and log p-value/effect size as MLflow metrics in `tracking.py`.
5. **Dataset governance like LangSmith/TruLens** — keep the dataset name+sample in run tags (already in `_make_tags`) but add a dataset **version/checksum tag** and frozen split labels (dev/validation/held-out) so judge tuning never touches held-out data ([LangSmith splits/versioning](https://docs.smith.langchain.com/evaluation), [TruLens golden sets](https://www.trulens.org/component_guides/evaluation/llm_judge_alignment/)). The multi-hop gold derivation in `gold_retrieval_metrics.py` (supporting_facts → gold set) is the right substrate for a HotpotQA/MuSiQue split.
6. **Answer-level scoring: adopt CRAG's +1/0/−1 shape** — `custom_metrics.py` already detects refusals (`is_refusal_answer`, custom_metrics.py:60). CRAG additionally *penalizes incorrect answers* (−1) while scoring "missing" as 0 ([CRAG repo](https://github.com/facebookresearch/CRAG)); add a `correct/incorrect/missing` categorization metric so "confidently wrong" systems don't look equal to "honest refusals".
7. **MLflow 3 migration watch** — the repo uses `mlflow` tracing + manual logging (tracking.py), not the deprecated `mlflow.evaluate`; that is the durable path. When MLflow ≥3.7 drops the legacy API nothing breaks here, but the new `mlflow.genai.evaluate` + `make_judge` pattern ([migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)) is a good fit for a future "judge experiment" mode (comparing critic models on the same run traces) without touching the core pipeline.
8. **External adapters: require doc IDs in the response schema** — RAGAS's ID-based context precision/recall and BEIR qrels both assume IDs ([RAGAS](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/), [BEIR](https://github.com/beir-cellar/beir)). `benchmark/adapters/http.py` already maps `metadata`/`doc_id` fields; make the doc-ID field a first-class, validated adapter requirement (fall back to content-hash IDs only if the external system cannot expose provenance, and flag it in `adapter_diagnostics` which `main.py` already collects).
9. **MCP/tool-call benchmarking is becoming table stakes** — TruLens defines an MCP span type and tool-call evaluators ([README](https://github.com/truera/trulens)); RAGAS defines tool-call accuracy/F1 ([metrics index](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)). `benchmark/adapters/mcp.py` should record per-call tool name/arguments/output/latency into the QA log so future tool-call metrics have data without another protocol change.
10. **Synthetic testsets: add as a generator, not a source of truth** — a RAGAS-style KG-based generator over the repo's own corpora ([docs](https://docs.ragas.io/en/stable/concepts/test_data_generation/rag/)) would extend the dataset matrix cheaply, but per LangSmith guidance synthetic examples should be seeded from curated goldens and tagged with a `synthetic=true` split so they can be filtered out of headline results ([LangSmith](https://docs.smith.langchain.com/evaluation)); RAGBench's finding on LLM-judge weakness ([arXiv:2407.11005](https://arxiv.org/abs/2407.11005)) argues for keeping gold-ID metrics authoritative.

---

## Could not verify against primary sources

- **"RAG Toast"** — no primary source found; it appears to be a demo-app name from a secondary blog write-up about test-set generation. Not used as a factual anchor above.
- **RAGAS "question evolution" taxonomy** (simple/multi-context/reasoning/conditional) — described in the original RAGAS paper/old docs era ([arXiv:2309.15217](https://arxiv.org/abs/2309.15217)); the exact taxonomy was not re-verified this session (current docs only describe the KG-based pipeline).
- **TruLens/Snowflake vendor benchmark numbers** (groundedness F1 0.81 on LLM-AggreFact, context-relevance NDCG@5 0.93) — first-party README claims ([README](https://github.com/truera/trulens)) backed by a Snowflake engineering blog, not independently verified here.
- **Exact list of MLflow 3 built-in retrieval judges** — the migration table references them but the target page was not fetched; the legacy `precision_at_k/recall_at_k/ndcg_at_k` mapping is verified ([migration guide](https://github.com/mlflow/mlflow/blob/master/docs/docs/genai/eval-monitor/legacy-llm-evaluation.mdx)).
- **OpenInference spec** — cited as background ([github.com/Arize-ai/openinference](https://github.com/Arize-ai/openinference)); repo not fetched this session.
- **Azure AI Evaluation (Microsoft Learn)** — listed as a candidate source but not consulted; no claims from it appear in this report.
- **HotpotQA native statistics** — cited via the BEIR README table and the KILT catalogue rather than hotpotqa.github.io itself.
