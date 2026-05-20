# Research Improvements

Academic-grade enhancements for paper quality and research rigor.

## 1. Statistical Rigor

### Confidence Intervals
No confidence intervals on any metric. Every reported number is a point estimate.

**Fix:** Bootstrap 95% CI for all metrics. Report as `mean [CI_low, CI_high]`.

### Significance Testing
No statistical tests between configurations. Cannot claim one config is better than another.

**Fix:**
- Paired bootstrap test for metric differences
- Wilcoxon signed-rank test (non-parametric, appropriate for non-normal metric distributions)
- Bonferroni correction for multiple comparisons (128+ configs)
- Report p-values alongside effect sizes (Cohen's d)

### Sample Size Justification
No power analysis. N=100 chosen arbitrarily.

**Fix:**
- Compute minimum sample size for desired statistical power (e.g., 80% power to detect 5% difference)
- Plot metric stability curves (at what N do metrics stabilize?)
- Current data can answer this: re-score subsets of the 100 samples

### Effect Size Reporting
No effect sizes. Statistical significance != practical significance.

**Fix:** Report Cohen's d or Cliff's delta for every pairwise comparison.

## 2. Validity Threats

### Internal Validity
- **Evaluator bias:** RAGAS metrics depend on critic LLM. Different critic = different scores. Run sensitivity analysis with multiple critics.
- **Data leakage:** Ensure ground truth never appears in retrieved context. Add automated leakage detection.
- **Order effects:** Configs run sequentially. GPU thermal throttling may bias later runs. Randomize order or cool down between runs.

### External Validity
- **Dataset generalization:** Results on SQuAD may not generalize. Test on 3+ datasets.
- **Model generalization:** Tested 3 LLMs. Need more model diversity.
- **Temporal validity:** Model weights update. Pin exact model versions and dates.

### Construct Validity
- **Metric validity:** Do RAGAS metrics actually measure what they claim? Cross-validate with human judgments.
- **Ground truth quality:** SQuAD ground truths are extractive spans, not comprehensive answers. May penalize abstractive but correct answers.

## 3. Reproducibility

### Environment Tracking
Missing: Python version, package versions, GPU model/driver, CUDA version, OS, random seeds.

**Fix:** Log full environment snapshot (`pip freeze`, `nvidia-smi`, `python --version`). Store in each run's results.

### Determinism
No seed control. Results may vary between identical configs.

**Fix:** Set `random.seed()`, `numpy.random.seed()`. Log whether results are deterministic.

### Experiment Cards
No structured experiment documentation.

**Fix:** Adopt experiment cards recording hypothesis, config, result, conclusion for each run.

## 4. Analysis Improvements

### Ablation Studies
EVAL_MATRIX tests combinatorial configs but no systematic ablation.

**Fix:** One-factor-at-a-time ablation isolating effect of each parameter.

### Interaction Effects
No analysis of parameter interactions.

**Fix:** Two-way ANOVA or interaction plots for key pairs (chunk_size x retrieval_strategy, template x llm_model).

### Metric Correlation Analysis
12+ metrics computed but no correlation analysis.

**Fix:** Pairwise correlation matrix. Identify redundant metrics. Validate `ndcg@5`, `context_relevance`, `bert_score_f1 + meteor` as key lenses.

### Failure Mode Analysis
No categorization of failure modes.

**Fix:** Taxonomy: (1) Retrieval failure, (2) Generation failure, (3) Evaluation failure, (4) Hallucination. Quantify per config.

## 5. Paper-Specific Improvements

### Baselines
No baselines compared against:
- **No-retrieval baseline:** LLM answers without context
- **Random retrieval baseline:** Random chunks (tests if retrieval works)
- **Oracle retrieval baseline:** Always correct chunks (upper bound)

### Visualization
Need paper-quality figures: radar charts, heatmaps, box plots, Pareto fronts.

### Related Work
Compare against published RAG benchmark results (RAGAS paper, TruLens, SQuAD RAG baselines).
