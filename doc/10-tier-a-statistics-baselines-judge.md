# Tier A Implementation: Statistics, Baselines, Datasets, Judge

What was added, how it changes results, and what has to change in your setup.

## 1. Statistical significance testing (`benchmark/statistics.py`)

**What it does.** For every pair of configs in a sweep, the framework now
computes a paired bootstrap confidence interval and permutation test for each
metric, from the per-sample scores you already collect. No model calls —
pure post-processing on stored numbers.

**New files / changes**

- `benchmark/statistics.py` — `bootstrap_ci`, `paired_bootstrap_delta_ci`,
  `paired_permutation_pvalue`, `cohens_d`, `paired_comparison`,
  `compare_per_sample` (defaults: 10,000 resamples, 95% CI, seed 42)
- `benchmark/reporting/comparisons.py` — after every sweep, `generate_report`
  writes two new files into `results/runN/`:
  - `comparisons.json` — machine-readable: baseline config, per-metric
    `{delta, ci_low, ci_high, p_value, effect_size, reliable}`
  - `comparisons.md` — human-readable table, each metric row ending in a
    verdict: **real** (CI excludes 0 and p < 0.05) or **noise**
- `tests/test_statistics.py`, `tests/test_comparisons_report.py`

**How results change.** The first config in your experiment acts as the
baseline; every other config gets a delta vs. baseline per metric. A "+0.023"
that used to look like a win now reads `+0.023 [0.008, 0.039] p=0.004 → real`
or `+0.012 [-0.009, 0.031] p=0.21 → noise`. Nothing about the benchmark runs
themselves changes; statistics never break a run (failures are printed and
skipped).

**Rules of interpretation**

- Comparisons are *paired* (both configs answer the same questions in the
  same order) — that is why noise from question difficulty cancels out.
- Fewer than 3 valid paired samples for a metric → metric is skipped.
- With very small N the permutation test has a resolution floor: at N=4 the
  smallest possible two-sided p is 0.125, so nothing can ever be "real".
  Use N ≥ 20, ideally your usual 100.

**Setup changes:** none. Runs automatically; no env vars.

## 2. Baselines triangle (`benchmark/adapters/baselines.py`)

Three new `RAG_SYSTEM_ADAPTER` values, all using the framework's generator
and evaluator so they are directly comparable with `internal`:

| Adapter | Meaning | What it gives you |
| --- | --- | --- |
| `no_retrieval` | closed-book LLM, empty context | retrieval *lower bound*: parametric knowledge only |
| `random_retrieval` | top_k documents drawn at random from the corpus (seeded per question → deterministic) | sanity floor: a system scoring below this is worse than chance |
| `oracle_retrieval` | the gold context of the question | retrieval *upper bound*: `oracle − internal` = remaining headroom |

Registered in the adapter registry; ready-to-run manifest:
`mcp-example/experiments/baselines-triangle.yaml`. `random_retrieval`
requires a shared corpus (`dataset.corpus_path` or a `has_shared_corpus`
dataset). Tests: `tests/test_baseline_adapters.py`.

**How results change.** New rows appear in the summary as ordinary configs.
The headline claim becomes computable: `internal − no_retrieval` isolates the
contribution of retrieval; `oracle − internal` shows how much better
retrieval could still get on this dataset.

**Setup changes:** none new — set `rag_system_adapter` in the matrix.

## 3. New datasets (`benchmark/dataset_adapters.py`)

- `hotpotqa` — multi-hop QA (`hotpot_qa`, config `distractor` by default,
  validation split). Contexts are the 10 gold+distractor paragraphs; the
  corpus is deduplicated like SQuAD (`has_shared_corpus`), so gold-doc
  retrieval metrics and `oracle_retrieval` work. `supporting_facts` is kept
  in sample metadata for supporting-fact gold metrics.
- `nq_open` — Natural Questions open (`google-research-datasets/nq_open`,
  validation split). Ships **no context**: it is a closed-book-friendly
  dataset, ideal together with `no_retrieval`. List answers are joined with
  `|`. For retrieval benchmarks on NQ you must supply a corpus
  (`DATASET_NAME=jsonl-shared` with your own `corpus_path`).

Both follow the SQuAD adapter pattern and are selected with
`DATASET_NAME` / `dataset.name`. Tests: `tests/test_new_dataset_adapters.py`.

**Setup changes:** `dataset.subset` defaults to `distractor` for HotpotQA;
the first run downloads the dataset from HuggingFace.

## 4. Full RAGAS metrics + LLM-as-judge

**RAGAS preset** — `evaluation.py` gained `metric_preset`:

- `core` (default, unchanged): faithfulness, context_recall,
  semantic_similarity
- `full`: adds `answer_relevancy`, `answer_correctness`, `context_precision`.
  These land in the existing `ragas_*` columns that were previously always
  empty, plus stats columns. Expect ~2–3× critic-LLM cost per sample.

Configured via `RAGAS_METRIC_PRESET` (env) or `eval_ragas_preset` (YAML).

**LLM-as-judge** — new `benchmark/llm_judge.py`: every answer is scored on a
4-criterion rubric (accuracy, completeness, relevance, honesty) by the critic
LLM, twice — with the rubric criteria in opposite orders.

- `llm_judge_score` — mean of both judgments, 0–1 (merged into custom
  metrics: appears in `custom_*` columns, per-sample CSV, and comparisons)
- `llm_judge_order_bias` — |score₁ − score₂|; a judge-robustness diagnostic.
  If this is large relative to real quality differences, your judge's verdicts
  depend on prompt wording, and judge-based claims are weak.

Configured via `LLM_JUDGE_ENABLED=true` and `LLM_JUDGE_LLM` (defaults to
`EVAL_CRITIC_LLM`), or `llm_judge_enabled` / `llm_judge_llm` in YAML.
Judge failures are recorded per sample and never abort a run. Tests:
`tests/test_llm_judge.py`, `tests/test_evaluation_presets.py`.

**Setup changes (`.env`)**

```bash
RAGAS_METRIC_PRESET=full      # optional: all six RAGAS metrics
LLM_JUDGE_ENABLED=true        # optional: judge scoring
LLM_JUDGE_LLM=gemma3:12b      # optional: separate judge model
```

All three default to the previous behavior (`core` preset, judge off), so
**existing workflows and past results remain comparable** — nothing changes
until you opt in.

## Cost impact summary

| Feature | Extra LLM calls per sample | Extra runtime work |
| --- | --- | --- |
| Statistics | 0 | ~1 s CPU per config pair |
| Baselines | same as an `internal` run (one generation per sample) | random/oracle skip embedding+indexing → cheaper than internal |
| RAGAS `full` | ~2–3× critic calls | — |
| LLM judge | 2 judge calls | — |

## Suggested first run

```bash
BENCHMARK_CONFIG_FILE=mcp-example/experiments/baseline-internal.yaml python main.py
BENCHMARK_CONFIG_FILE=mcp-example/experiments/baselines-triangle.yaml python main.py
```

Then read `results/runN/comparisons.md`: the internal-vs-baseline deltas with
CI/p-value verdicts are exactly the "what does retrieval contribute" numbers
for the paper.
