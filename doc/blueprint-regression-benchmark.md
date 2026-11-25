# Blueprint Regression Benchmark

A fixed, bounded evaluation for iterative changes to the Enterprise RAG
Blueprint (issue #64). Instead of re-running a full configuration sweep, every
change is a **candidate** that differs from a reference in one component and
is judged on a **versioned question set** against explicit improvement and
trade-off criteria.

| Piece | Where |
|---|---|
| Versioned question set (dev + holdout) | `datasets/blueprint_regression/v1/manifest.json` (split files are rebuilt, not committed) |
| Immutable reference baseline | `experiments/blueprint-regression/baseline.yaml` |
| Current best configuration | `experiments/blueprint-regression/current-best.yaml` |
| Example candidate (similarity → hybrid) | `experiments/blueprint-regression/candidate-hybrid.yaml` |
| Smoke test | `experiments/blueprint-regression/smoke.yaml` |
| Improvement / trade-off criteria | `experiments/blueprint-regression/criteria.yaml` |
| Freeze, verify, compare | `python -m benchmark.regression` (`benchmark/regression.py`) |

## Question set v1

Built from RAGBench TechQA `train` with the loader's seeded shuffle
(`shuffle(seed=42)`), the same selection runs 7–9 used:

| Split | Size | Answerable | Unanswerable | sha256 of the split file |
|---|---|---|---|---|
| `dev` | 500 | 452 | 48 | `b7a4707d…f63c` |
| `holdout` | 200 | 182 | 18 | `7fd22497…9cd8` |

- `dev` is the first 500 shuffled rows — **identical to the questions of
  `results/Enterprise-RAG/run7..run9`** (checked against run8: same 500
  question/reference pairs), so those runs can be compared directly.
- `holdout` takes the following rows whose question text is not in `dev`.
  It is reserved for the final confirmation of a candidate (see below).
- Categories: `answerable` / `unanswerable` from RAGBench's
  `all_relevant_sentence_keys` (empty = the documents do not support an
  answer). They are stored as `metadata.category` in each row.
- Corpus snapshot: the documents travel with each question (`context`), so
  the corpus of a run is the context set of its split. The files' hashes in
  the manifest pin questions, references, labels *and* corpus.
- **TechQA quirk**: every question appears twice with the same documents but a
  different reference answer, and 47 such pairs even carry different
  answerable labels. Samples are therefore identified by question **and**
  reference answer (`benchmark.regression.sample_keys`), never by question
  text alone.

### Rebuild and verify (once per machine)

Only `manifest.json` is committed (the split files are ~30 MB, mostly
documents, and gitignored). Rebuilding is deterministic; `freeze` refuses to
write files whose hashes differ from the committed manifest:

```bash
python -m benchmark.regression freeze --dataset ragbench --subset techqa \
  --split train --dev-size 500 --holdout-size 200 --slack 491 \
  --out datasets/blueprint_regression/v1
python -m benchmark.regression verify --question-set datasets/blueprint_regression/v1
```

`--slack` loads extra rows to replace holdout rows that repeat a dev question.
The total (`dev + holdout + slack` = 1191) must stay **below** the split size
(1192): the HF loader skips the seeded shuffle when the request covers the
whole split, and `freeze` aborts in that case.

A new question set is a new version directory (`v2`), never an edit of `v1`;
`freeze` refuses to overwrite a directory that already holds split files.

## Configurations

`baseline.yaml` is the **immutable reference**: the issue's example
configuration on the Blueprint (`enterprise_rag_mcp` adapter, plain RAG):

| Setting | Value |
|---|---|
| Generator | `openai:qwen3.8-27b` (temperature 0, `concise` prompt) |
| Embedding | `nomic-embed-text:latest` |
| Chunking | `recursive`, size 1000, overlap 200 |
| Retrieval | `similarity`, top-k 12, fetch-k 50, no reranker, no HyDE, no multi-hop |
| Blueprint | query refinement on, guardrails on |
| Judge | `openai:deepseek-v4-flash`, embeddings `ollama:nomic-embed-text:latest` |

Never edit it once it has a run; a new reference is a new file with its own
run. `current-best.yaml` holds the best known configuration (run7 grid
winner: recursive 2048/64, top-k 10, hybrid alpha 0.5). It is replaced (new
file) only when a candidate is accepted on dev **and** confirmed on holdout.

Model versions: the reproducibility record of each run
(`<run>/reproducibility/manifest.json`) stores the full resolved config, git
commit and package versions. The served model weights behind a tag
(`qwen3.8-27b`, `nomic-embed-text:latest`) are not resolved to a digest by
the framework — record them with the run (e.g. `ollama show --modelfile`,
vLLM model path) when a model server changes.

## Protocol

1. Rebuild + `verify` the question set.
2. Run `baseline.yaml` and `current-best.yaml` once on `dev` (re-run only when
   the Blueprint code, model server or judge changes).
3. For each idea, copy `baseline.yaml` (or `current-best.yaml` when the idea
   builds on it) into `candidate-<name>.yaml` and change **one** component.
   `tests/test_regression_benchmark.py` checks that `candidate-hybrid.yaml`
   differs from the baseline only in the retrieval strategy.
4. Run the candidate with the same script and environment:

   ```bash
   scripts/run_blueprint_rag.sh results/Enterprise-RAG/<runN> \
     experiments/blueprint-regression/candidate-hybrid.yaml
   ```

5. Compare against both references:

   ```bash
   python -m benchmark.regression compare \
     --question-set datasets/blueprint_regression/v1 \
     --criteria experiments/blueprint-regression/criteria.yaml \
     --baseline <baseline run>/configs/<config>.json \
     --best <current-best run>/configs/<config>.json \
     --candidate <candidate run>/configs/<config>.json \
     --out results/Enterprise-RAG/<runN>/regression
   ```

   Report files (`<config>.json`) and their `<config>.samples.json.gz`
   sidecars are both accepted. Writes `comparison.json` and `report.md`.
   `compare` refuses runs that cover different questions or not the whole
   frozen split.
6. If two accepted changes may interact (e.g. hybrid retrieval and a larger
   top-k), evaluate the combination as its own candidate instead of assuming
   the gains add up.
7. Before promoting a candidate to `current-best`, run it and the reference on
   `holdout` (manifest with `path: datasets/blueprint_regression/v1/holdout.jsonl`,
   `sample_size: 200`) and compare with `--split holdout`. Repeated wins on
   `dev` are not independent evidence — many candidates are tried against the
   same 500 questions; holdout is used once per promotion.

Rescoring without regeneration: the per-sample answers and contexts stay in
`<config>.samples.json.gz`; `scripts/rescore_nlg_refusals.py` and
`scripts/judge_answers.py` re-score them without calling the generator.

## Evaluation dimensions

| Dimension | Metric (`compare` name) | Implementation | Labels | Judge |
|---|---|---|---|---|
| Correctness / completeness | `answer_correctness` (primary), `bert_score_f1` (fallback) | RAGAS answer correctness; BERTScore F1 vs reference | reference answer | critic LLM + embedding (RAGAS); none (BERTScore) |
| Faithfulness | `faithfulness` | RAGAS faithfulness of the answer against the retrieved contexts | none | critic LLM |
| Retrieval | `hit@k`, `recall@k`, `ndcg@k`, `relevant_sentence_recall`, `supporting_fact_coverage@k` | gold-document metrics (`custom_retrieval_metrics_mode: gold_doc`) | gold document IDs; RAGBench relevant sentence keys | none |
| Abstention | `abstained` (on unanswerable), `false_refusal` (on answerable) | `benchmark.answer_match.declines` on the answer | `metadata.category` | none |
| Latency | `total_seconds`, `ttft_seconds` | per-sample wall-clock answer time | — | — |
| Cost | `estimated_cost_usd`, `total_tokens` | adapter token usage × pricing | — | — |

- Retrieval metrics are computed on answerable questions with gold labels
  only; a dimension without data is reported as **unavailable** with the
  reason (e.g. RAGAS off, no unanswerable questions, adapter reports no
  tokens), never as 0.
- The Blueprint adapter reports neither tokens nor cost, so the cost
  dimension is currently unavailable; latency is the operational trade-off
  that is measured. `ttft_seconds` equals `total_seconds` (no streaming).
- `hit@k`, `recall@k` and `supporting_fact_coverage@k` coincide on TechQA
  (one gold document per question); they are kept because other datasets
  differentiate them.

## When is a candidate better?

`criteria.yaml` makes the rule explicit:

- **Primary**: the first metric of `primary` scored by both runs
  (`answer_correctness`, else `bert_score_f1`). For each metric `compare`
  reports the paired mean delta (candidate − reference), a paired-bootstrap
  95% CI (2000 resamples) and, for 0/1 metrics, an exact McNemar test.
- **accept**: CI of the primary delta above 0 **and** gain ≥
  `min_improvement` (0.02), **and** no guardrail breached.
- **reject**: CI entirely below 0, or a guardrail breached — the
  improvement does not outweigh the trade-off.
- **inconclusive**: anything else; keep the reference.
- **Guardrails** (worsening allowed at most, point estimate): faithfulness
  −0.03, `hit@5` −0.03, abstention on unanswerable −0.10, false refusals
  +0.02, `total_seconds` +25 %, cost +25 %. Guardrails whose metric is not
  scored in both runs are skipped and listed in the report.

## Verification runs

- **Smoke** (`smoke.yaml`, first 5 dev questions, RAGAS off) on the DGX
  Spark: runs end to end; correctness (`bert_score_f1` 0.576), retrieval
  (`hit@k`, `relevant_sentence_recall` 0.914) and latency (mean 1.52 s) are
  reported, faithfulness/abstention/cost correctly marked unavailable. Its
  corpus is only the 5 questions' documents, so its retrieval scores are not
  comparable with full runs.
- **Compare on existing data**: run8 cells (all on the dev questions) fed to
  `compare` — candidate query refinement **on** (`managed-a6b2f4cd80`) vs
  refinement off (`managed-2c9672c978`), both hybrid alpha 0.5:
  `bert_score_f1` −0.056 (CI −0.089..−0.018), false refusals +0.066
  (McNemar p < 0.001) → **reject**, consistent with the run8 analysis.

## Proposed follow-up candidates

From runs 5–9 (see [Enterprise RAG Blueprint Results](enterprise-rag-blueprint-results.md)):
hybrid retrieval vs similarity (`candidate-hybrid.yaml`), query refinement
off, larger chunks (2048/64), top-k 10 vs 12, a reranker, and a different
refusal prompt (false refusal rate ~19 % on answerable questions).
