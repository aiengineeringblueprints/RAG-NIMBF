# Custom MLflow answer_correctness judge

Date: 2026-11-17

## Summary
Adds `answer_correctness` (issue #45, parent #40) to the post-run MLflow
scorers. MLflow does not wrap RAGAS `answer_correctness`, and the built-in
`Correctness` judge only checks that expected facts are present, so wrong
extra content goes unpunished. The new judge approximates RAGAS semantics:
the critic LLM classifies claims, the framework computes a claim F1 that
penalises both missing facts and wrong/unsupported added claims.

## Changes

### Config
- `config.py`: `answer_correctness` added to `MLFLOW_SCORER_NAMES`, so it is
  selectable via `mlflow_scorers` (YAML list/string) and `MLFLOW_SCORERS`.

### Scorer
- `benchmark/mlflow_scorers.py`:
  - `ANSWER_CORRECTNESS_INSTRUCTIONS`: `make_judge` prompt over
    `{{ inputs }}`, `{{ outputs }}`, `{{ expectations }}` that splits answer
    and reference into atomic claims at the same granularity (a place and a
    date are two facts), lists them in the rationale, and returns
    `{"true_positives", "false_positives", "false_negatives"}`
    (`feedback_value_type=dict[str, int]`), judged only against the reference.
  - `ClaimCounts` (TP/FP/FN) and `claim_f1(counts)` =
    `TP / (TP + 0.5 * (FP + FN))` (RAGAS factuality F1, beta = 1), computed in
    Python, not by the LLM.
  - `answer_correctness_scorer(claims_judge)` wraps the judge into an MLflow
    scorer named `answer_correctness`: numeric 0–1 value, rationale
    `TP=…, FP=…, FN=…` plus the judge's claim lists. A missing
    `expected_response` (judge not called), judge errors, malformed,
    non-whole-number (e.g. `1.7`, `"2"`, `true`, NaN) or negative counts and
    an all-zero verdict become `Feedback.error` (excluded from the
    mean) instead of a fake score.
  - Registered in `SCORER_FACTORIES`; the judge uses the critic endpoint and
    `eval_critic_temperature` / `eval_critic_max_tokens` like the built-ins.
- Logged like every scorer: `mlflow_judge_answer_correctness_mean` on the
  config run, per-trace assessments with rationale.

### Manifests and docs
- New `experiments/answer-correctness-smoke.yaml` (`correctness` vs.
  `answer_correctness`, local `ollama:gemma3:12b` critic).
- `doc/results-and-mlflow.md`: scorer table row and an `answer_correctness`
  section (formula, differences from RAGAS); `doc/env-vars.md`
  (`MLFLOW_SCORERS` names); `doc/00-README.md` index.

## Behaviour changes and migration
None. New opt-in scorer name; existing selections are unchanged.

## Verification
- Tests (`tests/test_mlflow_scorers.py`): fixed Marie-Curie examples through a
  stub claim judge — complete answer (1.0), missing fact (2/3), wrong added
  content (0.8), wrong answer (0.0), rationale with counts; malformed judge
  output, no claims, and a missing reference → error (judge not called for
  the latter); strict whole-number counts; an end-to-end test proving
  errored traces are excluded from `mlflow_judge_answer_correctness_mean`; factory wires `make_judge` to the critic model and
  inference params; end-to-end `score_run_traces` on a traced run logs
  `mlflow_judge_answer_correctness_mean` and passes trace ground truth to the
  judge. `tests/test_config.py`: name is selectable.
- Full suite on the DGX Spark: 1007 passed, 3 failed — the same 3 fail on
  `dev` without this change (`test_hybrid_defaults` picks up the Spark `.env`,
  `test_worker_pipeline_runs_parsed_corpus_unchanged`, and the order-dependent
  `test_internal_and_stubbed_external_runs_are_structurally_identical`).
- Live on the Spark, critic `ollama:gemma3:12b` (temperature 0), reference
  "Marie Curie was born in Warsaw in 1867.":

  | Answer | `correctness` | `answer_correctness` |
  |---|---|---|
  | complete | yes | 1.0 (TP=2) |
  | "She was born in Warsaw." | no | 0.67 (TP=1, FN=1) |
  | complete + "born in France, three Nobel Prizes" | **yes** | **0.67** (TP=1, FP=1) |
  | "Paris in 1901" | no | 0.0 (FP=2, FN=2) |

- Worker smoke on the Spark: `experiments/answer-correctness-smoke.yaml`,
  `--run-dir results/run98` (first prompt) and `results/run99` (final code),
  3 SQuAD questions each: run tagged
  `mlflow_scorers=logged`, `mlflow_judge_answer_correctness_mean = 1.0`,
  `mlflow_judge_correctness_mean = 1.0`; each trace carries an
  `answer_correctness` assessment with a `TP=1, FP=0, FN=0` rationale.

## PR review follow-up
PR #54 review (approved) noted that `int()` silently truncated fractional or
string/bool counts and that exclusion of errored assessments from the mean was
unverified. Counts are now validated strictly (`_count`), and the exclusion is
covered by `test_answer_correctness_errors_are_excluded_from_the_run_mean`.

## Follow-ups
- Claim granularity still depends on the judge model: with the first prompt
  gemma3:12b counted the two-fact reference as one claim; the final prompt
  fixes that case, but in the "wrong added" case it still merges two wrong
  claims into FP=1. Absolute values are therefore not directly comparable to
  direct `ragas_answer_correctness`; compare configs judged by the same
  critic. Splitting extraction and classification into separate calls (as
  RAGAS does) would be the next step if this proves too noisy.
- Prompt behaviour is verified only live on the Spark; unit tests stub the
  judge and cover the scoring logic and wiring.
- No semantic-similarity term (would need an embedding model in MLflow).
