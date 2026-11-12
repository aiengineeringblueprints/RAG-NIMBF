# LLM answer judge for stored runs

Date: 2026-11-12

## Summary
The deterministic `answer_match` scored MultiHop-RAG comparison and temporal
queries at 8–15 % although the answers were mostly right: the references are
"Yes"/"no"/"Consistent", the Blueprint answers in prose, and 484 of 569 such
answers contain neither "yes" nor "no". A critic-LLM judge now re-scores the
stored answers of finished runs as correct, incorrect or declined, without a
new benchmark run.

## Changes
- `benchmark/answer_match.py`: `make_answer_judge(chat_model)` returns an
  `AnswerJudge` `(question, reference, answer) -> "correct" | "incorrect" |
  "declined" | None`. One critic call per answer covers correctness and
  declines; `None` on an unparseable reply or a failed call.
- `benchmark/reporting/answer_judging.py`: `judge_runs(run_dirs, dataset,
  judge, cache_path=, max_workers=)` reads run directories (read-only), judges
  each distinct (question, answer) pair once and reports per variant and
  category `judged`, `judge_correct`, `judge_incorrect`, `judge_declined` next
  to the stored `answer_match`. Samples with `expected_behavior: decline` are
  correct when judged declined. Verdicts are appended to a JSONL cache as they
  arrive, so an interrupted run resumes; unparseable verdicts are retried.
- `scripts/judge_answers.py`: CLI using `EVAL_CRITIC_LLM` and its endpoint from
  `.env` (`--critic` overrides, `--workers` defaults to
  `EVAL_CRITIC_MAX_WORKERS`).
- `datasets/multihop_rag/README.md`, `experiments/enterprise-rag-multihop-rag.yaml`:
  evaluation commands.

## Behaviour changes and migration
None. Runs and `answer_match` are unchanged; the judge only runs offline. The
existing decline judge (`enterprise_rag_mcp_decline_judge`) is untouched.

## Verification
- Tests: `tests/test_answer_match.py` (verdict parsing, prompt content, failed
  call), `tests/test_answer_judging.py` (rates per variant/category, decline
  samples, dedupe and cache, unparseable verdicts). Full suite: 794 passed.
- MultiHop-RAG run1 on the DGX Spark (1000 questions, 4 configs,
  `openai:deepseek-v4-flash`, 8 workers, 6 min):

  | variant | comparison | inference | temporal | null | ALL | `answer_match` ALL |
  |---|---|---|---|---|---|---|
  | similarity k10 | 0.577 | 0.799 | 0.502 | 0.838 | 0.660 | 0.439 |
  | similarity k20 | 0.670 | 0.904 | 0.635 | 0.838 | 0.755 | 0.448 |
  | hybrid-a0.5 k10 | 0.622 | 0.885 | 0.541 | 0.795 | 0.706 | 0.441 |
  | hybrid-a0.5 k20 | 0.756 | 0.908 | 0.588 | 0.838 | 0.774 | 0.454 |

  (`judge_correct` per category.)
- Judge sanity checks: on inference queries (entity references, where pattern
  matching is reliable) judge and `answer_match` agree on 88 % of answers; in
  sampled disagreements the judge was mostly right (e.g. an answer naming the
  wrong person while mentioning the reference in passing). Yes/no references
  are judged correct at 65 % (yes) vs 56 % (no), so no strong polarity bias.

## Follow-ups
- Validate the judge on a hand-labelled sample (~50 yes/no answers) before
  quoting the numbers in a publication.
- Optional: run the judge inside the adapter for future runs.
- Persist `ground_truth_doc_ids` in the samples of jsonl-shared runs (currently
  empty; gold metrics fall back to the dataset).
