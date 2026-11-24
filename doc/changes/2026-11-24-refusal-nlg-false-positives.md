# Refusal check no longer zeroes NLG metrics of genuine answers

Date: 2026-11-24

## Summary
`is_refusal_answer()` matched broad substrings ("insufficient", "provided
context", "does not provide", "no information", …) and set ROUGE-L, BLEU,
METEOR and BERTScore of genuine answers to 0 (issue #57). It now uses the
refusal phrases and "the context does not contain …" patterns of
`benchmark/answer_match.py` on the answer's first sentence. Re-scored from the
stored samples, BERT F1 of run7/run8 rises by 0.07–0.14 and ROUGE-L by
0.02–0.03 per config; in run7 the chunk-size ranking of BERT F1 mostly
disappears (spread 0.06 → 0.03).

## Changes

### Refusal detection (`benchmark/answer_match.py`)
- New public `states_refusal(answer)`: empty answer, unambiguous refusal phrase
  (`_REFUSAL_PHRASES`) or source-subject pattern (`_REFUSAL_PATTERNS`).
- `declines()` = `states_refusal()` or an ambiguous phrase ("error", "does not
  exist", "there is no", …) in an answer of at most 30 words. Behaviour
  unchanged.

### NLG skip (`benchmark/custom_metrics.py`)
- `is_refusal_answer(text)` = `states_refusal(<first sentence of text>)`.
- Why not `declines()` as the issue proposed: its ambiguous short-answer rule
  was built for the MCP trust question set and zeroes short technical answers
  such as "According to dataset-doc-250, this error occurs … because the … table
  view does not exist." (0–1 answers per 500 in run7).
- Why the first sentence only: `states_refusal()` also counts a source caveat
  after an answer ("To resolve the issue, upgrade to Firefox 38.5 ESR … The
  provided documents do not contain a specific fix."). Of 2 868 run7/run8
  answers it flags, 2 629 lead with the refusal ("The provided context does not
  contain …", median 63 words) and 239 are answers with a trailing caveat. A
  refusal leads; an answer with a caveat keeps its NLG scores.

### Re-scoring script (`scripts/rescore_nlg_refusals.py`)
- Reads `configs/*.samples.json.gz` of a run, recomputes the NLG metrics of
  samples the old check zeroed (stored `rouge_l == 0`) and the new one does not
  (BLEU with the default `max_n=4`, BERTScore on CPU; BERT fields stay `None`
  when the run had no BERT model), zeroes samples the new
  check counts as refusals, keeps all other stored scores.
- Writes old and new per-config means to `<run_dir>/nlg_rescore_issue57.json`;
  the run's own files stay untouched (`results/` is append-only).
- `python scripts/rescore_nlg_refusals.py results/Enterprise-RAG/run8
  [--bert-model roberta-large]` (run7/run8 used roberta-large).

## Re-scored results (DGX Spark)

run8 (500 TechQA questions, rec 2048/64, k 10):

| Refinement | α | rescored | newly zeroed | BERT F1 old → new | ROUGE-L old → new |
|---|---|---|---|---|---|
| on | 0.5 | 47 | 1 | 0.587 → 0.675 | 0.151 → 0.170 |
| on | 0.3 | 49 | 1 | 0.583 → 0.675 | 0.147 → 0.168 |
| off | 0.5 | 58 | 0 | 0.643 → 0.755 | 0.171 → 0.197 |
| off | 0.3 | 69 | 0 | 0.624 → 0.757 | 0.168 → 0.199 |

run7: see `doc/enterprise-rag-blueprint-results.md` (run7 table, columns
"BERT F1 (fixed)" / "ROUGE-L (fixed)") and
`results/Enterprise-RAG/run7/nlg_rescore_issue57.json`.

Reading: the gap between refinement off and on grows from 0.04–0.06 to 0.08
BERT F1, since refinement-off configs had more genuine answers zeroed. With
refinement off, α 0.3 now edges α 0.5 on BERT F1 (0.757 vs 0.755) instead of
trailing it.

## Behaviour changes and migration
- New runs report higher NLG means; compare against old runs only through
  `nlg_rescore_issue57.json` (run7, run8) or a re-score with the script.
- `declines()` / `declined_rate` unchanged.
- `scripts/analyze_blueprint_ablation.py` uses `is_refusal_answer()`; its
  `is_refusal` column now follows the new rule, so re-runs differ from earlier
  printed results.
- The re-score uses the stored answers and `custom_scores`; MLflow metrics of
  run7/run8 keep the old values.

## Verification
- `tests/test_custom_metrics_refusal.py` (new): four genuine answers from
  run7/run8 keep ROUGE-L 1.0 against themselves; empty answer, explicit
  refusals and a refusal followed by an explanation score 0.
- `tests/test_answer_match.py` unchanged and green.
- Full suite on the DGX Spark: 1034 passed.

## Follow-ups
- A leading sentence matching `according to the retrieved context, there is
  no …` still counts as a refusal although "there is no limit to the number of
  items …" is an answer (1 question in run8, 2 of its configs).
- Only the first sentence is checked: a refusal in the second sentence ("The
  question asks about X. However, the provided context does not contain …")
  keeps its NLG scores, and the split `(?<=[.!?])\s+` ends a sentence early at
  abbreviations such as "e.g. ". A hand check of 8 random answers among the 239
  trailing-caveat answers found only genuine answers; the reverse error was not
  measured.
- Earlier runs (run6 and before) were not re-scored.
- The re-scored NLG means live only in `nlg_rescore_issue57.json` on the DGX
  Spark (run7/run8 results are not in git), not in MLflow.
