# Fix declined_rate false positives on unlabelled datasets

Date: 2026-11-16 · Issue: #18

## Summary
`declined_rate` reported 58 % refusals on RAGBench TechQA (run6), while the real
refusal rate is ~33–38 %. The phrase list built for the MCP trust question set
matched words such as "error" and "failed" in normal technical answers. The
heuristic now separates unambiguous refusals from ambiguous words, and
datasets without decline labels no longer report the phrase-list rate.

## Changes

### Decline heuristic (`benchmark/answer_match.py`, `declines()`)
- Always a decline: empty answer, refusal phrases ("cannot answer", "cannot
  provide", "no information", …) and "the source does not cover it"
  statements whose subject is the retrieved material ("the provided context
  does not contain …", "… is not mentioned in the documents", "based on the
  provided context, there is no …").
- Ambiguous phrases from the old list ("error", "failed", "unknown",
  "permission", "unable to", "not found", "there is no", …) count only in
  answers of at most 30 words. Short trust-set declines keep matching.

### Decision: unlabelled datasets (`score_answer()`, `benchmark/mcp_host.py`)
- `declined` (phrase list) is set only for labelled samples: those with
  `expected_behavior` or `answer_patterns`. Free-text datasets such as TechQA
  get no `declined` label, so `declined_rate` is omitted from the run summary
  and per-category entries.
- The decline judge (`enterprise_rag_mcp_decline_judge: true`) still scores
  every sample; `declined_judge_rate` is the refusal measure for unlabelled
  datasets. `decline_judge_agreement_rate` uses samples that carry both labels.

Why omit instead of reporting the corrected heuristic: the phrase list is a
heuristic tuned on labelled question sets and cannot be validated on arbitrary
domains; the judge is the general measure and costs one 8-token critic call
per sample.

## Behaviour changes and migration
- TechQA / RAGBench / other unlabelled runs: `declined_rate` disappears from
  `adapter_metrics`. Enable `enterprise_rag_mcp_decline_judge` for a refusal
  rate. Older results are not rescored.
- MCP trust question set: all 210 samples carry `expected_behavior`, so
  `declined_rate` is still reported. Declines are short and keep matching;
  long answers (> 30 words) that only mention an ambiguous word are no longer
  declines, which can lower `declined_rate` and the `answer_match` of decline
  samples that relied on such a word. No trust-run samples were available
  locally to quantify this.
- MultiHop-RAG (labelled): on `results/Enterprise-RAG/multihop/run2` the rate
  drops from 270 to 249 of 1000; declines on null queries (expected decline)
  rise from 101 to 102 of 117, so `answer_match` of decline samples is
  practically unchanged.

## Verification
- New tests in `tests/test_answer_match.py`: TechQA false positives (error /
  failed / permission answers), real refusals ("The provided context does not
  contain …"), short trust-style declines, unlabelled samples without
  `declined`. New test in `tests/test_mcp_host.py`: `declined_rate` omitted,
  `declined_judge_rate` kept.
- `pytest`: 874 passed.
- Rescored run6 answers (`results/Enterprise-RAG/run6`, n = 500) with the new
  `declines()`: 167 declines (33.4 %, old 286 / 58 %); reference answers
  flagged 26 (old 225); long answers flagged only because of "error": 0.

## Follow-ups
- Run the MCP trust question set once with the judge on and compare
  `declined_rate` with `declined_judge_rate` to confirm the 30-word threshold.
