# OCR-08: Stage gate — test-suite verification + docs (no benchmark runs)

**What to build:** Proof end to end via the automated test suite only — no real benchmark runs (the user runs those themselves): worker-level tests with a stub parser and fixture GT producing known scores, plus a corpus_parser RAG test with a stub parser; experiment manifests for the three v1 datasets; docs updated (parser integration guide, README env-var table, EVAL_MATRIX note).

**Blocked by:** OCR-06, OCR-07.

**Status:** ready-for-agent

- [ ] Full pytest passes (RAGAS/custom metrics off; no network, no model downloads, no paid API calls)
- [ ] Parser benchmark test: stub parser, fixture GT subset → run dir (temp), fake MLflow tracker, leaderboard report with expected scores
- [ ] corpus_parser test: parsed corpus flows through the RAG pipeline and evaluates with a stub parser
- [ ] Experiment manifests for OmniDocBench/DP-Bench/olmOCR-Bench exist with license flags
- [ ] Parser integration guide + README env-var table updated
