# OCR-06: Parser evaluation runner through the worker

**What to build:** A parser benchmark run driven by the existing worker loop: manifest declares parser(s) × document dataset(s); the worker executes cells resumably (checkpoint per document), produces a leaderboard-style report (per-parser, per-category, per-page scores: text metrics, TEDS, overall), and logs everything to the run dir and MLflow like any other benchmark.

**Blocked by:** OCR-02, OCR-03, OCR-04, OCR-05.

**Status:** ready-for-agent

- [ ] Manifest → config matrix expansion for parser experiments (follows the matrix conventions)
- [ ] Resumable per-document checkpoints; rerun continues where it stopped
- [ ] Report: per-parser × per-category score table + per-page details in the run dir; MLflow parent/child run hierarchy matches the RAG-run pattern
- [ ] Matching-algorithm version, parser identity/version, and dataset license flags present in run metadata
- [ ] Worker-level tests with a stub parser and tiny fixture GT; fake tracker, temp run dirs
