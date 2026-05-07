# Known Risks and Gaps

Project risks observed from current code and docs:

- External services and models are central. Ollama, OpenAI-compatible endpoints, Hugging Face datasets, RAGAS, MLflow, and optional tracing can fail independently.
- RAGAS and model output parsing are brittle with verbose reasoning models; see [[Generation Layer]] and optimization notes about parser issues.
- Vector-store caching is powerful but can hide stale data if fingerprint coverage misses a behavior-affecting input.
- Agentic exploration depends on local model tool-call quality; fallback heuristics reduce but do not remove this risk.
- Benchmark results are expensive and may be statistically noisy for small sample sizes.
- `EVAL_MATRIX.md`, `results/`, MLflow, and paper claims can drift unless synchronized.
- Runtime artifacts are mixed into the repo tree. Be careful with cleanup and git status.
- The agentic workflow duplicates main benchmark orchestration and currently lacks some `main.py` behavior such as shared-corpus loading and full reporting/tracing.
- Some helper scripts depend on current JSON shapes or MLflow SQLite internals.
- Prompt-specific answer cleanup is tuned for reasoning/numeric models and can distort open-form tasks if used carelessly.
- Some older optimization/audit notes are stale; see [[Known Stale Docs]] before relying on them.

Documentation gaps to keep reducing:

- Exact `.env` examples and known-good model combinations.
- More explicit mapping from report metrics to paper claims.
- Clearer ownership of generated artifacts versus source-controlled docs.
- Smoke-test instructions that do not require large local models.

Related notes:

- [[Operations Runbook]]
- [[Testing and Coverage]]
- [[Research Notes]]
- [[Known Stale Docs]]
