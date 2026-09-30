# Experiment TODOs

What to test next with the existing experiment manifests. Smoke tests first, full runs after.
For all smoke runs: `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false` and a small `DATASET_SAMPLE_SIZE`.

## 1. Smoke tests (fast, cheap, verify wiring)

- [ ] `experiments/optimaiserag-northstar-smoke.yaml` — 5-question connectivity + contract check (worker `plan` + `run`).
- [ ] `experiments/enterprise_rag_demo.yaml` — 5 samples, internal-style pipeline with enterprise adapter; verify MLflow logging path.
- [ ] `experiments/external_rag_demo.yaml` — 20 samples via `demo_python` plugin adapter; verify custom retrieval metrics in heuristic mode.
- [ ] `experiments/full-grid-example.yaml` — only `plan` (matrix expansion + resumability check), no full run.
- [ ] Resume check: interrupt a `worker run`, restart with the same `--run-dir`, confirm no duplicate work.

## 2. OCR parser benchmarks (newest feature, OCR-01…08)

- [ ] `experiments/ocr_parser_dp_bench.yaml` — point `DATASET_PATH` at a local DP-Bench export, set `PARSER_HTTP_ENDPOINT_URL`, verify CER/WER + TEDS metrics and GT↔prediction alignment.
- [ ] `experiments/ocr_parser_omnidocbench.yaml` — same flow with OmniDocBench layout_dets export.
- [ ] `experiments/ocr_parser_olmocr_bench.yaml` — verify `gt_supported=false` rows are skipped correctly.
- [ ] Parser plugin adapter (not just `http`): register a local parser plugin and rerun one manifest.
- [ ] `corpus_parser` ingestion knob: build a RAG corpus through a parser adapter (OCR-07) and run a downstream RAG experiment on it.

## 3. RAG-vs-MCP comparisons

- [ ] `experiments/rag-vs-mcp.yaml` — full comparison with fairness validation enabled (default).
- [ ] `experiments/rag-vs-mcp-answer.yaml` — answer-only comparison.
- [ ] `experiments/rag-vs-mcp-agentic.yaml` — agentic variant.
- [ ] One intentionally asymmetric run with `MCP_ENFORCE_FAIRNESS=false` to confirm the fairness check actually fires.

## 4. Internal pipeline sweeps (full-grid)

- [ ] Chunking sweep: recursive vs. semantic, chunk sizes 256/512/1024, overlap 0/64/128 on one dataset.
- [ ] Retrieval sweep: top-k 3/5/10, with and without reranker, dense-only vs. hybrid (if enabled).
- [ ] Vector store comparison: chroma vs. lancedb on identical corpus + embeddings.
- [ ] Dataset sweep: squad vs. ragas-wikiqa vs. ragbench at fixed pipeline settings (generalization check).
- [ ] Multi-hop: run a multihop-capable dataset (see `doc/multihop-retrieval.md`) against the best config from the sweeps.

## 5. Evaluation quality

- [ ] Enable all RAGAS metrics (answer_relevancy, answer_correctness, context_precision) on one small run; verify they land in reports + MLflow.
- [ ] Custom metrics spot-check: hit@k, nDCG@k, ROUGE-L, BLEU on known-input fixture data (no tests exist yet — validate by hand, see `doc/04-testing-gaps.md`).
- [ ] Verify `llm_answer_strip_mode: tags_only` + `value_fallback` against a thinking-model output.

## 6. Reporting & tracking

- [ ] After the first full run: check `results/runN/` exports (JSON/CSV/Markdown), plots, and MLflow run/log records.
- [ ] Run twice into the same `--run-dir` and confirm reports reflect both configs.

## Suggested commands

```bash
source .venv/bin/activate
python -m benchmark.worker plan experiments/<name>.yaml
python -m benchmark.worker run experiments/<name>.yaml --keep-going --run-dir results/runN
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false DATASET_SAMPLE_SIZE=5 \
  python -m benchmark.worker run experiments/optimaiserag-northstar-smoke.yaml --run-dir results/smoke
```
