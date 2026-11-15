# optimAIseRAG Benchmark (2/6): Local Pre-Flight

Date: 2026-11-13

Closes aiengineeringblueprints/RAG-NIMBF#27.

## Summary

Executed the fully offline pre-flight for the optimAIseRAG Northstar
benchmark: dataset validation, reproducibility check, and `benchmark.worker
plan` for both manifests. All acceptance criteria passed; no provider call
was made. No code or manifest changes were required — the manifests, dataset,
and matrix were already sound.

## Changes

- Documentation only: this session record. No framework code, manifests, or
  dataset files were modified.

## Behaviour changes and migration

None.

## Verification

Commands (see `doc/OptimiseRAG_Benchmark.md` § 3), all on branch
`feat/27-local-preflight`:

| Check | Command | Result |
| --- | --- | --- |
| Dataset validation | `python datasets/northstar_facilities/validate.py` | `status: pass`, 360 questions, 60 documents, 0 errors |
| Reproducibility | `python datasets/northstar_facilities/generate.py --check` | `reproducibility check passed: 360 questions, 60 sources` |
| Smoke plan | `python -m benchmark.worker plan experiments/optimaiserag-northstar-smoke.yaml` | `num_configs: 1`, `total_questions: 5` (`qwen2.5:7b`, `BAAI/bge-m3`) |
| Full plan | `python -m benchmark.worker plan experiments/optimaiserag-northstar.yaml` | `num_configs: 1`, `total_questions: 360` |

- `worker plan` expands the matrix locally only — no HTTP calls to the
  optimAIseRAG deployment occurred (the deployment is currently not
  reachable; see issue #26 status).
- Full test suite: `pytest` → 829 passed.

Note on the full plan: 1 configuration is the expected count. The manifest
deliberately carries a single matrix entry per model/chunking/reranker; the
parameter sweep (108 configurations) is planned as a separate ticket
(#25, item 5) using a copied sweep manifest.

## Follow-ups

- Issue #26 (deployment readiness) is still blocked: no RAGFlow endpoint is
  reachable and `OPTIMAISE_RAG_API_KEY` is not exported. The retrieval smoke
  (issue #28) cannot start before that is resolved.
- The OpenAI-compatible gateway at `optimaise.ddnss.de` serves
  `nvidia/Qwen3.8-27B-NVFP4` / `nvidia/Qwen3.6-35B-A3B-NVFP4` only — neither
  `qwen2.5:7b` nor `BAAI/bge-m3`. If benchmarking should target that host,
  the manifest matrix and deployment model config must be aligned first.
