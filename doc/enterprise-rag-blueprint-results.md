# Enterprise RAG Blueprint — Results Overview

Status: 2026-11-24. Source: `results/Enterprise-RAG/run1..run8` on the DGX Spark
(`~/source/Benchmarking-Framework`), manifest `experiments/enterprise-rag-blueprint-rag.yaml`
(run7: `enterprise-rag-gold-run7.yaml`, run8: `enterprise-rag-run8-ablation.yaml`).
Setup guide: [enterprise-rag-blueprint-setup.md](enterprise-rag-blueprint-setup.md).

All runs benchmark the Blueprint as **plain RAG (MCP off)** through the
`enterprise_rag_mcp` adapter (`inprocess` transport). The framework rebuilds the
Blueprint's Chroma index per config (chunker + embedder from the matrix); the
Blueprint retrieves and answers itself, with guardrails and query refinement on.

## Fixed settings (all runs)

| Setting | Value |
|---|---|
| Generator LLM | `qwen3.8-27b` (OpenAI-compatible), temperature 0, top_p 0.3 |
| Retrieval | similarity in run1–run5, hybrid (dense + BM25, RRF, alpha 0.5, fetch-k 50) since run6; `retrieval_top_k` 5 in run1–run4, 10 / 12 since run5 (= contexts per question), threshold 0.2, no HyDE / multihop / reranker |
| Vector DB | Chroma (Blueprint) |
| Blueprint flags | guardrails on, query refinement on, MCP off, role `General` |
| Prompt template | `concise` (not used by the Blueprint) |
| Retrieval metrics mode | `heuristic` in run1–run6 (no gold doc IDs, see caveats), `gold_doc` in run7 (RAGAS off, custom metrics on) |

## Runs and combinations

| Run | Date (UTC) | Dataset | n | Embedding | Critic LLM / critic embedding | Configs | Status |
|---|---|---|---|---|---|---|---|
| run1 | 2026-10-29 | SQuAD | 20 | nomic-embed-text | claude-opus-5-5 / nomic | rec 512/64 | 1/1 done (smoke) |
| run2 | 2026-10-29 → 30 | SQuAD | 500 | nomic-embed-text | deepseek-v4-flash / nomic | rec 512/64, 512/128, 1024/64, 1024/128, semantic p95 | 5/5 done |
| run3 | 2026-10-30 → 31 | RAGBench TechQA (train) | 500 | Qwen3-Embedding-4B | deepseek-v4-flash / Qwen3-Embedding-4B | same 5 | 4/5 done, semantic **failed** |
| run4 | 2026-10-31 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | same 5 | 3/5 done, rec 1024/128 **hung, stopped manually**, semantic not started |
| run5 | 2026-11-04 → 06 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | rec 512/1024/2048 × overlap 64/128 + semantic p95, each at top-k 10 and 12 (14) | 14/14 done |
| run6 | 2026-11-08 → 11 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | same 14 as run5, **hybrid retrieval** | 14/14 done (semantic k10 failed once, re-run 2026-11-11) |
| run7 | 2026-11-13 → 15 | RAGBench TechQA (train) | 500 | nomic-embed-text | critic off / nomic (RAGAS off) | same 14 as run5/6, **gold_doc retrieval metrics** | 14/14 done (both semantic configs failed twice on the Ollama runner batch bug, fixed + re-run 2026-11-14/15) |
| run8 | 2026-11-24 | RAGBench TechQA (train) | 500 | nomic-embed-text | critic off / nomic (RAGAS off) | rec 2048/64 k10 hybrid; **query refinement on/off × alpha 0.5/0.3** (4) | 4/4 done |
| run9 | 2026-11-24 → | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic (RAGAS on) | rec 2048/64 k10 hybrid α 0.5; refinement on/off (2) | running |

Combination matrix (✅ done, ❌ failed, ⏹ aborted, — not run):

| Chunking | SQuAD · nomic (run2, k5) | TechQA · Qwen3-Emb-4B (run3, k5) | TechQA · nomic (run4, k5) | TechQA · nomic (run5, k10 + k12) | TechQA · nomic (run6 + run7, hybrid, k10 + k12) |
|---|---|---|---|---|---|
| recursive 512 / 64 | ✅ (+ run1 smoke) | ✅ | ✅ | ✅ | ✅ |
| recursive 512 / 128 | ✅ | ✅ | ✅ | ✅ | ✅ |
| recursive 1024 / 64 | ✅ | ✅ | ✅ | ✅ | ✅ |
| recursive 1024 / 128 | ✅ | ✅ | ⏹ | ✅ | ✅ |
| recursive 2048 / 64 | — | — | — | ✅ | ✅ |
| recursive 2048 / 128 | — | — | — | ✅ | ✅ |
| semantic (percentile 95) | ✅ | ❌ | — | ✅ | ✅ (after Ollama embed-batching fix) |

Failure details:

- run3 semantic: `Post "http://127.0.0.1:41851/tokenize": ... connection reset by peer`
  — the Ollama runner for Qwen3-Embedding-4B died during semantic chunking.
- run4 recursive 1024/128: started 2026-10-31 15:37 UTC, no progress afterwards;
  the run was stopped and the summary JSON covers the 3 finished configs.
- run7 semantic k10 + k12 (first two passes): `Post "http://127.0.0.1:<port>/tokenize":
  connection reset by peer` — langchain_ollama sent the whole chunk list (3,530
  texts) as one `/api/embed` POST and Ollama's llama-server runner died during
  tokenization. Fixed in `04a171f6` (server-side 64-text batching, see
  `doc/changes/2026-11-14-resilient-ollama-embeddings.md`); single-text requests
  never failed (0/3,530).

## Results

### SQuAD, nomic-embed-text (run2, n = 500)

| Chunking | Faithf. | Ans. rel. | Ans. corr. | Ctx prec. | Ctx recall | Sem. sim. | Answer match | hit@1 | hit@5 | nDCG@5 | BERT F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rec 512/64 | 0.870 | 0.784 | 0.508 | 0.837 | 0.936 | 0.618 | 0.748 | 0.720 | 0.804 | 0.755 | 0.724 |
| rec 512/128 | 0.877 | 0.785 | 0.515 | 0.832 | **0.945** | 0.619 | **0.766** | **0.726** | 0.812 | 0.760 | 0.732 |
| rec 1024/64 | 0.913 | **0.793** | 0.531 | 0.839 | 0.931 | 0.623 | 0.760 | 0.672 | 0.760 | 0.706 | **0.794** |
| rec 1024/128 | **0.916** | 0.787 | 0.548 | **0.849** | 0.841 | **0.624** | 0.760 | 0.664 | 0.760 | 0.703 | 0.776 |
| semantic p95 | 0.896 | 0.774 | **0.566** | 0.828 | 0.826 | 0.619 | 0.752 | 0.722 | **0.822** | **0.763** | 0.747 |

Smoke run1 (rec 512/64, n = 20, critic Opus 5.5): faithfulness 0.590, answer
relevancy 0.834, correctness 0.523, context precision 0.908, context recall 0.950,
answer match 0.80. Different critic and n — not comparable to run2.

### RAGBench TechQA (n = 500): Qwen3-Embedding-4B (run3) vs nomic-embed-text (run4)

| Chunking | Embedding | Faithf. | Ans. rel. | Ans. corr. | Ctx prec. | Ctx recall | BERT F1 | Sem. sim.* | hit@5* |
|---|---|---|---|---|---|---|---|---|---|
| rec 512/64 | Qwen3-4B | 0.784 | 0.460 | 0.393 | 0.493 | 0.258 | 0.408 | 0.732 | 0.854 |
| rec 512/64 | nomic | 0.802 | 0.490 | 0.420 | 0.585 | 0.269 | 0.452 | 0.831 | 1.000 |
| rec 512/128 | Qwen3-4B | 0.773 | 0.485 | 0.363 | 0.529 | 0.284 | 0.439 | 0.724 | 0.868 |
| rec 512/128 | nomic | 0.770 | 0.487 | 0.448 | 0.565 | 0.282 | 0.446 | 0.832 | 1.000 |
| rec 1024/64 | Qwen3-4B | 0.810 | 0.505 | 0.399 | 0.527 | 0.372 | **0.489** | 0.727 | 0.910 |
| rec 1024/64 | nomic | **0.815** | **0.517** | **0.468** | **0.611** | 0.370 | 0.458 | 0.838 | 1.000 |
| rec 1024/128 | Qwen3-4B | 0.804 | 0.501 | 0.395 | 0.579 | **0.376** | 0.466 | 0.722 | 0.916 |
| rec 1024/128 | nomic | ⏹ | | | | | | | |

\* Embedding-dependent metric, scored with the run's own embedding model —
not comparable between run3 and run4 (see caveats).

### RAGBench TechQA, nomic-embed-text, top-k 10 / 12 (run5, n = 500)

Same generator, critic, critic embedding and dataset sample as run4; only
`retrieval_top_k` (5 → 10 / 12) and the chunking axes differ, so run4 and run5
are directly comparable. "Ctx chars" is the mean retrieved context length per
question.

| Chunking | k | Faithf. | Ans. rel. | Ans. corr. | Ctx prec. | Ctx recall | BERT F1 | Ctx chars | Refusals |
|---|---|---|---|---|---|---|---|---|---|
| rec 512/64 | 10 | 0.791 | 0.539 | 0.401 | 0.536 | 0.359 | 0.502 | 4 037 | 40 % |
| rec 512/128 | 10 | 0.811 | 0.507 | 0.422 | 0.605 | 0.422 | 0.502 | 4 097 | 38 % |
| rec 1024/64 | 10 | 0.830 | 0.532 | 0.478 | 0.619 | 0.525 | 0.489 | 8 251 | 39 % |
| rec 1024/128 | 10 | 0.820 | 0.532 | **0.513** | 0.623 | 0.541 | 0.516 | 8 315 | 39 % |
| rec 2048/64 | 10 | 0.816 | 0.520 | **0.513** | 0.599 | 0.581 | 0.504 | 16 705 | 36 % |
| rec 2048/128 | 10 | 0.841 | 0.527 | 0.486 | 0.602 | 0.578 | 0.512 | 16 758 | 37 % |
| semantic p95 | 10 | 0.826 | 0.512 | 0.493 | 0.582 | **0.606** | 0.520 | 35 893 | 38 % |
| rec 512/64 | 12 | 0.804 | 0.524 | 0.476 | 0.603 | 0.507 | 0.494 | 4 832 | 39 % |
| rec 512/128 | 12 | 0.837 | 0.522 | 0.482 | 0.621 | 0.502 | 0.475 | 4 929 | 38 % |
| rec 1024/64 | 12 | 0.850 | 0.524 | 0.484 | 0.595 | 0.577 | 0.516 | 9 929 | 38 % |
| rec 1024/128 | 12 | 0.843 | 0.535 | 0.488 | **0.627** | 0.564 | 0.520 | 10 005 | 38 % |
| rec 2048/64 | 12 | 0.843 | **0.549** | 0.498 | 0.601 | 0.603 | **0.525** | 19 958 | 35 % |
| rec 2048/128 | 12 | **0.853** | 0.546 | 0.494 | 0.581 | 0.605 | 0.514 | 20 109 | 37 % |
| semantic p95 | 12 | 0.842 | 0.543 | 0.497 | 0.572 | 0.604 | 0.523 | 42 768 | 34 % |

Refusals: share of answers matching a "context does not contain / cannot
answer" pattern (regex over the sample answers; run4 had 43–45 %). Heuristic
hit@k is ~0.99 for every config and omitted (see caveats).

Paired bootstrap (95 % CI, per-question differences) against rec 1024/128 k10
for answer correctness: rec 2048/64 k10 is tied (−0.002, CI [−0.019, +0.014]);
semantic k12 is borderline (−0.016, CI [−0.033, +0.001]); every other config
is significantly lower, rec 512/64 k10 (−0.114) and rec 512/128 k10 (−0.091)
by a wide margin.

### RAGBench TechQA: dense (run5) vs hybrid retrieval (run6), n = 500

run6 repeats run5 with `retrieval_strategy: hybrid` (dense + BM25 via weighted
RRF, `retrieval_hybrid_alpha` 0.5, `retrieval_fetch_k` 50, Blueprint branch
`feat/hybrid-retrieval` 8409d63). Everything else is identical (same 500
questions, generator, critic, critic embedding, chunking, top-k), so all
differences are paired per question. Values are run5 → run6; `*` marks a
paired-bootstrap 95 % CI that excludes 0. semantic p95 k10 failed in the first
run6 pass (Ollama `/tokenize` reset) and was re-run on 2026-11-11 into the same
run dir (separate summary `benchmark_20261111_180200.json` and a separate MLflow
parent run).

Two extra columns use the RAGBench labels (see
[enterprise-rag-techqa-refusal-analysis.md](enterprise-rag-techqa-refusal-analysis.md)):
**Gold@k** is the share of questions whose own document set (`dataset-doc-{i}`)
is among the retrieved chunks. **Rel. sent.** is the mean share of RAGBench
relevant sentences contained in the retrieved context. Refusals use the strict
refusal regex, not `declined_rate`.

| Chunking | k | Ans. corr. | Faithf. | Ctx prec. | Ctx recall | Refusals | Gold@k | Rel. sent. |
|---|---|---|---|---|---|---|---|---|
| rec 512/64 | 10 | 0.401 → 0.472* | 0.791 → 0.825* | 0.536 → 0.616* | 0.359 → 0.496* | 41.4 → 37.6 % | 0.840 → 0.910* | 0.342 → 0.383* |
| rec 512/128 | 10 | 0.422 → 0.476* | 0.811 → 0.825 | 0.605 → 0.637* | 0.422 → 0.514* | 38.8 → 39.2 % | 0.844 → 0.904* | 0.359 → 0.388* |
| rec 1024/64 | 10 | 0.478 → 0.490 | 0.830 → 0.863* | 0.619 → 0.641 | 0.525 → 0.588* | 40.8 → 34.4 %* | 0.834 → 0.922* | 0.508 → 0.568* |
| rec 1024/128 | 10 | 0.513 → 0.504 | 0.820 → 0.854* | 0.623 → 0.644 | 0.541 → 0.589* | 39.0 → 35.0 %* | 0.844 → 0.906* | 0.530 → 0.576* |
| rec 2048/64 | 10 | 0.513 → 0.502 | 0.816 → 0.851* | 0.599 → 0.634* | 0.581 → 0.643* | 38.6 → 34.4 %* | 0.866 → 0.938* | 0.628 → 0.712* |
| rec 2048/128 | 10 | 0.486 → 0.501 | 0.841 → 0.849 | 0.602 → 0.633* | 0.578 → 0.626* | 39.6 → 33.2 %* | 0.842 → 0.942* | 0.621 → 0.710* |
| semantic p95 | 10 | 0.493 → 0.475* (lower) | 0.826 → 0.852* | 0.582 → 0.616* | 0.606 → 0.670* | 38.0 → 32.8 %* | 0.858 → 0.964* | 0.680 → 0.819* |
| rec 512/64 | 12 | 0.476 → 0.487 | 0.804 → 0.841* | 0.603 → 0.608 | 0.507 → 0.512 | 41.4 → 37.6 % | 0.874 → 0.920* | 0.378 → 0.413* |
| rec 512/128 | 12 | 0.482 → 0.485 | 0.837 → 0.827 | 0.621 → 0.620 | 0.502 → 0.521 | 39.8 → 37.2 % | 0.868 → 0.914* | 0.387 → 0.416* |
| rec 1024/64 | 12 | 0.484 → 0.498 | 0.850 → 0.834 | 0.595 → 0.617 | 0.577 → 0.615* | 38.8 → 33.8 %* | 0.860 → 0.934* | 0.535 → 0.597* |
| rec 1024/128 | 12 | 0.488 → 0.496 | 0.843 → 0.851 | 0.627 → 0.642 | 0.564 → 0.604* | 38.4 → 35.4 % | 0.850 → 0.926* | 0.549 → 0.597* |
| rec 2048/64 | 12 | 0.498 → 0.512 | 0.843 → 0.852 | 0.601 → 0.623 | 0.603 → 0.664* | 36.8 → 31.8 %* | 0.896 → 0.960* | 0.648 → 0.740* |
| rec 2048/128 | 12 | 0.494 → 0.503 | 0.853 → 0.864 | 0.581 → 0.620* | 0.605 → 0.646* | 38.8 → 31.4 %* | 0.880 → 0.954* | 0.640 → 0.733* |
| semantic p95 | 12 | 0.497 → **0.523*** | 0.842 → 0.858 | 0.572 → 0.642* | 0.604 → **0.672*** | 34.6 → **31.2 %** | 0.872 → **0.966*** | 0.692 → **0.843*** |

semantic p95 k10 is the only config where answer correctness drops
significantly (−0.018) although every retrieval metric improves (Gold@k
+0.106, relevant-sentence coverage +0.140, refusals −5.2 pp) and answer
relevancy rises by +0.072. Context grows from 35 k to 44 k chars per question.

Pooled over the first 13 matched configs, without semantic k10 (paired; the
CIs treat question × config as independent and are therefore optimistic): answer correctness +0.018,
faithfulness +0.017, answer relevancy +0.019, context precision +0.030, context
recall +0.056, refusals −4.2 pp, Gold@k +0.071, relevant-sentence coverage
+0.066. Refusal flips: 719 refused → answered, 446 answered → refused. The
answerable subset (questions with RAGBench relevant sentences) shows the same
deltas.

Hybrid retrieval concentrates the chunks on fewer documents (distinct doc IDs
per question 5.6–9.1 → 5.0–8.2) and picks slightly longer chunks (+3–6 %
context chars for recursive, +26–27 % for semantic, 35 k → 44 k at k10 and
41 k → 52 k at k12), while
distinct context texts stay at ~k. Blueprint answering time rises from
641–863 s to 1,148–1,390 s per config for recursive chunking (~1.3 → ~2.4 s
per question), from 1,110 to 1,515 s for semantic k10 and from 1,225 to
1,690 s for semantic k12.


| Run | Index build (`adapter_prepare`) | Blueprint answering | RAGAS eval | Custom metrics | Total | Critic tokens |
|---|---|---|---|---|---|---|
| run2 SQuAD · nomic | 3–17 | 490–498 | 6 934–7 482 | 160–168 | 7 603–8 163 (~2.1–2.3 h) | 6.4–7.0 M |
| run3 TechQA · Qwen3-4B | 1 031–1 830 | 663–692 | 7 151–8 037 | 711–841 | 9 732–11 161 (~2.7–3.1 h) | 8.7–9.7 M |
| run4 TechQA · nomic | 83–151 | 618–641 | 7 205–7 890 | 216–225 | 8 201–8 852 (~2.3–2.5 h) | 8.8–10.6 M |
| run5 TechQA · nomic, k10/12 | 59–319 | 641–1 225 | 9 297–13 597 | 278–480 | 10 379–15 660 (~2.9–4.4 h) | 12.8–33.1 M |

RAGAS (critic LLM) dominates every config at ~85–90 % of the wall time. In
run5, larger chunks and semantic chunking raise Blueprint answering time
(median 1.3 s/question at 512 → 1.6–1.7 s at 2048 → 2.1–2.4 s semantic, p95 up
to 3.7 s) and critic tokens, because the context grows.

### RAGBench TechQA: gold-doc retrieval metrics (run7), n = 500

run7 repeats the run6 retrieval setup (hybrid, same corpus and generator) with
`custom_retrieval_metrics_mode: gold_doc` — hit@k / recall@k / nDCG@k are now
computed against the RAGBench gold document IDs instead of the heuristic.
RAGAS is off (`ragas_enabled: false`, issue 16 verification: no paid critic
calls), so answer-quality numbers below are critic-free custom metrics only.
Summary sources: `results/Enterprise-RAG/run7/configs/*.json` (final report
`benchmark_20261115_141853.json` covers only the last re-run config).

| Chunking | k | hit@1 | hit@5 | hit@10 | recall@10 | nDCG@10 (raw, invalid — see below) | Ctx. rel. | BERT F1 | ROUGE-L | BERT F1 (fixed) | ROUGE-L (fixed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rec 512/64 | 10 | 0.480 | 0.842 | 0.912 | 0.912 | 1.646 | 0.717 | 0.530 | 0.126 | 0.661 | 0.154 |
| rec 512/64 | 12 | 0.482 | 0.836 | 0.906 | 0.906 | 1.638 | 0.713 | 0.534 | 0.129 | 0.665 | 0.159 |
| rec 512/128 | 10 | 0.484 | 0.828 | 0.898 | 0.898 | 1.664 | 0.719 | 0.533 | 0.129 | 0.655 | 0.155 |
| rec 512/128 | 12 | 0.486 | 0.832 | 0.900 | 0.900 | 1.663 | 0.715 | 0.541 | 0.133 | 0.678 | 0.162 |
| rec 1024/64 | 10 | 0.486 | 0.858 | **0.932** | **0.932** | 1.609 | 0.712 | 0.564 | 0.140 | 0.665 | 0.160 |
| rec 1024/64 | 12 | 0.486 | 0.854 | 0.928 | 0.928 | 1.605 | 0.706 | 0.560 | 0.141 | 0.689 | 0.169 |
| rec 1024/128 | 10 | 0.490 | 0.850 | 0.910 | 0.910 | 1.610 | 0.712 | 0.577 | 0.145 | 0.673 | 0.166 |
| rec 1024/128 | 12 | 0.488 | 0.856 | 0.912 | 0.912 | 1.611 | 0.707 | 0.571 | 0.144 | **0.690** | 0.169 |
| rec 2048/64 | 10 | 0.500 | 0.868 | 0.940 | 0.940 | 1.512 | 0.698 | 0.587 | 0.151 | 0.667 | 0.168 |
| rec 2048/64 | 12 | 0.500 | 0.870 | 0.946 | 0.946 | 1.525 | 0.693 | 0.587 | 0.151 | 0.671 | 0.169 |
| rec 2048/128 | 10 | **0.508** | 0.886 | 0.948 | 0.948 | 1.526 | 0.699 | 0.581 | 0.150 | 0.673 | 0.169 |
| rec 2048/128 | 12 | 0.502 | 0.880 | 0.942 | 0.942 | 1.530 | 0.694 | **0.593** | **0.154** | 0.687 | **0.174** |
| semantic p95 | 10 | 0.484 | 0.886 | 0.948 | 0.948 | 1.416 | 0.693 | 0.583 | 0.148 | 0.658 | 0.164 |
| semantic p95 | 12 | 0.484 | **0.890** | **0.948** | **0.948** | 1.418 | 0.687 | 0.570 | 0.146 | 0.658 | 0.164 |

Reading of the numbers (compare `doc/enterprise-rag-techqa-refusal-analysis.md`
§2.4, which reconstructed gold top-10 coverage manually at 91–97 %):

- The built-in `gold_doc` hit@10 values (0.898–0.948) sit right in the band of
  the manual reconstruction — the run6 hybrid configs' ad-hoc Gold@10 column
  (0.904–0.966) and run7's framework metric agree within ~1–2 pp for the
  matched recursive configs, so the gold_doc mode reproduces the manual
  analysis and issue 16 can be closed.
- hit@k = recall@k here because TechQA questions have a single gold document.
- Retrieval ranking across chunkings mirrors run6: 512-token chunks worst
  (hit@10 0.898–0.912), 2048 and semantic best (0.940–0.948); k10 → k12 moves
  hit@k by ≤ 0.006 everywhere (at 12 contexts nearly every gold doc is
  already in the top 10).
- **nDCG bug found and fixed (was: open caveat).** The nDCG@10 column above
  (1.41–1.66) is the raw report value and is invalid: `_ndcg_at_k` counted
  every retrieved chunk whose doc id is in the gold set, and hybrid retrieval
  returns several chunks of the same document — with a single gold doc, DCG
  summed over duplicate positions while IDCG assumed one relevant item
  (reproducible: ndcg@5 of 1.887 for `[doc1, x, doc1, x, doc1]`, gold = doc1).
  Fixed in `benchmark/gold_retrieval_metrics.py` (PR #33): ranks keep their
  original positions, a gold document is credited once at its first rank,
  and chunks without a provider doc id stay in place as non-relevant
  positions. Regression tests in `tests/test_gold_retrieval_metrics.py`.
- **Corrected nDCG@10 pending.** A first recomputation (0.711–0.739) used an
  interim fix that deduplicated and then renumbered ranks; that moved gold
  documents to better ranks than they had, so those values are too high and
  are withdrawn. The run7 samples must be re-scored with the final fix on the
  DGX Spark (raw data lives there; the six recursive k10 configs' samples were
  not persisted). hit@k / recall@k are set-based and unaffected by
  duplicates; they only change if some retrieved chunks lacked a doc id
  (those chunks previously shifted later chunks into the top-k window) —
  check the run7 metadata for gaps during the re-score.
- Context relevance (0.687–0.719) and the original BERT F1 (0.530–0.593)
  suggested the run6 order (2048/semantic > 1024 > 512). That order was
  largely an artefact of the refusal check (issue #57): it zeroed more genuine
  answers in the small-chunk configs. With the fixed check (BERT F1
  0.656–0.690) the spread shrinks to 0.035, k12 beats k10 for every recursive
  chunking, and semantic p95 sits with the 512 k10 configs at the bottom —
  chunk size alone no longer explains answer quality.
- Infra note: both semantic configs needed the Ollama embed-batching fix
  (`04a171f6`); with it, index builds run clean (see failure details above).

### RAGBench TechQA: query refinement × hybrid alpha ablation (run8), n = 500

Issue #22. Manifest `experiments/enterprise-rag-run8-ablation.yaml`: fixed cell
rec 2048/64, top-k 10, nomic, hybrid (fetch-k 50), `gold_doc` metrics, RAGAS
off, MLflow tracing on; 2×2 over Blueprint query refinement (on/off) and
hybrid alpha (0.5/0.3). Same 500 questions in every config (paired). Ran
2026-11-24 05:34–07:25 UTC, ~28 min per config, 4/4 done. Config hash
`managed-a6b2f4cd80` = refinement on, `managed-2c9672c978` = refinement off.
Refusals, latency and McNemar tests: `scripts/analyze_blueprint_ablation.py`.

| Refinement | α | hit@1 | hit@5 | hit@10 | nDCG@10 | sent. recall | BERT F1 | ROUGE-L | declined | declined with gold@10 | answer time mean / p95 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| on (baseline) | 0.5 | 0.502 | 0.876 | 0.944 | 0.730 | 0.718 | 0.587 | 0.151 | 32.2 % | 148 | 2.48 / 2.96 |
| on | 0.3 | 0.506 | 0.880 | 0.958 | 0.739 | 0.733 | 0.583 | 0.147 | 31.2 % | 146 | 2.49 / 2.98 |
| off | 0.5 | 0.540 | 0.908 | 0.972 | 0.768 | 0.740 | **0.643** | **0.171** | **23.8 %** | 116 | **2.24 / 2.68** |
| off | 0.3 | **0.572** | **0.938** | **0.976** | **0.788** | **0.749** | 0.624 | 0.168 | 24.6 % | 120 | 2.27 / 2.70 |

`declined` = `declines()` from `benchmark/answer_match.py`. Exact paired
McNemar tests (discordant questions in parentheses):

| Comparison | hit@1 | hit@10 | declined |
|---|---|---|---|
| refinement off vs on, α 0.5 | p = 0.040 (48 vs 29) | p = 0.007 (19 vs 5) | p < 0.001 (20 vs 62 refused only by one side) |
| refinement off vs on, α 0.3 | p < 0.001 (59 vs 26) | p = 0.049 (13 vs 4) | p < 0.001 (29 vs 62) |
| α 0.3 vs 0.5, refinement on | p = 0.86 | p = 0.039 (8 vs 1) | p = 0.49 |

- **Query refinement hurts on TechQA.** Turning it off raises hit@1 by 4–7 pp
  and hit@10 by 2–3 pp, cuts refusals from ~32 % to ~24 % (−42 refused
  questions net, significant at both alphas), raises BERT F1 by 0.04–0.06 and
  saves ~0.25 s per answer (one LLM call less). The refined query loses
  vocabulary that matters on TechQA (product names, error codes); the adapter
  does not yet log the refined query, so this mechanism is unverified.
- **Alpha 0.3 vs 0.5 is secondary**: +0.4–3 pp retrieval, only hit@10 with
  refinement on is (barely) significant; no effect on refusals.
- **Refusals are a generation problem.** 92–98 % of refused questions have the
  gold document in the top 10 (148 of 161 in the baseline).
- **nDCG@10 is ≤ 1 with the PR #33 fix** (0.730–0.788). The baseline (0.730)
  sits inside the withdrawn run7 interim range, but run7 has still not been
  re-scored, so do not compare the two directly.
- **NLG metrics above were biased by a refusal-detector bug (fixed, issue
  #57).** The old `is_refusal_answer()` zeroed BERT/ROUGE/BLEU/METEOR for
  genuine answers containing "insufficient", "provided context" or "does not
  provide". Re-scored from the stored samples
  (`results/Enterprise-RAG/run8/nlg_rescore_issue57.json`,
  `changes/2026-11-24-refusal-nlg-false-positives.md`):

  | Refinement | α | BERT F1 (fixed) | ROUGE-L (fixed) |
  |---|---|---|---|
  | on (baseline) | 0.5 | 0.675 | 0.170 |
  | on | 0.3 | 0.675 | 0.168 |
  | off | 0.5 | 0.755 | 0.197 |
  | off | 0.3 | **0.757** | **0.199** |

  Refinement off still wins, by 0.08 BERT F1 instead of 0.04–0.06.
- RAGAS follow-up (faithfulness, correctness) for refinement on vs off at
  α 0.5: run9, `experiments/enterprise-rag-run9-ragas.yaml`.

## Findings

1. **TechQA is much harder than SQuAD for the Blueprint.** Context recall falls
   from 0.83–0.95 (SQuAD) to 0.26–0.38, answer correctness from 0.51–0.57 to
   0.36–0.47, answer relevancy from ~0.78 to ~0.50. The Blueprint often answers
   "the context does not explain this … I used my own knowledge", which lowers
   faithfulness and relevancy. Five retrieved chunks rarely cover TechQA's long
   technical answers.
2. **Larger chunks help on TechQA.** 1024-token chunks raise context recall by
   ~0.10 (0.26–0.28 → 0.37–0.38) for both embedders, plus faithfulness and
   correctness. On SQuAD, 1024 raises faithfulness (0.87 → 0.91) and correctness
   but lowers heuristic hit@k; 1024/128 and semantic lose ~0.1 context recall.
3. **nomic-embed-text ≥ Qwen3-Embedding-4B on TechQA** in the LLM-judged
   metrics for the three matched configs: context precision +0.04–0.09, answer
   correctness +0.03–0.09, context recall equal, faithfulness equal within ±0.02.
   Qwen3-4B costs ~10× the indexing time (1 000–1 800 s vs 80–150 s) and crashed
   the Ollama runner on semantic chunking.
4. **Overlap 64 vs 128 makes little difference** (≤ 0.03 on most metrics).
5. **Best configs so far:** SQuAD — semantic p95 (best correctness 0.566 and
   hit@5 0.822) or rec 512/128 (best context recall and answer match). TechQA —
   rec 1024/128 at top-k 10 with nomic (run5, correctness 0.513, tied with
   rec 2048/64 k10 at half the context length).
6. **Higher top-k lifts context recall much more than correctness (run5 vs
   run4).** rec 1024/64: context recall 0.370 (k5) → 0.525 (k10) → 0.577 (k12),
   correctness only 0.468 → 0.478 → 0.484. k12 adds recall and faithfulness but
   no correctness over the best k10 configs. 512-token chunks stay clearly worst
   at k10.
7. **Refusals are the main TechQA bottleneck.** ~34–40 % of answers say the
   context does not contain the answer (run4: 43–45 %); they score answer
   relevancy ≈ 0 and explain most of the low mean relevancy (~0.53). About 35 %
   of answerable questions (reference is not a refusal; ~155 of 500 questions)
   are refused. Refused
   questions have low context recall (0.18–0.44 vs 0.48–0.70 when answered),
   so retrieval misses dominate, but 10–29 refusals per config have context
   recall ≥ 0.8 — the generator is over-cautious there. When the Blueprint does
   answer, correctness (0.55–0.57) and relevancy (~0.71) are almost
   config-independent: chunking and top-k mainly move the refusal rate.
8. **Semantic p95 chunks are long** (~3.6 k chars each, 36–43 k context chars at
   k10/12, about twice rec 2048), yet not better than rec 1024/2048, and the
   slowest. Compare it only at a matched context budget.
9. **Hybrid retrieval (run6) improves retrieval clearly, answers only a little.** Gold@k rises in every config (+0.05–0.10), relevant-sentence coverage
   +0.03–0.15, context recall +0.04–0.14 and refusals drop by 3–7 pp in most
   configs. Answer correctness rises significantly only where dense retrieval
   was weakest (rec 512 k10: +0.05–0.07) and for semantic k12 (+0.030); for
   semantic k10 it drops (−0.018) despite the best retrieval gain. The former
   best configs (rec 1024/128 and 2048/64 at k10, 0.513) stay tied (−0.01,
   n.s.). Best TechQA config by correctness is **semantic p95 k12 hybrid**
   (0.523, context recall 0.672, refusals 31 %), but at ~52 k context chars per
   question and with the opposite sign at k10, so its lead is not robust. **rec
   2048/64 k12 hybrid** (0.512, 20 k chars, refusals 32 %) is the recommended
   default. Hybrid roughly doubles the Blueprint's answering time per question.
   Better retrieval converts into correctness only weakly: answer correctness
   is no longer limited mainly by retrieval but by refusals, judge noise and
   the reference quality (see the refusal analysis).
10. **The gold_doc metric mode (run7) validates the manual reconstruction.**
    Built-in gold hit@10 (0.898–0.948) matches run6's ad-hoc Gold@10 column and
    the manual §2.4 reconstruction (91–97 %) within ~1–2 pp, so issue 16's
    framework-side verification is done and future runs no longer need the
    manual reconstruction. Retrieval is saturated at k ≥ 10 (k12 adds ≤ 0.006),
    so the remaining answer-quality levers are not retrieval-side. run7 also
    exposed and fixed the nDCG duplicate-document bug (corrected nDCG@10
    0.711–0.739, see the run7 section).
11. **Turn Blueprint query refinement off for TechQA (run8).** On the
    recommended rec 2048/64 k10 hybrid cell, refinement off is significantly
    better on gold hit@1/@10 and refusals (32 % → 24 %), answers faster, and
    raises BERT F1; alpha 0.3 vs 0.5 is a minor second-order effect. Best cell:
    refinement off, α 0.3 (retrieval) or α 0.5 (answer metrics, refusals).
    Almost all remaining refusals have the gold document in the context, so
    the next lever is the generation side (prompt, model), not retrieval.

## Caveats

- **Heuristic retrieval metrics are not trustworthy here.** With no gold doc IDs
  (`ground_truth_doc_ids` is empty), hit@k / nDCG / recall@k count a context as
  relevant when cosine(ground-truth answer, context) ≥ 0.5 under the critic
  embedding (`benchmark/custom_metrics.py`, `determine_relevance`). nomic gives
  high baseline similarity in-domain, so run4 scores hit@1 = 1.0 for every config
  (run5: 0.99) — a threshold artefact, not perfect retrieval (RAGAS context
  recall is 0.27–0.37 in run4). In run5, recall@k is simply k/10 because all ten
  contexts count as relevant. Use RAGAS context precision/recall for retrieval
  quality.
- **Embedding-based metrics differ by critic embedding**: run3 scores semantic
  similarity, context relevance, vector distances and heuristic hit@k with
  Qwen3-Embedding-4B, run4 with nomic. RAGAS answer correctness and answer
  relevancy also use the critic embedding in part. Faithfulness, context precision
  and context recall are purely LLM-judged and the fairest cross-embedding view.
- **Answer match is 0 on TechQA by design**: the deterministic match targets short
  answers (SQuAD spans); long-form TechQA references never match.
- Retrieved doc IDs repeat (several chunks of the same document in the top 5), so
  the five contexts cover fewer than five distinct documents.
- **The TechQA corpus contains duplicate documents** under different IDs (e.g.
  `dataset-doc-0` and `dataset-doc-271` share text). In run5 only ~7.2–7.8 of 10
  (k10) and ~8.7–9.4 of 12 (k12) retrieved contexts are distinct texts, and
  5.6–9.1 distinct doc IDs.
- n = 500 gives roughly ±0.02–0.04 standard error on the means; differences below
  ~0.03 are noise. Paired comparisons per question (as in run5) are tighter
  (CI half-width ~0.016 for correctness).
- run5 reports token counts of 0 and empty `llm_performance_metrics` although
  `llm_performance_enabled` is true: the `enterprise_rag_mcp` adapter does not
  report generator tokens and the load test does not run for it.

## Open / next

Refusal causes, runtime and next steps for run6 are analysed in
[enterprise-rag-techqa-refusal-analysis.md](enterprise-rag-techqa-refusal-analysis.md).

Further Blueprint changes are evaluated one component at a time with the fixed
regression benchmark instead of new sweeps:
[blueprint-regression-benchmark.md](blueprint-regression-benchmark.md) (its
`dev` split is exactly the 500 questions of run7–run9).

- The run4 gaps (rec 1024/128, semantic at k5) are superseded by run5; retry
  semantic with Qwen3-4B only if the embedder comparison is still needed.
- Deduplicate the TechQA corpus before indexing: about a quarter of the context
  slots are duplicate texts.
- Inspect the Blueprint prompt and query refinement on refusals with high
  context recall (over-cautious generator). run8 shows refinement itself
  causes ~42 extra refusals; logging the refined query in the adapter is
  needed to see what it drops.
- Fix `is_refusal_answer()` false positives (see run8) and re-score NLG
  metrics.
- Re-test semantic chunking with a bounded chunk size or a matched context
  budget.
- Gold-doc retrieval metrics for TechQA (map RAGBench documents to
  `ground_truth_doc_ids`, `custom_retrieval_metrics_mode: gold_doc`) would replace
  the heuristic hit@k.
