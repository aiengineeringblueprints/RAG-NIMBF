# Enterprise RAG Blueprint — Results Overview

Status: 2026-11-01. Source: `results/Enterprise-RAG/run1..run4` on the DGX Spark
(`~/source/Benchmarking-Framework`), manifest `experiments/enterprise-rag-blueprint-rag.yaml`.
Setup guide: [enterprise-rag-blueprint-setup.md](enterprise-rag-blueprint-setup.md).

All runs benchmark the Blueprint as **plain RAG (MCP off)** through the
`enterprise_rag_mcp` adapter (`inprocess` transport). The framework rebuilds the
Blueprint's Chroma index per config (chunker + embedder from the matrix); the
Blueprint retrieves and answers itself, with guardrails and query refinement on.

## Fixed settings (all runs)

| Setting | Value |
|---|---|
| Generator LLM | `qwen3.8-27b` (OpenAI-compatible), temperature 0, top_p 0.3 |
| Retrieval | similarity, `retrieval_top_k` 5 (5 contexts per question), threshold 0.2, no HyDE / multihop / reranker |
| Vector DB | Chroma (Blueprint) |
| Blueprint flags | guardrails on, query refinement on, MCP off, role `General` |
| Prompt template | `concise` (not used by the Blueprint) |
| Retrieval metrics mode | `heuristic` (no gold doc IDs, see caveats) |

## Runs and combinations

| Run | Date (UTC) | Dataset | n | Embedding | Critic LLM / critic embedding | Configs | Status |
|---|---|---|---|---|---|---|---|
| run1 | 2026-10-29 | SQuAD | 20 | nomic-embed-text | claude-opus-5-5 / nomic | rec 512/64 | 1/1 done (smoke) |
| run2 | 2026-10-29 → 30 | SQuAD | 500 | nomic-embed-text | deepseek-v4-flash / nomic | rec 512/64, 512/128, 1024/64, 1024/128, semantic p95 | 5/5 done |
| run3 | 2026-10-30 → 31 | RAGBench TechQA (train) | 500 | Qwen3-Embedding-4B | deepseek-v4-flash / Qwen3-Embedding-4B | same 5 | 4/5 done, semantic **failed** |
| run4 | 2026-10-31 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | same 5 | 3/5 done, rec 1024/128 **hung, stopped manually**, semantic not started |

Combination matrix (✅ done, ❌ failed, ⏹ aborted, — not run):

| Chunking | SQuAD · nomic (run2) | TechQA · Qwen3-Emb-4B (run3) | TechQA · nomic (run4) |
|---|---|---|---|
| recursive 512 / 64 | ✅ (+ run1 smoke) | ✅ | ✅ |
| recursive 512 / 128 | ✅ | ✅ | ✅ |
| recursive 1024 / 64 | ✅ | ✅ | ✅ |
| recursive 1024 / 128 | ✅ | ✅ | ⏹ |
| semantic (percentile 95) | ✅ | ❌ | — |

Failure details:

- run3 semantic: `Post "http://127.0.0.1:41851/tokenize": ... connection reset by peer`
  — the Ollama runner for Qwen3-Embedding-4B died during semantic chunking.
- run4 recursive 1024/128: started 2026-10-31 15:37 UTC, no progress afterwards;
  the run was stopped and the summary JSON covers the 3 finished configs.

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

### Runtime (seconds per config, n = 500)

| Run | Index build (`adapter_prepare`) | Blueprint answering | RAGAS eval | Custom metrics | Total | Critic tokens |
|---|---|---|---|---|---|---|
| run2 SQuAD · nomic | 3–17 | 490–498 | 6 934–7 482 | 160–168 | 7 603–8 163 (~2.1–2.3 h) | 6.4–7.0 M |
| run3 TechQA · Qwen3-4B | 1 031–1 830 | 663–692 | 7 151–8 037 | 711–841 | 9 732–11 161 (~2.7–3.1 h) | 8.7–9.7 M |
| run4 TechQA · nomic | 83–151 | 618–641 | 7 205–7 890 | 216–225 | 8 201–8 852 (~2.3–2.5 h) | 8.8–10.6 M |

RAGAS (critic LLM) dominates every config at ~85–90 % of the wall time.

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
   rec 1024/64 with nomic (best faithfulness, relevancy, correctness, precision).

## Caveats

- **Heuristic retrieval metrics are not trustworthy here.** With no gold doc IDs
  (`ground_truth_doc_ids` is empty), hit@k / nDCG / recall@k count a context as
  relevant when cosine(ground-truth answer, context) ≥ 0.5 under the critic
  embedding (`benchmark/custom_metrics.py`, `determine_relevance`). nomic gives
  high baseline similarity in-domain, so run4 scores hit@1 = 1.0 for every config —
  a threshold artefact, not perfect retrieval (RAGAS context recall is 0.27–0.37
  in the same run). Use RAGAS context precision/recall for retrieval quality.
- **Embedding-based metrics differ by critic embedding**: run3 scores semantic
  similarity, context relevance, vector distances and heuristic hit@k with
  Qwen3-Embedding-4B, run4 with nomic. RAGAS answer correctness and answer
  relevancy also use the critic embedding in part. Faithfulness, context precision
  and context recall are purely LLM-judged and the fairest cross-embedding view.
- **Answer match is 0 on TechQA by design**: the deterministic match targets short
  answers (SQuAD spans); long-form TechQA references never match.
- Retrieved doc IDs repeat (several chunks of the same document in the top 5), so
  the five contexts cover fewer than five distinct documents.
- n = 500 gives roughly ±0.02–0.04 standard error on the means; differences below
  ~0.03 are noise.

## Open / next

- Finish the TechQA · nomic matrix: rec 1024/128 and semantic p95 (resume with
  `python -m benchmark.worker run experiments/enterprise-rag-blueprint-rag.yaml --keep-going --run-dir results/Enterprise-RAG/run4`
  after restoring the run4 matrix) and retry semantic with Qwen3-4B.
- The manifest on the Spark is already edited for the next sweep and has not run
  yet: `retrieval_top_k [10, 12]`, `chunk_sizes [512, 1024, 2048]`,
  `chunk_overlaps [64, 128]`, recursive + semantic, nomic → 14 configs, ~33–35 h
  at the current per-config time. A higher top-k targets the low TechQA context recall.
- Gold-doc retrieval metrics for TechQA (map RAGBench documents to
  `ground_truth_doc_ids`, `custom_retrieval_metrics_mode: gold_doc`) would replace
  the heuristic hit@k.
