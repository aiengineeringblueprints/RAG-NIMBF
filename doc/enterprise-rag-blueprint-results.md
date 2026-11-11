# Enterprise RAG Blueprint — Results Overview

Status: 2026-11-11. Source: `results/Enterprise-RAG/run1..run6` on the DGX Spark
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
| Retrieval | similarity in run1–run5, hybrid (dense + BM25, RRF, alpha 0.5, fetch-k 50) in run6; `retrieval_top_k` 5 in run1–run4, 10 / 12 in run5–run6 (= contexts per question), threshold 0.2, no HyDE / multihop / reranker |
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
| run5 | 2026-11-04 → 06 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | rec 512/1024/2048 × overlap 64/128 + semantic p95, each at top-k 10 and 12 (14) | 14/14 done |
| run6 | 2026-11-08 → 11 | RAGBench TechQA (train) | 500 | nomic-embed-text | deepseek-v4-flash / nomic | same 14 as run5, **hybrid retrieval** | 13/14 done, semantic k10 **failed**, re-run in progress |

Combination matrix (✅ done, ❌ failed, ⏹ aborted, — not run):

| Chunking | SQuAD · nomic (run2, k5) | TechQA · Qwen3-Emb-4B (run3, k5) | TechQA · nomic (run4, k5) | TechQA · nomic (run5, k10 + k12) |
|---|---|---|---|---|
| recursive 512 / 64 | ✅ (+ run1 smoke) | ✅ | ✅ | ✅ |
| recursive 512 / 128 | ✅ | ✅ | ✅ | ✅ |
| recursive 1024 / 64 | ✅ | ✅ | ✅ | ✅ |
| recursive 1024 / 128 | ✅ | ✅ | ⏹ | ✅ |
| recursive 2048 / 64 | — | — | — | ✅ |
| recursive 2048 / 128 | — | — | — | ✅ |
| semantic (percentile 95) | ✅ | ❌ | — | ✅ |

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
paired-bootstrap 95 % CI that excludes 0. semantic p95 k10 failed in run6
(Ollama `/tokenize` reset) and is being re-run.

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
| semantic p95 | 10 | 0.493 → pending | | | | | | |
| rec 512/64 | 12 | 0.476 → 0.487 | 0.804 → 0.841* | 0.603 → 0.608 | 0.507 → 0.512 | 41.4 → 37.6 % | 0.874 → 0.920* | 0.378 → 0.413* |
| rec 512/128 | 12 | 0.482 → 0.485 | 0.837 → 0.827 | 0.621 → 0.620 | 0.502 → 0.521 | 39.8 → 37.2 % | 0.868 → 0.914* | 0.387 → 0.416* |
| rec 1024/64 | 12 | 0.484 → 0.498 | 0.850 → 0.834 | 0.595 → 0.617 | 0.577 → 0.615* | 38.8 → 33.8 %* | 0.860 → 0.934* | 0.535 → 0.597* |
| rec 1024/128 | 12 | 0.488 → 0.496 | 0.843 → 0.851 | 0.627 → 0.642 | 0.564 → 0.604* | 38.4 → 35.4 % | 0.850 → 0.926* | 0.549 → 0.597* |
| rec 2048/64 | 12 | 0.498 → 0.512 | 0.843 → 0.852 | 0.601 → 0.623 | 0.603 → 0.664* | 36.8 → 31.8 %* | 0.896 → 0.960* | 0.648 → 0.740* |
| rec 2048/128 | 12 | 0.494 → 0.503 | 0.853 → 0.864 | 0.581 → 0.620* | 0.605 → 0.646* | 38.8 → 31.4 %* | 0.880 → 0.954* | 0.640 → 0.733* |
| semantic p95 | 12 | 0.497 → **0.523*** | 0.842 → 0.858 | 0.572 → 0.642* | 0.604 → **0.672*** | 34.6 → **31.2 %** | 0.872 → **0.966*** | 0.692 → **0.843*** |

Pooled over the 13 matched configs (paired; the CIs treat question × config
as independent and are therefore optimistic): answer correctness +0.018,
faithfulness +0.017, answer relevancy +0.019, context precision +0.030, context
recall +0.056, refusals −4.2 pp, Gold@k +0.071, relevant-sentence coverage
+0.066. Refusal flips: 719 refused → answered, 446 answered → refused. The
answerable subset (questions with RAGBench relevant sentences) shows the same
deltas.

Hybrid retrieval concentrates the chunks on fewer documents (distinct doc IDs
per question 5.7–9.1 → 5.0–8.2) and picks slightly longer chunks (+3–6 %
context chars for recursive, +27 % for semantic, 41 k → 52 k), while
distinct context texts stay at ~k. Blueprint answering time rises from
641–863 s to 1,148–1,390 s per config for recursive chunking (~1.3 → ~2.4 s
per question) and from 1,225 to 1,690 s for semantic k12.


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
9. **Hybrid retrieval (run6) improves retrieval clearly, answers only a little.**
   Gold@k rises in every config (+0.05–0.10), relevant-sentence coverage
   +0.03–0.15, context recall +0.04–0.14 and refusals drop by 3–7 pp in most
   configs. Answer correctness rises significantly only where dense retrieval
   was weakest (rec 512 k10: +0.05–0.07) and for semantic k12 (+0.030). The
   former best configs (rec 1024/128 and 2048/64 at k10, 0.513) stay tied
   (−0.01, n.s.). **New best TechQA config: semantic p95 k12 hybrid** (correctness
   0.523, context recall 0.672, refusals 31 %), at the cost of ~52 k context
   chars per question; rec 2048/64 k12 hybrid (0.512, 20 k chars) is the
   cheaper alternative. Hybrid roughly doubles the Blueprint's answering time
   per question.

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

- The run4 gaps (rec 1024/128, semantic at k5) are superseded by run5; retry
  semantic with Qwen3-4B only if the embedder comparison is still needed.
- Deduplicate the TechQA corpus before indexing: about a quarter of the context
  slots are duplicate texts.
- Inspect the Blueprint prompt and query refinement on refusals with high
  context recall (over-cautious generator).
- Re-test semantic chunking with a bounded chunk size or a matched context
  budget.
- Gold-doc retrieval metrics for TechQA (map RAGBench documents to
  `ground_truth_doc_ids`, `custom_retrieval_metrics_mode: gold_doc`) would replace
  the heuristic hit@k.
