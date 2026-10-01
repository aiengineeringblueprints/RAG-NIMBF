# Results: qwen3.8-27b (critic: deepseek-v4-flash)

Run: `results/qwen3.8-27b/deepseek-v4-flash` · 2026-10-01 · git `37cd6c83` · source `benchmark_20261001_122446.json`
Dataset: `squad` (subset field `FinQA`), 500 questions · GPU: NVIDIA GB10

## Configuration by component

| Component | Setting | concise | detailed |
|---|---|---|---|
| RAG adapter | `rag_system_adapter` | internal | internal |
| Chunking | strategy / size / overlap | recursive / 500 / 50 | recursive / 500 / 50 |
| Chunking | resulting chunks | 970 | 970 |
| Embedding | model | nomic-embed-text:latest | nomic-embed-text:latest |
| Vector store | backend | chroma | chroma |
| Retrieval | strategy / top_k | similarity / 12 | similarity / 12 |
| Retrieval | HyDE / multihop | off / off | off / off |
| Retrieval | similarity threshold | 0.2 | 0.2 |
| Reranker | model | none | none |
| Generation | LLM | openai:qwen3.8-27b | openai:qwen3.8-27b |
| Generation | context_k | 6 | 6 |
| Generation | temperature / top_p | 0.1 / 0.3 | 0.1 / 0.3 |
| Generation | freq / presence penalty | 0.7 / 0.4 | 0.7 / 0.4 |
| **Generation** | **prompt template** | **concise** | **detailed** |
| Evaluation | critic LLM | openai:deepseek-v4-flash | openai:deepseek-v4-flash |
| Evaluation | critic embedding | nomic-embed-text:latest | nomic-embed-text:latest |
| Evaluation | BERTScore model | roberta-large | roberta-large |

Only the prompt template differs between the two configs.

## Overall scores

Best value per row in bold. Retrieval metrics are identical because retrieval is identical.

### Answer quality (RAGAS)

| Metric | concise | detailed |
|---|---|---|
| Faithfulness | 0.830 | **0.943** |
| Context recall | 0.956 | 0.956 |
| Semantic similarity | **0.749** | 0.678 |
| Answer relevancy | n/a | n/a |
| Answer correctness | n/a | n/a |
| Context precision | n/a | n/a |

### Answer quality (custom metrics)

| Metric | concise | detailed |
|---|---|---|
| ROUGE-L | **0.517** | 0.256 |
| BLEU | **0.103** | 0.047 |
| METEOR | **0.563** | 0.449 |
| BERTScore P | 0.833 | **0.908** |
| BERTScore R | 0.840 | **0.928** |
| BERTScore F1 | 0.836 | **0.918** |
| Vec. dist. question ↔ answer | 0.366 | 0.149 |
| Valid answers (`answer_valid`) | 425 / 500 | **473 / 500** |

### Retrieval

| Metric | @1 | @3 | @5 |
|---|---|---|---|
| Hit | 0.698 | 0.776 | 0.798 |
| nDCG | 0.698 | 0.652 | 0.655 |
| Recall | 0.236 | 0.406 | 0.523 |

Context relevance 0.631 · Vec. dist. question ↔ ground truth 0.491

### Cost and runtime

| Metric | concise | detailed |
|---|---|---|
| Generator tokens in / out | 627,874 / 6,169 | 580,374 / 17,069 |
| Critic tokens in / out | 2,116,750 / 145,541 | 2,150,368 / 176,849 |
| Avg. latency per answer | **0.20 s** | 0.34 s |
| Avg. TTFT | 0.117 s | **0.110 s** |
| RAG stage | **110 s** | 179 s |
| RAGAS eval | 2,097 s | 2,339 s |
| Total runtime | **40.3 min** | 45.8 min |

## Notes

- **concise** wins on lexical overlap (ROUGE-L, BLEU, METEOR) and semantic similarity: short answers match SQuAD's span-style ground truth.
- **detailed** wins on faithfulness, BERTScore, and valid answers (+48), at roughly 2.8× the output tokens and 1.7× the latency.
- Answer relevancy, answer correctness and context precision are `null` in the result file: these RAGAS metrics did not produce scores in this run.
- Faithfulness for concise is based on 499 of 500 samples.
- The manifest sets `RETRIEVAL_USE_HYDE=false,true`, but only the two HyDE-off configs exist in this run.
- `dataset_subset: FinQA` with `dataset_name: squad` looks like a leftover setting; the questions are SQuAD questions.
