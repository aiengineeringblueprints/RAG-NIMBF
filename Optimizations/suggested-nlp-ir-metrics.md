# Suggested NLP & IR Metrics for the Framework

## IR Metrics (Retrieval Quality)

| Metric | What it adds beyond current metrics |
| ------ | ----------------------------------- |
| **Hit@k** | Binary: did at least one relevant chunk appear in top-k? Current context precision is a softer, averaged score — Hit@k gives a hard pass/fail signal that's easy to interpret and compare across configs |
| **nDCG** | Graded ranking quality — unlike context precision, it penalizes relevant results appearing lower in the ranking. If order matters (top chunks matter more than bottom ones), this is the right metric |
| **Recall@k** | What fraction of all relevant chunks were retrieved in top-k — current context recall is RAGAS-flavored and uses LLM judging. A traditional recall@k is deterministic and directly comparable across embedding models |
| **Context Relevance** | Broader signal: is the retrieved context on-topic for the question at all, regardless of whether it matches ground truth. Useful for diagnosing configs where chunks are retrieved but contribute nothing |

**Verdict:** All four fill a gap — the current retrieval evaluation relies entirely on RAGAS LLM-judged metrics. Adding deterministic IR metrics gives reproducible, cost-free baselines that don't depend on a critic model's quality.

---

## NLG Metrics (Answer Quality)

| Metric | What it adds beyond current metrics |
| ------ | ----------------------------------- |
| **ROUGE-L** | Longest common subsequence overlap with ground truth — captures structural/sequential similarity. Current answer correctness uses LLM judging + semantic similarity. ROUGE-L adds a cheap, deterministic lexical baseline |
| **BLEU** | N-gram precision against reference — standard in MT/summarization. Less suited for open-ended QA (synonyms score as 0), but useful if prompt templates produce formulaic answers that should match closely |
| **METEOR** | Like BLEU but with synonym/stem matching and recall weighting — addresses BLEU's biggest weakness. Better fit for RAG where paraphrasing is common |
| **BERTScore** | Contextual embedding similarity between generated and reference answer — bridges the gap between lexical metrics (ROUGE/BLEU) and LLM-judged RAGAS metrics. Computationally heavier than ROUGE but much cheaper than LLM judging |

**Verdict:**
- **ROUGE-L** and **BERTScore** are the strongest picks — ROUGE-L as a fast deterministic baseline, BERTScore for semantic similarity without LLM cost
- **METEOR** > BLEU for RAG use cases (synonym handling matters when answers paraphrase)
- **BLEU** is the weakest fit here — it's too strict for generative QA

---

## Token Metrics

| Metric | What it adds |
| ------ | ------------ |
| **Prompt Tokens** | Output token count is already tracked — prompt tokens capture the full cost picture (context + prompt template + question). Essential for cost estimation |
| **Completion Tokens** | Already tracked as "Token Count", just needs consistent naming |

**Verdict:** Prompt tokens is a no-brainer — it's available for free in every LLM API response and directly enables cost tracking.

---

## Summary Ranking

### Tier 1 — High value, easy add

1. Prompt Tokens
2. Hit@k / Recall@k (deterministic, no LLM cost)
3. ROUGE-L (deterministic, no LLM cost)

### Tier 2 — High value, moderate effort

4. nDCG (requires graded relevance, but very informative)
5. BERTScore (needs embedding computation, but that infrastructure already exists)
6. Context Relevance (can reuse existing embedding models)

### Tier 3 — Nice to have

7. METEOR (better than BLEU for RAG)
8. BLEU (only useful for full coverage of standard NLG metrics)
