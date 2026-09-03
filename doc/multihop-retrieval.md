# Multi-Hop Retrieval Support (HotpotQA & friends)

Added 2026-09-03. This document explains **why** these changes were made and
**how** to configure them.

## The problem

Running HotpotQA (`DATASET_NAME=multihop-generic`) with the default pipeline
(`recursive` chunking, 500/200, `similarity` retrieval, top_k=12) produced a
large number of `"I cannot answer from the provided context."` responses
(~31% in run99's first 130 questions).

### Root-cause analysis

1. **The refusal string comes from the prompt template.**
   `benchmark/prompt_templates/concise.py` instructs the model to say
   *"I cannot answer from the provided context."* whenever the answer is not
   in the retrieved chunks. So every refusal means: **the required fact was
   not retrieved.**

2. **HotpotQA questions are true 2-hop questions.** Each question requires
   joining facts from *two different* Wikipedia articles, e.g.
   *"What nationality was Oliver Reed's character in the film Royal Flash?"*
   (film article → character article). The answered questions in the failing
   run were mostly single-hop questions where one passage sufficed.

3. **Dense retrieval cannot bridge hops.** A single query embedding is
   compared against all chunks. Hop-2 documents often share no surface form
   with the question (the bridge goes question → doc A → doc B), so
   `nomic-embed-text` surfaces doc A at best.

4. **The old corpus layout made it worse.** `multihop-generic` previously had
   `has_shared_corpus=False`, so each question's flattened 10-paragraph
   context (2 gold + 8 distractor paragraphs) was chunked separately with
   `CHUNK_SIZES=500 / CHUNK_OVERLAPS=200`. Consequences:
   - Chunks cut across paragraph boundaries, mixing documents.
   - Paragraphs shared between questions were chunked and embedded multiple
     times (500 questions → 8448 chunks with heavy duplication).
   - The 12-chunk budget was often consumed by chunks of one hop plus
     lexically-adjacent *distractor* paragraphs, crowding out hop 2.
   - No `doc_id`/`gold_doc_ids` metadata existed, so gold-doc retrieval
     metrics (`CUSTOM_RETRIEVAL_METRICS_MODE=gold_doc`) could not work.

## The changes

### 1. Paragraph-level shared corpus (`benchmark/dataset.py`)

`multihop-generic` now sets `has_shared_corpus=True`. The new
`_load_multihop_corpus()` splits each flattened context into its individual
paragraphs and deduplicates them into a shared corpus:

- Each unique paragraph = **one corpus document** with `title` and `doc_id`
  metadata.
- Each question gets `metadata["gold_doc_ids"]` — the corpus doc IDs of its
  `supporting_facts` titles. This enables gold-retrieval metrics
  (`hit@k`, `ndcg@k`, `recall@k`) via `CUSTOM_RETRIEVAL_METRICS_MODE=gold_doc`.
- For 500 HotpotQA samples this yields ~5000 unique paragraph documents
  instead of ~8400 duplicated cross-paragraph chunks.

Code: `benchmark/dataset.py` → `_load_multihop_corpus()`; adapter flag in
`benchmark/dataset_adapters.py`.

### 2. `paragraph` chunking strategy (`benchmark/chunking.py`)

New strategy `CHUNKING_STRATEGIES=paragraph` (`ParagraphChunker`): emits
**one chunk per corpus document without splitting**. Since corpus documents
are already individual paragraphs, this is the natural retrieval unit for
HotpotQA. Metadata (`title`, `doc_id`) is preserved on every chunk, so
gold-doc metrics still work after chunking.

### 3. Iterative multi-hop retrieval (`benchmark/retrieval.py`)

New function `retrieve_multihop(vector_store, llm, question, top_k, rounds=...)`:

1. Round 1: standard retrieval (similarity or MMR, HyDE-compatible).
2. Each further round: the generator LLM is shown the question plus the
   retrieved documents and writes **one follow-up search query** targeting
   the still-missing fact. That query is retrieved, and new unique documents
   are merged into the result set (deduplicated by `doc_id`).
3. Stops early if the LLM answers `NONE` (context complete) or fails.
4. Returns at most `top_k` documents, round-1 ranking first.

It composes with HyDE, MMR, and the reranker (reranking still runs on the
merged result). Implementation: `retrieve_multihop()` +
`_generate_followup_query()` in `benchmark/retrieval.py`; wired into the
generation loop in `main.py`.

### 4. Configuration plumbing (`config.py`)

| Env variable | Default | Meaning |
|---|---|---|
| `RETRIEVAL_MULTIHOP` | `false` | Enable iterative multi-hop retrieval |
| `RETRIEVAL_MULTIHOP_ROUNDS` | `1` | Total retrieval rounds (2 = one follow-up) |

Both flow through the experiment matrix / YAML overrides. When enabled, the
config name gets a suffix like `_mh2` so runs don't collide with baselines.

## Recommended configuration for HotpotQA

```bash
# .env
DATASET_NAME=multihop-generic
DATASET_HF_ID=hotpotqa/hotpot_qa
DATASET_SUBSET=distractor
DATASET_SPLIT=validation

CHUNKING_STRATEGIES=paragraph
RETRIEVAL_MULTIHOP=true
RETRIEVAL_MULTIHOP_ROUNDS=2
CUSTOM_RETRIEVAL_METRICS_MODE=gold_doc   # to measure hop coverage directly
```

## Notes & gotchas

- **Index rebuild required:** the shared corpus + paragraph chunking changes
  the collection contents, so the first run with the new settings rebuilds
  the Chroma/LanceDB index (different collection name / chunk count). This
  is expected and happens automatically.
- **Cost:** `RETRIEVAL_MULTIHOP` adds one extra LLM call per question per
  additional round (the follow-up query generation). With `ROUNDS=2` that is
  1 extra call per question.
- **Baseline comparison:** keep a baseline config
  (`CHUNKING_STRATEGIES=recursive`, `RETRIEVAL_MULTIHOP=false`) in the same
  run to quantify the improvement. The `_mh2` suffix keeps result names
  distinct.
- **Verifying the fix:** compare the gold-doc hit rate
  (`CUSTOM_RETRIEVAL_METRICS_MODE=gold_doc`) and the count of
  "I cannot answer" responses before/after. Failures that persist even with
  both gold docs retrieved are *generation* problems, not retrieval ones.
