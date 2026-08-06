# Benchmark a Managed RAG System

A managed RAG adapter lets the framework benchmark systems that own their
dataset ingestion, retrieval, and answer generation. The benchmark keeps one
provider-neutral experiment model; a thin adapter translates it to the target
system's API. optimiseRAG is supported through the `optimaiserag` adapter (the
`ragflow` name is an equivalent alias).

For a concrete copy/paste run sequence, environment checklist, result layout,
and sweep examples, see [OptimiseRAG_Benchmark.md](OptimiseRAG_Benchmark.md).

## What the framework controls

The lifecycle is:

1. Load one question set and, for shared datasets, one source corpus.
2. Validate the adapter's declared capabilities.
3. Create or reuse provider resources, upload sources, start indexing, and poll
   until indexing completes.
4. Retrieve and generate for every question.
5. Normalize returned chunks, source identifiers, scores, references, token
   usage, and timings.
6. Run the existing evaluation and reporting pipeline.
7. Delete resources only when cleanup is enabled and the adapter owns them.

Provider-generated document and chunk IDs are not treated as durable gold
labels. Gold evaluation uses the stable `source_id`/`gold_doc_id` carried by the
benchmark dataset. The provider adapter maps uploaded source names back to
those identifiers.

## Configure connectivity safely

Copy `.env.example` to `.env` and set only machine-local connectivity and
credentials there:

```dotenv
BENCHMARK_CONFIG_FILE=experiments/optimaiserag-northstar.yaml
RAG_MANAGED_BASE_URL=http://spark:8011
RAG_MANAGED_API_KEY_ENV=OPTIMAISE_RAG_API_KEY
OPTIMAISE_RAG_API_KEY=replace-with-the-real-secret
```

Use `http://<spark>:8011` for the documented development deployment or
`http://<spark>:8010` for staging. Do not put the key, an authorization header,
or a token-bearing URL in YAML, source code, logs, or committed result files.
`rag_managed_api_key_env` is the *name* of the environment variable, never its
value.

## Validate the fixture and plan the run

The bundled Northstar Facilities fixture contains 60 fictional Markdown
documents and 360 traceable questions:

```bash
python datasets/northstar_facilities/validate.py
python -m benchmark.worker plan experiments/optimaiserag-northstar.yaml
```

The validator must report `"status": "pass"`, 60 documents, 360 unique
questions, and evidence for all 360 questions. The plan should resolve
`jsonl-shared`, provider chunking, and the `optimaiserag` adapter before it
contacts the service.

The manifest's model identifiers must match models configured in the target
optimiseRAG installation. Change the example `llm_models`, `embedding_models`,
and optional `reranker_models` values if your deployment uses different names.

## Run a small smoke test first

Use the dedicated five-question smoke manifest:

```bash
python -m benchmark.worker run \
  experiments/optimaiserag-northstar-smoke.yaml \
  --run-dir results/optimaiserag-smoke \
  --keep-going
```

This is a real API smoke test but avoids model-based evaluator calls. Confirm
that:

- exactly 60 sources are uploaded or a matching cached dataset is reused;
- indexing reaches its completed state without failed documents;
- each answer contains normalized contexts and chunk metadata;
- returned `source_id` values resemble `NF-001`, not only opaque provider IDs;
- no API key appears in terminal output or under `results/`.

Move to the full manifest only after those checks pass. The unit-level mocked
lifecycle smoke tests can be run without a live service:

```bash
pytest -q tests/test_managed_adapter_lifecycle.py
```

Provider-specific HTTP tests, when present, should use mocked responses and
must not read a real credential.

## Run retrieval without generation

Use the retrieval-only stage to tune embedding, chunking, hybrid weights,
thresholds, and rerankers without calling the chat LLM:

```bash
BENCHMARK_STAGE=retrieve \
BENCHMARK_CONFIG_FILE=experiments/optimaiserag-northstar.yaml \
python main.py
```

`retrieve` invokes the adapter's retrieval endpoint for each question, records
the returned ranked evidence, and computes gold-document retrieval metrics. It
also computes span-level hit, MRR, nDCG, and recall by matching each returned
chunk to the exact line-addressed gold evidence; a wrong section from the right
document therefore does not receive span credit. It skips answer generation
and model-based RAGAS evaluation. This is different
from `index`, which stops after creating an internal index, and `query`, which
means query an existing internal index. Keep `benchmark_stage` out of the YAML
when you want the `BENCHMARK_STAGE` environment variable to choose the stage.

## Run and resume the full benchmark

For a normal run:

```bash
BENCHMARK_CONFIG_FILE=experiments/optimaiserag-northstar.yaml python main.py
```

For a resumable matrix:

```bash
python -m benchmark.worker run \
  experiments/optimaiserag-northstar.yaml \
  --keep-going
```

Keep `rag_managed_reuse_resources: true` while tuning retrieval and generation
so unchanged ingestion configurations can reuse their indexed datasets. Use a
different dataset per embedding/chunking configuration: optimiseRAG cannot
change an embedding model after chunks exist.

Leave `rag_managed_cleanup: false` during initial development so a failed run
can be inspected. Enable cleanup for controlled CI runs once resource ownership
and reuse behavior have been verified. Cleanup must never delete a dataset or
chat that the adapter did not create.

## Add another provider

A provider adapter implements the normalized lifecycle in
`benchmark.adapters.base`:

- `capabilities()` declares controllable features;
- `prepare()` returns a `PreparedTarget` with provider resource IDs;
- `retrieve()` returns ranked `RetrievedChunk` values;
- `generate()` returns an `AdapterGenerationResult` and the exact references
  used to answer;
- `cleanup()` removes owned ephemeral resources.

Register a factory with `register_rag_adapter()`. Reject an unsupported sweep
dimension instead of silently ignoring it. Request mappings must use explicit
field access and JSON serialization; never use `eval`, shell interpolation, or
arbitrary templates on provider responses.

## Parameter mapping for optimiseRAG

| Benchmark field | optimiseRAG/RAGFlow field | Meaning |
|---|---|---|
| `embedding_model` | `embedding_model` | Dataset embedding model |
| `ingestion_method` | `chunk_method` | Provider parser/chunker |
| `ingestion_options_json.chunk_token_num` | `parser_config.chunk_token_num` | Target chunk size |
| `retrieval_candidate_k` | retrieval `top_k` | Candidate pool, not returned result count |
| `retrieval_top_k` | retrieval `page_size` | Ranked chunks returned to evaluation |
| `retrieval_similarity_threshold` | `similarity_threshold` | Minimum similarity |
| `retrieval_vector_similarity_weight` | `vector_similarity_weight` | Vector/keyword hybrid weight |
| `retrieval_keyword_enabled` | `keyword` | Enable keyword matching |
| `reranker_model` | `rerank_id` | Optional reranker |
| `generation_context_k` | prompt `top_n` | Chunks passed to the LLM |
| generation sampling fields | chat `llm` fields | Temperature, top-p, and penalties |

Do not report Recall@K using retrieval `top_k`; in this API it is the candidate
pool. Use the first K returned chunks (`page_size` / `retrieval_top_k`).

## Troubleshooting

- **401/403:** verify the variable named by `RAG_MANAGED_API_KEY_ENV` is exported
  in the same process and that it contains only the key value.
- **Index polling timeout:** inspect provider document status and
  `progress_msg`; increase the poll timeout only after ruling out failed parses.
- **No gold retrieval scores:** ensure normalized chunk metadata contains
  `source_id` or `doc_id` matching `NF-###`.
- **Duplicate provider datasets:** keep deterministic resource reuse enabled
  and use a unique dataset prefix per benchmark corpus/environment.
- **Unexpectedly large sweep:** plan first. Ingestion resources scale with the
  embedding × chunking configurations; retrieval/generation settings should
  reuse them.
- **Token counts look too high:** the current optimiseRAG OpenAI-compatible
  route reports Python string lengths in its `prompt_tokens` and
  `completion_tokens` fields. Treat them as provider-reported character-count
  proxies, not tokenizer-accurate usage, until the server implementation is
  corrected.
