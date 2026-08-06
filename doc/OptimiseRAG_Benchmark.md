# Run the optimiseRAG Benchmark

This runbook covers the concrete path from a running optimiseRAG deployment to
a reproducible retrieval or end-to-end benchmark. The implementation uses the
`optimaiserag` adapter; `ragflow` is an equivalent alias.

The generic adapter architecture and provider-field mapping are documented in
[Managed_RAG_System_Usage.md](Managed_RAG_System_Usage.md). The upstream API
contract is documented in the
[RAGFlow HTTP API reference](https://ragflow.io/docs/http_api_reference).

## What is still required

Before the first live run, confirm all of the following:

- optimiseRAG is reachable from the benchmark machine;
- you have an API key stored in a local environment variable;
- the embedding and generation model IDs in the manifest exactly match models
  configured in optimiseRAG;
- the deployment can parse Markdown/text documents and has enough storage for
  an indexed copy of the 60-document corpus;
- for full RAGAS evaluation, the separately configured critic LLM and embedding
  endpoints are reachable;
- the benchmark checkout is on a feature branch before you commit changes.

No changes inside the optimiseRAG source repository are required. The adapter
uses the public API and does not import code from the cloned server checkout.

## 1. Configure the connection

Keep URLs and credentials out of the experiment YAML. Add these values to your
uncommitted `.env`, or export them in the shell:

```dotenv
RAG_MANAGED_BASE_URL=http://spark:8011
RAG_MANAGED_API_KEY_ENV=OPTIMAISE_RAG_API_KEY
OPTIMAISE_RAG_API_KEY=replace-with-your-real-key
```

Use port `8011` for the described development deployment or `8010` for staging.
Do not place the key in a URL, YAML manifest, command argument, result file, or
Git commit.

Optional connectivity check:

```bash
curl --fail --silent --show-error \
  "$RAG_MANAGED_BASE_URL/v1/system/healthz"
```

If that deployment intentionally does not expose the health route, set the
trusted operator option below. Do not use arbitrary values from a dataset or
question as provider options.

```dotenv
RAG_MANAGED_OPTIONS_JSON={"health_check":false}
```

## 2. Set the deployed model names

Open either optimiseRAG manifest and replace these examples when necessary:

```yaml
matrix:
  llm_models:
    - qwen2.5:7b
  embedding_models:
    - BAAI/bge-m3
```

The embedding model is fixed when the optimiseRAG dataset is created. A sweep
therefore creates or reuses a separate provider dataset for every effective
embedding, parser configuration, and corpus-content fingerprint.

## 3. Validate locally before contacting optimiseRAG

```bash
python datasets/northstar_facilities/validate.py
python datasets/northstar_facilities/generate.py --check
python -m benchmark.worker plan \
  experiments/optimaiserag-northstar-smoke.yaml
```

Expected results:

- dataset validation reports `status: pass`, 360 questions, and 60 documents;
- reproducibility validation reports 360 questions and 60 sources;
- the smoke plan reports one configuration and five questions.

Planning does not upload documents or call the provider.

## 4. Run the five-question retrieval smoke test

Start with retrieval only. This creates or reuses a dataset, uploads the full
corpus, waits for parsing, and evaluates five questions without creating a chat
or calling the generation LLM.

```bash
BENCHMARK_STAGE=retrieve \
python -m benchmark.worker run \
  experiments/optimaiserag-northstar-smoke.yaml \
  --run-dir results/optimaiserag-smoke-retrieval \
  --keep-going
```

Confirm that:

- all 60 provider documents finish parsing;
- returned metadata contains stable IDs such as `NF-001`;
- `hit@k`, `ndcg@k`, `recall@k`, and `span_*` metrics are present;
- no credential appears in the terminal or result artifacts.

The span metrics check the exact gold evidence text. Retrieving an irrelevant
section from the correct document can receive document-level credit but not
span-level credit.

## 5. Run the five-question generation smoke test

Omit `BENCHMARK_STAGE=retrieve` to create/reuse the chat and request answers.
The smoke manifest disables model-based evaluators so this tests the provider
contract without requiring the critic models.

```bash
python -m benchmark.worker run \
  experiments/optimaiserag-northstar-smoke.yaml \
  --run-dir results/optimaiserag-smoke-generation \
  --keep-going
```

Inspect the per-question answer, returned contexts, references, and provider
timings before launching the complete benchmark.

## 6. Run all 360 questions

Retrieval-only benchmark:

```bash
BENCHMARK_STAGE=retrieve \
python -m benchmark.worker run \
  experiments/optimaiserag-northstar.yaml \
  --run-dir results/optimaiserag-retrieval \
  --keep-going
```

End-to-end retrieval plus generation and configured evaluators:

```bash
python -m benchmark.worker run \
  experiments/optimaiserag-northstar.yaml \
  --run-dir results/optimaiserag-full \
  --keep-going
```

The full manifest enables RAGAS and custom metrics. Its critic LLM and critic
embedding model are benchmark-side evaluation dependencies; they do not need
to be the same models used by optimiseRAG.

## 7. Configure a sweep

Use a copied experiment manifest for each study. The following example sweeps
embedding, chunk size, threshold, hybrid weight, and returned result count:

```yaml
matrix:
  embedding_models:
    - BAAI/bge-m3
    - your-second-embedding-model
  ingestion_options_json:
    - {chunk_token_num: 128}
    - {chunk_token_num: 256}
    - {chunk_token_num: 512}
  retrieval_similarity_threshold:
    - 0.1
    - 0.2
    - 0.35
  retrieval_vector_similarity_weight:
    - 0.3
    - 0.6
    - 0.9
  retrieval_top_k:
    - 5
    - 10
```

Always run `benchmark.worker plan` first. The example above contains 108
configurations before any generation-model or reranker variants. Start with
retrieval-only evaluation, then carry only the strongest settings into the
generation benchmark.

## 8. Resource reuse and cleanup

The supplied manifests use:

```yaml
rag_managed_reuse_resources: true
rag_managed_cleanup: false
```

This is recommended during development because repeated retrieval/generation
settings can reuse an identical indexed dataset and failed resources remain
available for inspection. Dataset identity includes the complete corpus hash,
embedding model, parser method, and parser configuration, preventing reuse of
a stale corpus under the same prefix.

For isolated CI runs, set `rag_managed_cleanup: true`. Cleanup deletes only
datasets/chats created by that run; reused provider resources are never deleted.

## 9. Find the results

Each worker run writes into the selected `--run-dir`:

- `progress.json`: resume/failure state for every configuration;
- `configs/*.json`: normalized result per configuration;
- `configs/*_qa.json`: question/answer log;
- `worker_manifest.json`: experiment and configuration identities;
- `reproducibility/`: environment, Git state, and package snapshot;
- generated CSV, Markdown, and plot reports when reporting is enabled.

Re-run the same command and run directory to resume completed matrices. Use
`--no-resume` only when every configuration should execute again.

## 10. Known provider limitations

- optimiseRAG does not expose idempotency keys for creation, upload, parsing,
  chat, or completion calls. The adapter therefore does not automatically retry
  those state-changing or billable operations.
- Provider chunk/document IDs are generated during ingestion. The adapter maps
  deterministic upload names and returned document IDs back to stable gold
  source IDs.
- The current optimiseRAG OpenAI-compatible response labels Python string
  lengths as token usage. Treat those values as character-count proxies, not
  tokenizer-accurate token counts.
- There is no capabilities endpoint. Unsupported adapter options fail locally
  instead of being silently ignored.

## Troubleshooting

- **401/403:** verify `OPTIMAISE_RAG_API_KEY` is exported in the worker process
  and that `rag_managed_api_key_env` contains the variable name, not the key.
- **Health check failure:** confirm the base URL/port, or disable only the health
  probe with the trusted option shown above.
- **Parsing timeout:** inspect optimiseRAG document `progress` and
  `progress_msg`; increase `rag_managed_poll_timeout_seconds` only after ruling
  out a parser failure.
- **Model not found:** replace the example model identifier with the exact model
  name configured in optimiseRAG.
- **Zero retrieval scores:** verify returned metadata maps to `NF-###`; opaque
  document IDs alone cannot match the gold fixture.
- **Unexpected duplicate datasets:** keep reuse enabled and use a stable dataset
  prefix. A changed corpus, embedding model, or chunk setting intentionally gets
  a new dataset.
