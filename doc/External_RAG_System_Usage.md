# External RAG System Integration

This document describes how to use the benchmark as a lightweight evaluation framework for any RAG system. The recommended default is the HTTP adapter: your RAG system stays independent and only needs one JSON `POST` endpoint.

## Quick Smoke Test

Start the bundled demo RAG endpoint:

```bash
python examples/http_rag_server.py
```

Run the benchmark against it with the sample JSONL dataset and expensive model-based metrics disabled:

```bash
RAG_SYSTEM_ADAPTER=http RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query DATASET_NAME=jsonl DATASET_PATH=examples/sample_dataset.jsonl RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py
```

This verifies the adapter contract, local dataset loading, answer capture, timing capture, result writing, and report generation without requiring a critic model or embedding model for evaluation.

## HTTP Adapter

Use this mode for existing RAG services, regardless of their internal stack.

```bash
RAG_SYSTEM_ADAPTER=http RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query RAG_HTTP_ANSWER_FIELD=answer RAG_HTTP_CONTEXTS_FIELD=contexts python main.py
```

In HTTP mode, the benchmark skips its internal chunking, retrieval, reranking, and generation. Your service owns the RAG pipeline. The framework sends benchmark questions, normalizes the response, evaluates the output, and writes reports/MLflow artifacts.

## Request Contract

For every benchmark sample, the framework sends a JSON `POST` body:

```json
{
  "question": "What is the answer?",
  "metadata": {
    "id": "sample-1"
  },
  "ground_truth": "Expected answer",
  "config": {
    "name": "recursive_cs1000_co200_model_llm_concise_http",
    "retrieval_top_k": 5,
    "prompt_template": "concise",
    "dataset_name": "jsonl"
  }
}
```

Your service should return a JSON object.

Minimal response:

```json
{
  "answer": "Generated answer"
}
```

Recommended response:

```json
{
  "answer": "Generated answer",
  "contexts": [
    "First retrieved passage",
    "Second retrieved passage"
  ],
  "metadata": [
    {"doc_id": "doc-1", "score": 0.91},
    {"doc_id": "doc-2", "score": 0.84}
  ],
  "timings": {
    "ttft_seconds": 0.12,
    "total_seconds": 1.47,
    "token_count": 128
  },
  "input_tokens": 42,
  "output_tokens": 128,
  "total_tokens": 170,
  "estimated_cost_usd": 0.0012
}
```

Return `contexts` whenever possible. RAGAS context metrics, groundedness checks, and retrieval-style custom metrics are only meaningful when retrieved evidence is present.

## Nested Response Fields

If your API response is nested, map fields with dotted paths:

```bash
RAG_SYSTEM_ADAPTER=http RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query RAG_HTTP_ANSWER_FIELD=result.answer RAG_HTTP_CONTEXTS_FIELD=result.sources RAG_HTTP_METADATA_FIELD=result.source_metadata RAG_HTTP_TIMINGS_FIELD=metrics python main.py
```

Context entries can be strings or objects. For object entries, the adapter checks these text keys: `text`, `content`, `page_content`, `context`.

## Headers And Auth

Static headers can be supplied as JSON:

```bash
RAG_HTTP_HEADERS='{"X-Project":"benchmark"}'
```

For a single auth header, use environment variables so secrets stay out of source files:

```bash
RAG_HTTP_AUTH_HEADER=Authorization RAG_HTTP_AUTH_VALUE="Bearer $RAG_API_TOKEN" python main.py
```

## Local JSONL And CSV Datasets

Use local datasets when you want to evaluate your own questions instead of built-in Hugging Face datasets.

JSONL example:

```jsonl
{"question":"What is X?","ground_truth":"X is ...","context":"Optional reference context","metadata":{"id":"1"}}
```

Run it:

```bash
DATASET_NAME=jsonl DATASET_PATH=path/to/samples.jsonl DATASET_QUESTION_FIELD=question DATASET_GROUND_TRUTH_FIELD=ground_truth DATASET_CONTEXT_FIELD=context DATASET_METADATA_FIELD=metadata python main.py
```

CSV uses the same field variables:

```bash
DATASET_NAME=csv DATASET_PATH=path/to/samples.csv python main.py
```

CSV `metadata` may be blank, a plain string, or a JSON object encoded as a string. JSONL rows must be JSON objects.

## Metric Controls

For fast endpoint smoke tests or answer-only checks:

```bash
RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false python main.py
```

- `RAGAS_ENABLED=false` skips RAGAS critic calls.
- `CUSTOM_METRICS_ENABLED=false` skips custom embedding/BERTScore metrics.
- Reports still include answers, contexts, timings, token counts, validity, and per-sample records.

Use full metrics when you want quality evaluation:

```bash
RAGAS_ENABLED=true CUSTOM_METRICS_ENABLED=true python main.py
```

That requires the configured critic LLM and embedding provider to be reachable.

## Native Python Adapter Plugins

Use a native adapter when HTTP is not enough and the benchmark should call Python code directly.

Create a module, for example `my_package/my_adapter.py`:

```python
from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput


class MyRagAdapter:
    name = "myrag"

    def prepare(self, config, data, corpus=None):
        # Optional setup/indexing step.
        return None

    def answer(self, sample, config):
        result = my_rag_pipeline.query(sample["question"])
        return RagSystemOutput(
            answer=result.answer,
            contexts=result.contexts,
            metadata=result.metadata,
            total_seconds=result.total_seconds,
            token_count=result.token_count,
            answer_valid=bool(result.answer.strip()),
        )


register_rag_adapter("myrag", lambda config: MyRagAdapter())
```

Run it by importing the module before adapter validation:

```bash
RAG_ADAPTER_MODULES=my_package.my_adapter RAG_SYSTEM_ADAPTER=myrag python main.py
```

## Important Variables

| Variable | Purpose |
| --- | --- |
| `RAG_SYSTEM_ADAPTER` | `internal`, `http`, or a registered custom adapter. |
| `RAG_ADAPTER_MODULES` | Comma-separated Python modules imported before adapter validation. |
| `RAG_HTTP_ENDPOINT_URL` | Required for `RAG_SYSTEM_ADAPTER=http`. |
| `RAG_HTTP_ANSWER_FIELD` | Dotted response path for answer text. |
| `RAG_HTTP_CONTEXTS_FIELD` | Dotted response path for retrieved contexts. |
| `RAG_HTTP_METADATA_FIELD` | Dotted response path for retrieval metadata. |
| `RAG_HTTP_TIMINGS_FIELD` | Dotted response path for timing data. |
| `RAG_HTTP_HEADERS` | JSON object of static request headers. |
| `RAG_HTTP_AUTH_HEADER` | Optional single auth header name. |
| `RAG_HTTP_AUTH_VALUE` | Optional single auth header value. |
| `DATASET_NAME` | Built-in dataset name, or `jsonl`/`csv` for local files. |
| `DATASET_PATH` | Required for local JSONL/CSV datasets. |
| `DATASET_QUESTION_FIELD` | Question field name for local datasets. |
| `DATASET_GROUND_TRUTH_FIELD` | Ground-truth answer field name for local datasets. |
| `DATASET_CONTEXT_FIELD` | Context field name for local datasets. |
| `DATASET_METADATA_FIELD` | Metadata field name for local datasets. |
| `RAGAS_ENABLED` | Disable with `false` to skip RAGAS. |
| `CUSTOM_METRICS_ENABLED` | Disable with `false` to skip custom metrics. |

## Recommended Integration Path

1. Start with `examples/http_rag_server.py` and `examples/sample_dataset.jsonl`.
2. Replace `RAG_HTTP_ENDPOINT_URL` with your service endpoint.
3. Use `DATASET_NAME=jsonl` and `DATASET_PATH=...` for your own question set.
4. Keep `RAGAS_ENABLED=false CUSTOM_METRICS_ENABLED=false` until the endpoint contract is stable.
5. Enable metrics once answer/context/timing fields are correct.
6. Move to a native Python adapter only if HTTP cannot express your setup needs.
