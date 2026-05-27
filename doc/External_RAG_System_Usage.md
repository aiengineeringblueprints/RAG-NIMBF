So how is the usage with different RAG systems now?


• You now have two integration levels.

  1. Black-box external RAG system

  Use the HTTP adapter. Your RAG service only needs a JSON POST endpoint.

  RAG_SYSTEM_ADAPTER=http \
  RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
  RAG_HTTP_ANSWER_FIELD=answer \
  RAG_HTTP_CONTEXTS_FIELD=contexts \
  python main.py

  The benchmark sends:

  {
    "question": "...",
    "metadata": {...},
    "ground_truth": "...",
    "config": {
      "name": "...",
      "retrieval_top_k": 5,
      "prompt_template": "concise",
      "dataset_name": "..."
    }
  }

  Your endpoint should return something like:

  {
    "answer": "...",
    "contexts": ["retrieved context 1", "retrieved context 2"],
    "metadata": [{"doc_id": "doc-1"}],
    "timings": {
      "ttft_seconds": 0.2,
      "total_seconds": 1.4
    },
    "token_count": 120
  }

  If your response is nested, configure dotted paths:

  RAG_HTTP_ANSWER_FIELD=result.answer
  RAG_HTTP_CONTEXTS_FIELD=result.sources
  RAG_HTTP_METADATA_FIELD=result.metadata
  RAG_HTTP_TIMINGS_FIELD=metrics

  2. First-class Python adapter

  For a native integration, register an adapter factory:

  from benchmark.adapters import register_rag_adapter
  from benchmark.adapters.base import RagSystemOutput

  class MyRagAdapter:
      name = "myrag"

      def prepare(self, config, data, corpus=None):
          # optional indexing/setup
          pass

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

  Then:

  RAG_SYSTEM_ADAPTER=myrag python main.py

  For vector DBs, same pattern now exists through
  register_vector_store_backend(), while built-ins remain:

  VECTOR_DB_BACKEND=chroma python main.py
  VECTOR_DB_BACKEND=lancedb python main.py

  So practically: use HTTP for fastest integration with any existing RAG
  service; use the Python registry when you want the framework to own setup/
  indexing/query logic directly.
