# System Architecture

The project has two execution layers over a shared benchmarking core.

1. [[Benchmark Pipeline]] is the deterministic grid runner driven by `config.py` and `main.py`.
2. [[Agentic Runner]] is an adaptive LangGraph-style loop that proposes, runs, analyzes, and iterates on configurations.

Shared services:

- [[Dataset Layer]] normalizes Hugging Face datasets into standard benchmark records.
- [[Chunking and Retrieval]] splits documents, caches embeddings in Chroma, retrieves by similarity or MMR, and optionally applies HyDE.
- [[Generation Layer]] calls chat models, streams timing metrics, strips thinking tags, and validates answers.
- [[Evaluation and Metrics]] computes RAGAS metrics plus custom retrieval and answer-quality metrics.
- [[Reporting and Tracking]] writes JSON/CSV/Markdown/plots, logs MLflow runs, and configures tracing.

High-level flow:

```text
configuration
  -> dataset loading
  -> chunking and vector store build, unless direct mode
  -> retrieval, optional HyDE, optional reranking
  -> prompt templating and answer generation
  -> RAGAS evaluation
  -> custom metrics
  -> per-sample and aggregate result model
  -> reports, plots, MLflow, tracing
```

Key design choices:

- Runs are sequential because local model/GPU resources are expected to be shared.
- Vector stores are persisted in `.chroma/` and keyed by dataset/model/chunking fingerprint.
- Dataset adapters hide dataset-specific fields behind a standard shape.
- The autonomous runner reuses benchmark modules rather than owning a separate implementation.

Related notes:

- [[Benchmark Pipeline]]
- [[Agentic Runner]]
- [[Configuration Reference]]
- [[Known Risks and Gaps]]

