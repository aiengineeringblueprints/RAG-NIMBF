# Operations Runbook

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the deterministic benchmark:

```bash
python main.py
```

Build/reuse vector indexes in separate stages:

```bash
BENCHMARK_STAGE=index python main.py
BENCHMARK_STAGE=query python main.py
```

Use LanceDB instead of Chroma:

```bash
VECTOR_DB_BACKEND=lancedb LANCEDB_PATH=.lancedb python main.py
```

Run the autonomous agent:

```bash
python -m agentic.cli --agent-model qwen3:8b --max-iterations 2 --sample-size 5
```

Run tests:

```bash
python -m unittest
```

Expected local services:

- Ollama at `OLLAMA_BASE_URL`, default `http://localhost:11434`, for local LLMs and embeddings.
- Optional OpenAI-compatible endpoint through `OPENAI_COMPAT_BASE_URL`.
- Optional MLflow UI can inspect `mlflow.db` and `mlruns/`.

Important runtime folders:

- `results/`: benchmark reports and per-config output.
- `.chroma/`: persisted Chroma cache.
- `.lancedb/`: persisted LanceDB tables when `VECTOR_DB_BACKEND=lancedb`.
- `mlruns/`, `mlflow.db`: MLflow tracking artifacts.

Before long runs:

- Confirm `.env` values from [[Configuration Reference]].
- Confirm required Ollama models are pulled.
- Use small `DATASET_SAMPLE_SIZE` for smoke tests.
- Be aware that RAGAS critic calls and BERTScore can dominate runtime.

Cleanup caution:

- Do not delete `.chroma/`, `results/`, `mlruns/`, or `mlflow.db` unless the user explicitly asks. They contain cache/history.

Related notes:

- [[Benchmark Pipeline]]
- [[Agentic Runner]]
- [[Reporting and Tracking]]
