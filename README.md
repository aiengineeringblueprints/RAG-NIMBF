# RAG Benchmarking Framework

This project benchmarks RAG systems across datasets, retrieval outputs, answer
quality metrics, runtime measurements, reports, and MLflow tracking.

The framework can run its built-in RAG pipeline, or evaluate an external RAG
system as a black-box HTTP service.

## Install

```bash
pip install -r requirements.txt
```

For a fully reproducible Python 3.12 environment, install the compiled lock:

```bash
pip install -r requirements-lock.txt
```

Regenerate it after dependency changes with the command recorded at the top of
`requirements-lock.txt`.

## Quick Start

Run the built-in pipeline (chunks/indexes the dataset, retrieves, generates,
evaluates, writes reports under `results/`, logs to MLflow):

```bash
BENCHMARK_CONFIG_FILE=experiments/full-grid-example.yaml python main.py
```

`python main.py` requires a manifest: it reads `BENCHMARK_CONFIG_FILE` or an
explicit path argument and exits with an actionable error when neither is set.
Keep models, chunking, retrieval, datasets, and evaluator settings in YAML;
keep API keys and machine-local service URLs in `.env`.

For long, resumable experiment matrices use the worker — it expands the
manifest into configs, writes every result immediately, and records resume
state in `results/runN/progress.json`:

```bash
python -m benchmark.worker plan experiments/full-grid-example.yaml
python -m benchmark.worker run experiments/full-grid-example.yaml --keep-going
# resume an interrupted run:
python -m benchmark.worker run experiments/full-grid-example.yaml --run-dir results/run3
```

Optional add-ons, documented separately:

- LLM load/latency testing inside the same run: [doc/llm-load-test.md](doc/llm-load-test.md)
- Unattended execution via ClearML agent: [doc/clearml-agent.md](doc/clearml-agent.md)

## Benchmark Your Own RAG System

Use this framework as a drop-in evaluation layer for an existing RAG system.
Choose an integration mode — full guide with dataset prep and minimal manifest:
**[doc/Benchmark_Your_RAG.md](doc/Benchmark_Your_RAG.md)**.

| Mode | Env | Guide |
| --- | --- | --- |
| HTTP endpoint | `RAG_SYSTEM_ADAPTER=http` | [External_RAG_System_Usage.md](doc/External_RAG_System_Usage.md) |
| Python plugin (no HTTP) | `RAG_SYSTEM_ADAPTER=myrag` + `RAG_ADAPTER_MODULES` | [External_RAG_Python_Adapter.md](doc/External_RAG_Python_Adapter.md) |
| Managed/hosted system with own index | `RAG_SYSTEM_ADAPTER=ragflow` | [Managed_RAG_System_Usage.md](doc/Managed_RAG_System_Usage.md) |
| MCP server / MCP host system | `RAG_SYSTEM_ADAPTER=mcp` | [mcp-benchmarking.md](doc/mcp-benchmarking.md), [mcp-host-benchmarking.md](doc/mcp-host-benchmarking.md) |

Fastest smoke test — run the bundled demo endpoint and sample dataset:

```bash
python examples/http_rag_server.py

RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
DATASET_NAME=jsonl \
DATASET_PATH=examples/sample_dataset.jsonl \
RAGAS_ENABLED=false \
CUSTOM_METRICS_ENABLED=false \
python main.py
```

Ready-made comparisons: Enterprise RAG Blueprint
([doc/Enterprise_RAG_Blueprint_Benchmark.md](doc/Enterprise_RAG_Blueprint_Benchmark.md)),
OptimiseRAG ([doc/OptimiseRAG_Benchmark.md](doc/OptimiseRAG_Benchmark.md)),
RAG-vs-MCP manifests `experiments/rag-vs-mcp*.yaml`.

## Benchmark Document Parsers

Your own parser (OCR / PDF-to-Markdown):
[doc/Benchmark_Your_Parser.md](doc/Benchmark_Your_Parser.md). Ready-to-run
manifests: `experiments/ocr_parser_*.yaml`. The document GT datasets
(`omnidocbench`, `dp-bench`, `olmocr-bench`) are parsing-stage-only and never
enter the RAG retrieval/QA evaluation path.

## Configuration

YAML-first: workflow settings belong in `experiments/*.yaml`; secrets and
machine-local URLs go in `.env` only. Every manifest field left unset falls
back to its env variable.

Full variable reference (retrieval, datasets, MCP, parsers, evaluation,
monitoring): **[doc/env-vars.md](doc/env-vars.md)**

## Results and MLflow

Runs write to `results/runN/`; everything lands in MLflow for comparison.
Rescoring finished runs with the LLM answer judge, compacting reports, and
MLflow UI usage: **[doc/results-and-mlflow.md](doc/results-and-mlflow.md)**.

## Documentation Index

All guides, gap analyses, and the change log: **[doc/00-README.md](doc/00-README.md)**

## Publication

- [RAG-NIMBF: A Non-Invasive, Modular Benchmarking Framework for Automated RAG Configuration Analysis](https://doi.org/10.48550/arXiv.2610.27816) — Niclas Hart and Nicolas Weeger, Ansbach University of Applied Sciences. Describes the framework architecture, the hybrid evaluation methodology, and an empirical configuration study across six generation models on SQuAD.
