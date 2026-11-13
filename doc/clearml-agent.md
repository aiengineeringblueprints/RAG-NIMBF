# ClearML Agent Workflow

Use ClearML when you want to clone a benchmark task in the Web UI, edit
hyperparameters, enqueue it, and let a `clearml-agent` execute it on a worker
machine.

## Create the baseline task

From `.env` or from the first expanded config in a manifest:

```bash
python -m benchmark.clearml_task experiments/full-grid-example.yaml \
  --project-name "RAG Benchmarking" \
  --task-name rag_eval_baseline
```

The task exposes non-secret `BenchmarkConfig` fields such as `llm_model`,
`embedding_model`, `chunk_size`, `chunk_overlap`, `retrieval_top_k`,
`prompt_template`, `reranker_model`, and dataset settings in ClearML
Hyperparameters. API keys, auth headers, and raw HTTP headers are intentionally
not published to ClearML.

## Run an agent

Start an agent on the execution machine:

```bash
clearml-agent daemon --queue rag-benchmark-gpu
```

Then clone the baseline task in the ClearML UI, edit Hyperparameters, and
enqueue the clone to `rag-benchmark-gpu`. For a local smoke submission you can
also let the script enqueue itself and exit locally:

```bash
python -m benchmark.clearml_task --remote-queue rag-benchmark-gpu
```

## Behaviour

The ClearML task still runs the existing worker core, writes `results/runN/`,
logs scalar benchmark metrics to ClearML, and keeps MLflow logging enabled
unless `--no-mlflow` is passed.
