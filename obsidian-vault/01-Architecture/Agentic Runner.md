# Agentic Runner

Sources:

- [agentic/README.md](../agentic/README.md)
- [agentic/cli.py](../agentic/cli.py)
- [agentic/graph.py](../agentic/graph.py)
- [agentic/config_proposer.py](../agentic/config_proposer.py)
- [agentic/benchmark_runner.py](../agentic/benchmark_runner.py)
- [agentic/result_analyzer.py](../agentic/result_analyzer.py)

The agentic layer performs adaptive benchmark search:

```text
START -> propose -> run -> analyze -> propose or END
```

Responsibilities:

- `config_proposer_node`: uses an Ollama tool-calling model and fallback heuristics to choose the next `ExplorationConfig`.
- `benchmark_runner_node`: executes one benchmark configuration through the shared benchmark pipeline.
- `result_analyzer_node`: analyzes completed runs, extracts insights, selects the best config, and detects convergence.
- `should_continue`: loops until convergence or `max_iterations`.
- `agentic.cli`: parses CLI args, builds the initial state, invokes the graph, and writes the final exploration report.

Explored parameters include chunking strategy, chunk size/overlap, retrieval top-k, similarity vs MMR, HyDE, prompt template, and optional cross-encoder reranking.

Outputs:

- `results/agent_runN/agent_exploration_log.json`
- `results/agent_runN/configs/*.json`
- MLflow entries through shared tracking utilities

Operational assumptions:

- The agent brain is a local Ollama chat model with tool-call support.
- Benchmark LLM and embedding models are fixed for an agent run.
- If model tool calls fail, heuristic proposal keeps the run moving.

Current gaps and drift risks:

- The agentic pipeline duplicates much of `main.run_single_benchmark()`, so behavior can drift.
- Unlike `main.py`, agentic loading does not use shared-corpus dataset mode.
- Agentic runs do not currently expose `retrieval_mode="direct"`.
- `seed_config` is initialized in state but the graph starts at `propose`, so the first proposal can overwrite it.
- Data caching in state is ineffective if loaded data is not returned into graph state.
- The prompt says not to repeat configs, but LLM proposals are not fully duplicate-checked before execution.
- Agentic MLflow logging skips tracing, GenAI eval logging, plot logging, and aggregate report generation compared with `main.py`.

Related notes:

- [[Benchmark Pipeline]]
- [[Configuration Reference]]
- [[Operations Runbook]]
- [[Known Risks and Gaps]]
