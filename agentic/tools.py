"""Tool definitions and pipeline execution for the autonomous agent.

Tools exposed to the agent LLM are decorated with @tool.
The main pipeline function (run_full_pipeline) is called programmatically
by the benchmark_runner node, not by the LLM.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path

from langchain_core.tools import tool
from rich.console import Console

from agentic.state import AgentState, ExplorationConfig, ExplorationResult

console = Console()
logger = logging.getLogger(__name__)



# ---------------------------------------------------------------------------
# Tools exposed to the agent LLM
# ---------------------------------------------------------------------------

@tool
def list_available_models(base_url: str = "http://localhost:11434") -> list[str]:
    """List Ollama models available on the local server."""
    try:
        result = subprocess.run(
            ["ollama", "list", "--format", "json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass
    return []


@tool
def propose_config(
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_top_k: int,
    retrieval_strategy: str,
    retrieval_use_hyde: bool,
    prompt_template: str,
    reranker_model: str | None = None,
) -> dict:
    """Propose the next benchmark configuration to explore.

    Args:
        chunking_strategy: One of "recursive", "character", "token", "semantic".
        chunk_size: Size of each chunk (200-1500).
        chunk_overlap: Overlap between chunks, must be < chunk_size.
        retrieval_top_k: Number of documents to retrieve (3-15).
        retrieval_strategy: "similarity" or "mmr".
        retrieval_use_hyde: Whether to use HyDE query expansion.
        prompt_template: "concise", "detailed", or "finqa".
        reranker_model: null for no reranker, or model identifier.
    """
    return {
        "chunking_strategy": chunking_strategy,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "retrieval_top_k": retrieval_top_k,
        "retrieval_strategy": retrieval_strategy,
        "retrieval_use_hyde": retrieval_use_hyde,
        "prompt_template": prompt_template,
        "reranker_model": reranker_model,
    }


# ---------------------------------------------------------------------------
# Internal: full pipeline execution via shared benchmark core
# ---------------------------------------------------------------------------

def run_full_pipeline(
    config: ExplorationConfig,
    state: AgentState,
) -> tuple[BenchmarkResultExtended | None, ExplorationResult]:
    """Execute one proposed config through the shared benchmark pipeline."""
    from dataclasses import replace
    from config import get_all_combinations
    from main import run_single_benchmark
    from benchmark.orchestration.worker import _load_data_once, save_config_result
    from benchmark.providers import parse_model_id
    from benchmark.tracking import log_benchmark_run

    run_start = time.perf_counter()
    try:
        base = get_all_combinations()[0]
        llm_provider, llm_model_name = parse_model_id(config["llm_model"])
        chunk_size = config["chunk_size"]
        chunk_overlap = config["chunk_overlap"]
        if config["chunking_strategy"] == "semantic":
            chunk_size = None
            chunk_overlap = None

        bench_config = replace(
            base,
            llm_provider=llm_provider,
            llm_model=llm_model_name,
            embedding_model=config["embedding_model"],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=config["chunking_strategy"],
            retrieval_top_k=config["retrieval_top_k"],
            retrieval_strategy=config["retrieval_strategy"],
            retrieval_use_hyde=config["retrieval_use_hyde"],
            prompt_template=config["prompt_template"],
            reranker_model=config.get("reranker_model"),
            dataset_name=state.get("dataset_name", base.dataset_name),
            dataset_subset=state.get("dataset_subset", base.dataset_subset) or "",
            dataset_sample_size=state.get("dataset_sample_size", base.dataset_sample_size),
            ollama_base_url=state.get("ollama_base_url", base.ollama_base_url),
        )

        console.print(f"\n[bold yellow]>>> Agent running: {bench_config.name}[/bold yellow]")

        cached_data = state.get("_data")
        if cached_data is None:
            data, corpus, load_data_seconds = _load_data_once(bench_config)
        else:
            data = cached_data
            corpus = None
            load_data_seconds = None

        run_dir = _ensure_agent_run_dir(Path("results"))
        full_result = run_single_benchmark(
            bench_config,
            data,
            run_dir=run_dir,
            corpus=corpus,
            load_data_seconds=load_data_seconds,
        )
        save_config_result(full_result, run_dir)

        try:
            log_benchmark_run(full_result)
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

        metrics = _compact_metrics(full_result)
        return full_result, ExplorationResult(
            config=config,
            metrics=metrics,
            total_time_seconds=full_result.total_time_seconds,
            num_chunks=full_result.num_chunks,
            error=full_result.evaluation_error,
        )
    except Exception as exc:
        logger.exception("Pipeline failed for proposed agent config")
        return None, ExplorationResult(
            config=config,
            metrics={},
            total_time_seconds=time.perf_counter() - run_start,
            num_chunks=0,
            error=str(exc),
        )


def _compact_metrics(result: BenchmarkResultExtended) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for key in (
        "ragas_faithfulness",
        "ragas_answer_relevancy",
        "ragas_answer_correctness",
        "ragas_context_precision",
        "ragas_context_recall",
        "ragas_semantic_similarity",
        "avg_ttft_seconds",
        "avg_tokens_per_second",
    ):
        value = getattr(result, key)
        if value is not None:
            metrics[key] = value
    if result.custom_metric_means:
        metrics.update(result.custom_metric_means)
    return metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_agent_run_dir: Path | None = None


def _ensure_agent_run_dir(base: Path) -> Path:
    """Get or create the agent run directory for this session."""
    global _agent_run_dir
    if _agent_run_dir is not None and _agent_run_dir.exists():
        return _agent_run_dir

    base.mkdir(parents=True, exist_ok=True)
    max_run = 0
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("agent_run"):
            try:
                n = int(child.name[len("agent_run"):])
                max_run = max(max_run, n)
            except ValueError:
                pass
    run_dir = base / f"agent_run{max_run + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _agent_run_dir = run_dir
    return run_dir


def reset_agent_run_dir() -> None:
    """Reset the cached run dir (used between sessions)."""
    global _agent_run_dir
    _agent_run_dir = None
