"""Entry point for the autonomous RAG benchmarking agent.

Usage:
    python -m agentic.cli --agent-model qwen3:8b --max-iterations 8

    # Quick test with a small sample
    python -m agentic.cli --agent-model llama3.1:8b --max-iterations 2 --sample-size 5
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Autonomous RAG Benchmarking Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agent-model", default="qwen3:8b",
        help="Ollama model for the agent LLM (default: qwen3:8b)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset name (default: from .env or t2-ragbench)",
    )
    parser.add_argument(
        "--subset", default=None,
        help="Dataset subset (default: from .env or FinQA)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of samples to benchmark (default: from .env or 20)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=8,
        help="Maximum exploration iterations (default: 8)",
    )
    parser.add_argument(
        "--llm-model", default=None,
        help="LLM model for benchmark generation (default: from .env or gemma3:4b)",
    )
    parser.add_argument(
        "--embedding-model", default=None,
        help="Embedding model (default: from .env or nomic-embed-text:latest)",
    )
    parser.add_argument(
        "--ollama-url", default=None,
        help="Override OLLAMA_BASE_URL",
    )
    args = parser.parse_args()

    load_dotenv()

    # Defaults from env
    ollama_url = args.ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    dataset_name = args.dataset or os.getenv("DATASET_NAME", "t2-ragbench")
    dataset_subset = args.subset or os.getenv("DATASET_SUBSET", "FinQA")
    sample_size = args.sample_size or int(os.getenv("DATASET_SAMPLE_SIZE", "20"))
    llm_model = args.llm_model or os.getenv("LLM_MODELS", "gemma3:4b").split(",")[0].strip()
    emb_model = (
        args.embedding_model
        or os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest").split(",")[0].strip()
    )

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    console.print("[bold]Autonomous RAG Benchmarking Agent[/bold]")
    console.print("=" * 50)
    console.print(f"  Agent model:    {args.agent_model}")
    console.print(f"  Benchmark LLM:  {llm_model}")
    console.print(f"  Embedding:      {emb_model}")
    console.print(f"  Dataset:        {dataset_name}/{dataset_subset} (n={sample_size})")
    console.print(f"  Max iterations: {args.max_iterations}")
    console.print(f"  Ollama URL:     {ollama_url}")
    console.print("=" * 50)

    # Setup MLflow
    from benchmark.tracking import setup_mlflow
    tracking_uri = setup_mlflow()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")

    # Build initial state
    from agentic.state import AgentState, make_seed_config
    from agentic.tools import reset_agent_run_dir

    reset_agent_run_dir()

    seed_config = make_seed_config(llm_model=llm_model, embedding_model=emb_model)

    initial_state: AgentState = {
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "dataset_sample_size": sample_size,
        "ollama_base_url": ollama_url,
        "agent_model": args.agent_model,
        "max_iterations": args.max_iterations,
        "completed_runs": [],
        "insights": [],
        "iteration": 0,
        "current_config": seed_config,
        "current_result": None,
        "analysis_text": "",
        "phase": "run",
        "should_continue": True,
        "_data": None,
        # Hidden fields used by proposer to know which models to use
        "_default_llm": llm_model,
        "_default_emb": emb_model,
    }

    # Build and run graph
    from agentic.graph import build_agent_graph

    graph = build_agent_graph()

    console.print("\n[bold green]Starting autonomous exploration...[/bold green]\n")

    final_state = graph.invoke(initial_state)

    # Generate final report
    _generate_final_report(final_state)

    # Print summary
    completed = final_state.get("completed_runs", [])
    console.print(f"\n[bold]Exploration complete: {len(completed)} configurations tested[/bold]")

    if completed:
        best = max(completed, key=lambda r: r["metrics"].get("ragas_faithfulness", 0) or 0)
        c = best["config"]
        m = best["metrics"]
        console.print(f"\n[bold green]Best configuration:[/bold green]")
        console.print(f"  strategy={c.get('chunking_strategy')} chunk_size={c.get('chunk_size')} "
                       f"overlap={c.get('chunk_overlap')} top_k={c.get('retrieval_top_k')} "
                       f"hyde={c.get('retrieval_use_hyde')} template={c.get('prompt_template')}")
        console.print(f"  faithfulness={m.get('ragas_faithfulness', 'N/A')} "
                       f"answer_relevancy={m.get('ragas_answer_relevancy', 'N/A')}")

    console.print("\n[bold]All insights:[/bold]")
    for ins in final_state.get("insights", []):
        console.print(f"  - {ins}")


def _generate_final_report(state: dict) -> None:
    """Generate comparative report and exploration log."""
    from agentic.tools import _ensure_agent_run_dir, reset_agent_run_dir
    from agentic.state import ExplorationResult

    run_dir = _ensure_agent_run_dir(Path("results"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    completed: list[ExplorationResult] = state.get("completed_runs", [])

    # Save exploration log
    log_path = run_dir / "agent_exploration_log.json"
    log_data = {
        "timestamp": timestamp,
        "agent_model": state.get("agent_model"),
        "dataset": f"{state.get('dataset_name')}/{state.get('dataset_subset')}",
        "sample_size": state.get("dataset_sample_size"),
        "iterations": state.get("iteration", 0),
        "insights": state.get("insights", []),
        "runs": completed,
    }
    log_path.write_text(json.dumps(log_data, indent=2, default=str))
    console.print(f"[dim]Exploration log saved to {log_path}[/dim]")

    # Try to generate the full comparative report if we have BenchmarkResultExtended data
    configs_dir = run_dir / "configs"
    if configs_dir.exists() and any(configs_dir.glob("*.json")):
        console.print(f"[dim]Individual config results saved in {configs_dir}[/dim]")


if __name__ == "__main__":
    main()
