"""Config proposer node: LLM proposes next config with heuristic fallback."""

from __future__ import annotations

import logging
import random
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console

from agentic.prompts import CONFIG_PROPOSER_SYSTEM_PROMPT
from agentic.state import (
    AgentState,
    ExplorationConfig,
    VALID_CONFIG_RANGES,
)

console = Console()
logger = logging.getLogger(__name__)


def config_proposer_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: have the LLM propose the next benchmark config."""
    iteration = state.get("iteration", 0)

    console.print(f"\n[bold cyan]Iteration {iteration + 1}: Proposing config...[/bold cyan]")

    llm = ChatOllama(
        model=state.get("agent_model", "qwen3:8b"),
        base_url=state.get("ollama_base_url", "http://localhost:11434"),
        temperature=0.7,
    )

    from agentic.tools import propose_config
    llm_with_tools = llm.bind_tools([propose_config])

    messages = [
        SystemMessage(content=CONFIG_PROPOSER_SYSTEM_PROMPT),
        HumanMessage(content=_build_proposer_input(state)),
    ]

    proposed: ExplorationConfig | None = None

    try:
        response = llm_with_tools.invoke(messages)
        if response.tool_calls:
            args = response.tool_calls[0]["args"]
            proposed = _merge_proposal(args, state)
            console.print(f"  [dim]LLM proposed config via tool call[/dim]")
    except Exception as exc:
        logger.warning("LLM tool call failed: %s", exc)

    if proposed is None:
        proposed = _heuristic_next_config(state)
        console.print(f"  [dim]Using heuristic fallback config[/dim]")

    # Validate
    proposed = _validate_config(proposed)

    return {
        "current_config": proposed,
        "phase": "run",
    }


def _build_proposer_input(state: AgentState) -> str:
    """Format completed runs as a compact table for the LLM."""
    lines: list[str] = []

    completed = state.get("completed_runs", [])
    if not completed:
        lines.append("No previous runs. Propose a good starting configuration.")
        lines.append(f"Use llm_model={state.get('_default_llm', 'gemma3:4b')} and "
                      f"embedding_model={state.get('_default_emb', 'nomic-embed-text:latest')}.")
        return "\n".join(lines)

    lines.append(f"Previous runs ({len(completed)} completed):\n")
    lines.append("| # | strategy | chunk_sz | overlap | top_k | retrieval | hyde | template | reranker | faithfulness | answer_rel | avg_ttft |")
    lines.append("|---|----------|----------|---------|-------|-----------|------|----------|----------|-------------|------------|----------|")

    for i, run in enumerate(completed):
        c = run["config"]
        m = run["metrics"]
        faith = m.get("ragas_faithfulness", "N/A")
        ans_rel = m.get("ragas_answer_relevancy", "N/A")
        ttft = m.get("avg_ttft_seconds", "N/A")
        if isinstance(faith, float):
            faith = f"{faith:.3f}"
        if isinstance(ans_rel, float):
            ans_rel = f"{ans_rel:.3f}"
        if isinstance(ttft, float):
            ttft = f"{ttft:.2f}s"

        reranker = c.get("reranker_model") or "none"
        lines.append(
            f"| {i+1} | {c['chunking_strategy']} | {c['chunk_size']} | "
            f"{c['chunk_overlap']} | {c['retrieval_top_k']} | "
            f"{c['retrieval_strategy']} | {c['retrieval_use_hyde']} | "
            f"{c['prompt_template']} | {reranker} | {faith} | {ans_rel} | {ttft} |"
        )

    # Add insights
    insights = state.get("insights", [])
    if insights:
        lines.append("\nAccumulated insights:")
        for ins in insights[-5:]:
            lines.append(f"  - {ins}")

    return "\n".join(lines)


def _merge_proposal(args: dict, state: AgentState) -> ExplorationConfig:
    """Merge LLM proposal with fixed model settings from state."""
    return ExplorationConfig(
        llm_model=state.get("_default_llm", "gemma3:4b"),
        embedding_model=state.get("_default_emb", "nomic-embed-text:latest"),
        chunking_strategy=args.get("chunking_strategy", "recursive"),
        chunk_size=int(args.get("chunk_size", 500)),
        chunk_overlap=int(args.get("chunk_overlap", 100)),
        retrieval_top_k=int(args.get("retrieval_top_k", 5)),
        retrieval_strategy=args.get("retrieval_strategy", "similarity"),
        retrieval_use_hyde=bool(args.get("retrieval_use_hyde", False)),
        prompt_template=args.get("prompt_template", "concise"),
        reranker_model=args.get("reranker_model"),
    )


def _validate_config(config: ExplorationConfig) -> ExplorationConfig:
    """Clamp config values to valid ranges and fix constraint violations."""
    cs = config["chunk_size"]
    co = config["chunk_overlap"]
    if co >= cs:
        co = max(50, cs // 5)

    valid_strategies = VALID_CONFIG_RANGES["chunking_strategy"]
    if config["chunking_strategy"] not in valid_strategies:
        config = {**config, "chunking_strategy": "recursive"}

    valid_templates = VALID_CONFIG_RANGES["prompt_template"]
    if config["prompt_template"] not in valid_templates:
        config = {**config, "prompt_template": "concise"}

    if config["retrieval_strategy"] not in ("similarity", "mmr"):
        config = {**config, "retrieval_strategy": "similarity"}

    return {**config, "chunk_overlap": co}


def _heuristic_next_config(state: AgentState) -> ExplorationConfig:
    """Deterministic fallback: pick the next unexplored config variation."""
    completed = state.get("completed_runs", [])
    tested_keys = set()
    for run in completed:
        c = run["config"]
        key = (
            c["chunking_strategy"], c["chunk_size"], c["chunk_overlap"],
            c["retrieval_top_k"], c["retrieval_strategy"],
            c["retrieval_use_hyde"], c["prompt_template"],
            c.get("reranker_model"),
        )
        tested_keys.add(key)

    llm_model = state.get("_default_llm", "gemma3:4b")
    emb_model = state.get("_default_emb", "nomic-embed-text:latest")

    # Build candidate configs with strategic parameter choices
    candidates: list[ExplorationConfig] = []

    # If we have a best run, vary from it
    best = _find_best(completed)
    if best:
        bc = best["config"]
        # Vary chunk_size
        for cs in [300, 800, 1000]:
            if cs != bc["chunk_size"]:
                candidates.append({**bc, "llm_model": llm_model, "embedding_model": emb_model, "chunk_size": cs})
        # Vary chunking strategy
        for strat in VALID_CONFIG_RANGES["chunking_strategy"]:
            if strat != bc["chunking_strategy"]:
                candidates.append({**bc, "llm_model": llm_model, "embedding_model": emb_model, "chunking_strategy": strat})
        # Try HyDE
        if not bc["retrieval_use_hyde"]:
            candidates.append({**bc, "llm_model": llm_model, "embedding_model": emb_model, "retrieval_use_hyde": True})
        # Try different top_k
        for k in [3, 8, 10]:
            if k != bc["retrieval_top_k"]:
                candidates.append({**bc, "llm_model": llm_model, "embedding_model": emb_model, "retrieval_top_k": k})
        # Try different templates
        for tmpl in VALID_CONFIG_RANGES["prompt_template"]:
            if tmpl != bc["prompt_template"]:
                candidates.append({**bc, "llm_model": llm_model, "embedding_model": emb_model, "prompt_template": tmpl})
    else:
        # No best yet — generate grid defaults
        for cs in [300, 800, 1000]:
            for strat in ["recursive", "semantic"]:
                candidates.append(ExplorationConfig(
                    llm_model=llm_model, embedding_model=emb_model,
                    chunk_size=cs, chunk_overlap=cs // 5,
                    chunking_strategy=strat,
                    retrieval_top_k=5, retrieval_strategy="similarity",
                    retrieval_use_hyde=False, prompt_template="concise",
                    reranker_model=None,
                ))

    # Shuffle and pick first untested
    random.shuffle(candidates)
    for c in candidates:
        c = _validate_config(c)
        key = (
            c["chunking_strategy"], c["chunk_size"], c["chunk_overlap"],
            c["retrieval_top_k"], c["retrieval_strategy"],
            c["retrieval_use_hyde"], c["prompt_template"],
            c.get("reranker_model"),
        )
        if key not in tested_keys:
            return c

    # Exhausted all candidates — random valid config
    cs = random.choice(VALID_CONFIG_RANGES["chunk_size"])
    co = random.choice([x for x in VALID_CONFIG_RANGES["chunk_overlap"] if x < cs])
    return ExplorationConfig(
        llm_model=llm_model, embedding_model=emb_model,
        chunk_size=cs, chunk_overlap=co,
        chunking_strategy=random.choice(VALID_CONFIG_RANGES["chunking_strategy"]),
        retrieval_top_k=random.choice(VALID_CONFIG_RANGES["retrieval_top_k"]),
        retrieval_strategy=random.choice(VALID_CONFIG_RANGES["retrieval_strategy"]),
        retrieval_use_hyde=random.choice(VALID_CONFIG_RANGES["retrieval_use_hyde"]),
        prompt_template=random.choice(VALID_CONFIG_RANGES["prompt_template"]),
        reranker_model=random.choice(VALID_CONFIG_RANGES["reranker_model"]),
    )


def _find_best(completed: list) -> dict | None:
    """Find the run with the best faithfulness score."""
    best_run = None
    best_score = -1.0
    for run in completed:
        score = run["metrics"].get("ragas_faithfulness", 0.0) or 0.0
        if score > best_score:
            best_score = score
            best_run = run
    return best_run
