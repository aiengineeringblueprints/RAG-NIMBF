"""Result analyzer node: LLM analyzes results and extracts insights."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich.console import Console

from agentic.prompts import RESULT_ANALYZER_SYSTEM_PROMPT
from agentic.state import AgentState

console = Console()
logger = logging.getLogger(__name__)


def result_analyzer_node(state: AgentState) -> dict[str, Any]:
    """LangGraph node: analyze results and extract insights."""
    iteration = state.get("iteration", 0)

    console.print(f"\n[bold cyan]Iteration {iteration + 1}: Analyzing results...[/bold cyan]")

    llm = ChatOllama(
        model=state.get("agent_model", "qwen3:8b"),
        base_url=state.get("ollama_base_url", "http://localhost:11434"),
        temperature=0.1,
    )

    messages = [
        SystemMessage(content=RESULT_ANALYZER_SYSTEM_PROMPT),
        HumanMessage(content=_build_analyzer_input(state)),
    ]

    analysis_text = ""
    try:
        response = llm.invoke(messages)
        analysis_text = response.content or ""
        console.print(f"  [dim]Analysis complete[/dim]")
    except Exception as exc:
        logger.warning("Analysis LLM call failed: %s", exc)
        analysis_text = f"Analysis unavailable: {exc}"

    # Extract structured insights
    new_insights = _parse_insights(analysis_text)

    # Also add a basic metric comparison insight
    completed = state.get("completed_runs", [])
    if len(completed) >= 2:
        prev = completed[-2]
        curr = completed[-1]
        prev_f = prev["metrics"].get("ragas_faithfulness", 0) or 0
        curr_f = curr["metrics"].get("ragas_faithfulness", 0) or 0
        if curr_f > prev_f:
            new_insights.append(
                f"Faithfulness improved: {prev_f:.3f} -> {curr_f:.3f} "
                f"(delta: +{curr_f - prev_f:.3f})"
            )
        elif curr_f < prev_f:
            new_insights.append(
                f"Faithfulness regressed: {prev_f:.3f} -> {curr_f:.3f} "
                f"(delta: {curr_f - prev_f:.3f})"
            )

    all_insights = list(state.get("insights", [])) + new_insights

    iteration += 1
    max_iter = state.get("max_iterations", 8)
    should_continue = iteration < max_iter

    if not should_continue:
        console.print(f"\n[bold]Reached max iterations ({max_iter}). Stopping.[/bold]")
    elif _detect_convergence(completed):
        console.print("\n[bold]Results have converged. Stopping early.[/bold]")
        should_continue = False

    return {
        "analysis_text": analysis_text,
        "insights": all_insights,
        "iteration": iteration,
        "phase": "done" if not should_continue else "propose",
        "should_continue": should_continue,
    }


def _build_analyzer_input(state: AgentState) -> str:
    """Format results for the analyzer LLM."""
    lines: list[str] = []
    completed = state.get("completed_runs", [])

    if not completed:
        return "No results to analyze."

    current = completed[-1]
    lines.append("Latest result:\n")
    _format_run(lines, current)

    if len(completed) > 1:
        lines.append(f"\nAll {len(completed)} results summary:\n")
        for i, run in enumerate(completed):
            c = run["config"]
            m = run["metrics"]
            faith = m.get("ragas_faithfulness", "N/A")
            if isinstance(faith, float):
                faith = f"{faith:.3f}"
            lines.append(
                f"  Run {i+1}: strategy={c.get('chunking_strategy')} "
                f"chunk_size={c.get('chunk_size')} overlap={c.get('chunk_overlap')} "
                f"top_k={c.get('retrieval_top_k')} hyde={c.get('retrieval_use_hyde')} "
                f"template={c.get('prompt_template')} -> faithfulness={faith} "
                f"time={run['total_time_seconds']:.1f}s"
            )

    # Find best so far
    best = _find_best(completed)
    if best:
        m = best["metrics"]
        lines.append(f"\nBest so far: faithfulness={m.get('ragas_faithfulness', 'N/A')}")

    return "\n".join(lines)


def _format_run(lines: list[str], run: dict) -> None:
    """Append a formatted run to lines."""
    c = run["config"]
    m = run["metrics"]

    lines.append(f"Config: strategy={c.get('chunking_strategy')}, "
                  f"chunk_size={c.get('chunk_size')}, overlap={c.get('chunk_overlap')}, "
                  f"top_k={c.get('retrieval_top_k')}, strategy={c.get('retrieval_strategy')}, "
                  f"hyde={c.get('retrieval_use_hyde')}, template={c.get('prompt_template')}")
    lines.append(f"Time: {run['total_time_seconds']:.1f}s, Chunks: {run['num_chunks']}")
    if run.get("error"):
        lines.append(f"Error: {run['error']}")

    lines.append("Metrics:")
    for key, val in sorted(m.items()):
        if isinstance(val, float):
            lines.append(f"  {key}: {val:.4f}")
        else:
            lines.append(f"  {key}: {val}")


def _parse_insights(text: str) -> list[str]:
    """Extract INSIGHT: and RECOMMENDATION: lines from LLM response."""
    insights: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        match = re.match(r"^(?:INSIGHT|RECOMMENDATION):\s*(.+)$", line, re.IGNORECASE)
        if match:
            insights.append(match.group(1).strip())
    return insights


def _detect_convergence(completed: list, window: int = 3, threshold: float = 0.02) -> bool:
    """Check if the last N runs have converged (scores within threshold)."""
    if len(completed) < window:
        return False

    recent = completed[-window:]
    scores = [r["metrics"].get("ragas_faithfulness", 0) or 0 for r in recent]
    if not scores:
        return False

    return max(scores) - min(scores) < threshold


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
