"""LangGraph StateGraph definition for the autonomous agent."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agentic.state import AgentState
from agentic.config_proposer import config_proposer_node
from agentic.benchmark_runner import benchmark_runner_node
from agentic.result_analyzer import result_analyzer_node


def should_continue(state: AgentState) -> Literal["propose", "end"]:
    """Router: determine next node after analysis."""
    if not state.get("should_continue", True) or state.get("phase") == "done":
        return "end"
    return "propose"


def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("propose", config_proposer_node)
    graph.add_node("run", benchmark_runner_node)
    graph.add_node("analyze", result_analyzer_node)

    # Linear edges
    graph.add_edge(START, "propose")
    graph.add_edge("propose", "run")
    graph.add_edge("run", "analyze")

    # Conditional loop
    graph.add_conditional_edges(
        "analyze",
        should_continue,
        {"propose": "propose", "end": END},
    )

    return graph.compile()
