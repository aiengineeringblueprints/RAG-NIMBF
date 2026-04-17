"""LangGraph state schema for the autonomous benchmarking agent."""

from __future__ import annotations

from typing import Any, TypedDict


class ExplorationConfig(TypedDict):
    """A single RAG configuration to explore."""

    llm_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str  # "recursive" | "character" | "token" | "semantic"
    retrieval_top_k: int
    retrieval_strategy: str  # "similarity" | "mmr"
    retrieval_use_hyde: bool
    prompt_template: str  # "concise" | "detailed" | "finqa"
    reranker_model: str | None


class ExplorationResult(TypedDict):
    """Aggregate result of running one configuration."""

    config: ExplorationConfig
    metrics: dict[str, float]
    total_time_seconds: float
    num_chunks: int
    error: str | None


class AgentState(TypedDict, total=False):
    """Full state carried through the LangGraph."""

    # Fixed parameters (set once at start)
    dataset_name: str
    dataset_subset: str | None
    dataset_sample_size: int
    ollama_base_url: str
    agent_model: str
    max_iterations: int

    # Accumulated history
    completed_runs: list[ExplorationResult]
    insights: list[str]

    # Current iteration
    iteration: int
    current_config: ExplorationConfig | None
    current_result: ExplorationResult | None
    analysis_text: str

    # Control flow
    phase: str  # "propose" | "run" | "analyze" | "done"
    should_continue: bool

    # Cached data (not serialized, kept in memory)
    _data: list[dict[str, Any]]


# Valid parameter ranges for the config proposer
VALID_CONFIG_RANGES: dict[str, Any] = {
    "chunking_strategy": ["recursive", "character", "token", "semantic"],
    "chunk_size": [200, 300, 500, 800, 1000, 1500],
    "chunk_overlap": [50, 100, 150, 200],
    "retrieval_top_k": [3, 5, 8, 10, 15],
    "retrieval_strategy": ["similarity", "mmr"],
    "retrieval_use_hyde": [True, False],
    "prompt_template": ["concise", "detailed", "finqa"],
    "reranker_model": [None, "huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2"],
}


def make_seed_config(
    llm_model: str = "gemma3:4b",
    embedding_model: str = "nomic-embed-text:latest",
) -> ExplorationConfig:
    """Return a sensible default starting configuration."""
    return ExplorationConfig(
        llm_model=llm_model,
        embedding_model=embedding_model,
        chunk_size=500,
        chunk_overlap=100,
        chunking_strategy="recursive",
        retrieval_top_k=5,
        retrieval_strategy="similarity",
        retrieval_use_hyde=False,
        prompt_template="concise",
        reranker_model=None,
    )
