"""Tool definitions and pipeline execution for the autonomous agent.

Tools exposed to the agent LLM are decorated with @tool.
The main pipeline function (run_full_pipeline) is called programmatically
by the benchmark_runner node, not by the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

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
# Internal: full pipeline execution (mirrors main.py:run_single_benchmark)
# ---------------------------------------------------------------------------

def run_full_pipeline(
    config: ExplorationConfig,
    state: AgentState,
) -> tuple[BenchmarkResultExtended | None, ExplorationResult]:
    """Execute the complete benchmark pipeline for one configuration.

    Returns (full_result, compact_result). full_result may be None on error.
    """
    from config import BenchmarkConfig
    from benchmark.dataset import load_benchmark_data
    from benchmark.chunking import get_chunker, chunk_documents
    from benchmark.retrieval import (
        build_vector_store, retrieve, expand_query_with_hyde, _cache_key,
    )
    from benchmark.generation import get_llm, generate_answer, GenerationResult
    from benchmark.prompt_templates import get_template
    from benchmark.evaluation import evaluate_results
    from benchmark.custom_metrics import compute_custom_metrics
    from benchmark.reranker import get_reranker
    from benchmark.reporting.models import (
        BenchmarkResultExtended, PerSampleResult, compute_stats,
    )
    from benchmark.reporting.exports import _result_to_dict
    from benchmark.tracking import log_benchmark_run

    run_start = time.perf_counter()

    # Load env defaults for fields the agent doesn't control
    llm_provider, llm_model_name = _parse_model_id(config["llm_model"])
    emb_provider, emb_model_name = _parse_model_id(config["embedding_model"])

    ollama_base_url = state.get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_api_key = os.getenv("OLLAMA_API_KEY") or None
    openai_compat_base_url = os.getenv("OPENAI_COMPAT_BASE_URL") or None
    openai_compat_api_key = os.getenv("OPENAI_COMPAT_API_KEY") or None

    # Build a BenchmarkConfig from the exploration config
    bench_config = BenchmarkConfig(
        llm_model=llm_model_name,
        llm_provider=llm_provider,
        embedding_model=config["embedding_model"],
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
        chunking_strategy=config["chunking_strategy"],
        retrieval_top_k=config["retrieval_top_k"],
        retrieval_strategy=config["retrieval_strategy"],
        retrieval_use_hyde=config["retrieval_use_hyde"],
        prompt_template=config["prompt_template"],
        reranker_model=config.get("reranker_model"),
        max_new_tokens=int(os.getenv("MAX_NEW_TOKENS", "256")),
        ollama_base_url=ollama_base_url,
        ollama_api_key=ollama_api_key,
        openai_compat_base_url=openai_compat_base_url,
        openai_compat_api_key=openai_compat_api_key,
        llm_ollama_base_url=os.getenv("LLM_OLLAMA_BASE_URL") or None,
        llm_ollama_api_key=os.getenv("LLM_OLLAMA_API_KEY") or None,
        llm_openai_compat_base_url=os.getenv("LLM_OPENAI_COMPAT_BASE_URL") or None,
        llm_openai_compat_api_key=os.getenv("LLM_OPENAI_COMPAT_API_KEY") or None,
        eval_critic_ollama_base_url=os.getenv("EVAL_CRITIC_OLLAMA_BASE_URL") or None,
        eval_critic_ollama_api_key=os.getenv("EVAL_CRITIC_OLLAMA_API_KEY") or None,
        eval_critic_openai_compat_base_url=os.getenv("EVAL_CRITIC_OPENAI_COMPAT_BASE_URL") or None,
        eval_critic_openai_compat_api_key=os.getenv("EVAL_CRITIC_OPENAI_COMPAT_API_KEY") or None,
        embedding_ollama_base_url=os.getenv("EMBEDDING_OLLAMA_BASE_URL") or None,
        embedding_ollama_api_key=os.getenv("EMBEDDING_OLLAMA_API_KEY") or None,
        eval_critic_max_tokens=int(os.getenv("EVAL_CRITIC_MAX_TOKENS", "4096")),
        dataset_name=state.get("dataset_name", os.getenv("DATASET_NAME", "t2-ragbench")),
        dataset_subset=state.get("dataset_subset", os.getenv("DATASET_SUBSET", "FinQA")),
        dataset_sample_size=state.get("dataset_sample_size", int(os.getenv("DATASET_SAMPLE_SIZE", "50"))),
        eval_critic_llm=os.getenv("EVAL_CRITIC_LLM", "gemma3:12b"),
        eval_critic_embedding=os.getenv(
            "EVAL_CRITIC_EMBEDDING",
            os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest").split(",")[0].strip(),
        ),
        reranker_top_k=int(os.getenv("RERANKER_TOP_K", "3")),
        semantic_breakpoint_type=os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile"),
        semantic_breakpoint_amount=int(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95")),
    )

    config_label = bench_config.name
    collection_name = config_label.replace(":", "_").replace("/", "_")
    console.print(f"\n[bold yellow]>>> Agent running: {config_label}[/bold yellow]")

    # Load data (cached in state)
    data = state.get("_data")
    if data is None:
        data = load_benchmark_data(
            dataset_name=bench_config.dataset_name,
            subset=bench_config.dataset_subset or None,
            sample_size=bench_config.dataset_sample_size,
        )

    try:
        # 1. Chunk
        chunker_kwargs: dict[str, Any] = {}
        semantic_emb = None
        if bench_config.chunking_strategy == "semantic":
            from benchmark.embedding import get_embedding_model

            semantic_emb = get_embedding_model(
                bench_config.embedding_model,
                bench_config.embedding_base_url(),
                bench_config.embedding_api_key(),
                provider=bench_config.embedding_provider,
            )
            chunker_kwargs = dict(
                embeddings=semantic_emb,
                breakpoint_threshold_type=bench_config.semantic_breakpoint_type,
                breakpoint_threshold_amount=bench_config.semantic_breakpoint_amount,
            )
        chunker = get_chunker(
            bench_config.chunking_strategy,
            bench_config.chunk_size,
            bench_config.chunk_overlap,
            **chunker_kwargs,
        )
        chunks = chunk_documents(chunker, data)
        console.print(f"  [dim]Chunked into {len(chunks)} pieces[/dim]")

        # 2. Vector store
        cache_k = _cache_key(
            bench_config.embedding_model,
            bench_config.chunk_size,
            bench_config.chunk_overlap,
            bench_config.chunking_strategy,
        )
        vector_store = build_vector_store(
            chunks, bench_config.embedding_model, collection_name,
            ollama_base_url=bench_config.embedding_base_url(),
            ollama_api_key=bench_config.embedding_api_key(),
            cache_key=cache_k,
            embedding_provider=bench_config.embedding_provider,
        )

        # 3. Generate answers
        llm = get_llm(
            provider=bench_config.llm_provider,
            model_name=bench_config.llm_model,
            base_url=bench_config.llm_base_url(),
            api_key=bench_config.llm_api_key(),
            max_new_tokens=bench_config.max_new_tokens,
        )
        reranker = get_reranker(bench_config.reranker_model)
        prompt_tmpl = get_template(bench_config.prompt_template)

        questions: list[str] = []
        ground_truths: list[str] = []
        all_contexts: list[list[str]] = []
        gen_results: list[GenerationResult] = []

        for i, sample in enumerate(data):
            query = sample["question"]
            if bench_config.retrieval_use_hyde:
                query = expand_query_with_hyde(llm, sample["question"])

            retrieved_docs = retrieve(
                vector_store, query, bench_config.retrieval_top_k,
                retrieval_strategy=bench_config.retrieval_strategy,
                fetch_k=bench_config.retrieval_fetch_k,
                mmr_lambda=bench_config.retrieval_mmr_lambda,
            )

            if reranker is not None:
                retrieved_docs = reranker.rerank(
                    sample["question"], retrieved_docs, bench_config.reranker_top_k,
                )

            context_texts = [doc.page_content for doc in retrieved_docs]

            result = generate_answer(
                llm, sample["question"], context_texts,
                system_prompt=prompt_tmpl.system_prompt,
                human_template=prompt_tmpl.human_template,
                strip_mode=bench_config.llm_answer_strip_mode,
                value_fallback=bench_config.llm_answer_value_fallback,
                ground_truth=sample["ground_truth"],
                prompt_template_name=bench_config.prompt_template,
            )

            questions.append(sample["question"])
            ground_truths.append(sample["ground_truth"])
            all_contexts.append(context_texts)
            gen_results.append(result)

        console.print(f"  [dim]Generated {len(gen_results)} answers[/dim]")

        # 4. Evaluate with RAGAS
        console.print(f"  [dim]Running RAGAS evaluation...[/dim]")
        eval_result = evaluate_results(
            questions, ground_truths,
            [r.answer for r in gen_results],
            all_contexts,
            llm_model=bench_config.llm_model,
            embedding_model=bench_config.embedding_model,
            critic_llm_model=bench_config.eval_critic_llm,
            critic_embedding_model=bench_config.eval_critic_embedding,
            ollama_base_url=bench_config.ollama_base_url,
            ollama_api_key=bench_config.ollama_api_key,
            openai_compat_base_url=bench_config.openai_compat_base_url,
            openai_compat_api_key=bench_config.openai_compat_api_key,
            critic_ollama_base_url=bench_config.eval_critic_ollama_base_url,
            critic_ollama_api_key=bench_config.eval_critic_ollama_api_key,
            critic_openai_compat_base_url=bench_config.eval_critic_openai_compat_base_url,
            critic_openai_compat_api_key=bench_config.eval_critic_openai_compat_api_key,
            critic_max_tokens=bench_config.eval_critic_max_tokens,
        )

        if eval_result.error:
            console.print(f"  [red]RAGAS failed: {eval_result.error}[/red]")
        else:
            console.print(f"  [dim]RAGAS complete[/dim]")

        ragas_means = eval_result.metric_means
        per_sample_ragas = eval_result.per_sample_scores

        # 4b. Custom metrics
        console.print("  [dim]Computing custom metrics...[/dim]")
        from benchmark.embedding import get_embedding_model

        _emb_model = semantic_emb or get_embedding_model(
            bench_config.embedding_model,
            bench_config.embedding_base_url(),
            bench_config.embedding_api_key(),
            provider=bench_config.embedding_provider,
        )
        _embed_fn = _emb_model.embed_query

        from sentence_transformers import SentenceTransformer
        _bert_model = SentenceTransformer("roberta-large")

        custom_result = compute_custom_metrics(
            questions, ground_truths,
            [r.answer for r in gen_results],
            all_contexts,
            embed_fn=_embed_fn,
            bert_model=_bert_model,
        )

        total_time = time.perf_counter() - run_start
        console.print(f"[bold green]<<< Done: {config_label} in {total_time:.1f}s[/bold green]")

        # 5. Build per-sample results
        per_sample_custom = custom_result.per_sample
        per_sample = tuple(
            PerSampleResult(
                question=q, ground_truth=gt, answer=gr.answer,
                contexts=tuple(ctx), ttft_seconds=gr.ttft_seconds,
                total_seconds=gr.total_seconds, token_count=gr.token_count,
                tokens_per_second=gr.tokens_per_second, gpu_usage=gr.gpu_usage,
                ragas_scores=per_sample_ragas[i] if i < len(per_sample_ragas) else {},
                custom_scores=per_sample_custom[i] if i < len(per_sample_custom) else {},
                answer_valid=gr.answer_valid,
            )
            for i, (q, gt, ctx, gr) in enumerate(
                zip(questions, ground_truths, all_contexts, gen_results)
            )
        )

        ttfts = [s.ttft_seconds for s in per_sample]
        tps_list = [s.tokens_per_second for s in per_sample]
        gpu_utils = [
            s.gpu_usage["gpu_utilization_pct"]
            for s in per_sample
            if s.gpu_usage and "gpu_utilization_pct" in s.gpu_usage
        ]
        gpu_mems = [
            s.gpu_usage["memory_used_mb"]
            for s in per_sample
            if s.gpu_usage and "memory_used_mb" in s.gpu_usage
        ]

        def _ragas_vals(key: str) -> list[float]:
            return [s.ragas_scores.get(key) for s in per_sample if s.ragas_scores.get(key) is not None]

        def _custom_vals(key: str) -> list[float]:
            return [
                s.custom_scores.get(key)
                for s in per_sample
                if s.custom_scores and s.custom_scores.get(key) is not None
            ]

        custom_means = custom_result.metric_means
        custom_stat_summaries = {
            key: compute_stats(_custom_vals(key)) for key in custom_means
        }

        full_result = BenchmarkResultExtended(
            config_name=config_label,
            llm_model=bench_config.llm_model,
            embedding_model=bench_config.embedding_model,
            prompt_template=bench_config.prompt_template,
            chunking_strategy=bench_config.chunking_strategy,
            chunk_size=bench_config.chunk_size,
            chunk_overlap=bench_config.chunk_overlap,
            num_chunks=len(chunks),
            num_questions=len(data),
            avg_ttft_seconds=sum(ttfts) / len(ttfts) if ttfts else 0,
            avg_tokens_per_second=sum(tps_list) / len(tps_list) if tps_list else 0,
            avg_gpu_utilization_pct=sum(gpu_utils) / len(gpu_utils) if gpu_utils else None,
            avg_gpu_memory_used_mb=sum(gpu_mems) / len(gpu_mems) if gpu_mems else None,
            ragas_faithfulness=ragas_means.get("faithfulness"),
            ragas_answer_relevancy=ragas_means.get("answer_relevancy"),
            ragas_answer_correctness=ragas_means.get("answer_correctness"),
            ragas_context_precision=ragas_means.get("context_precision"),
            ragas_context_recall=ragas_means.get("context_recall"),
            total_time_seconds=total_time,
            per_sample=per_sample,
            ttft_stats=compute_stats(ttfts),
            tps_stats=compute_stats(tps_list),
            gpu_util_stats=compute_stats(gpu_utils),
            gpu_mem_stats=compute_stats(gpu_mems),
            ragas_faithfulness_stats=compute_stats(_ragas_vals("faithfulness")),
            ragas_answer_relevancy_stats=compute_stats(_ragas_vals("answer_relevancy")),
            ragas_answer_correctness_stats=compute_stats(_ragas_vals("answer_correctness")),
            ragas_context_precision_stats=compute_stats(_ragas_vals("context_precision")),
            ragas_context_recall_stats=compute_stats(_ragas_vals("context_recall")),
            evaluation_error=eval_result.error,
            ragas_valid_sample_counts=eval_result.samples_with_valid_scores or None,
            custom_metric_means=custom_means if custom_means else None,
            custom_stats=custom_stat_summaries if custom_stat_summaries else None,
            reranker_model=bench_config.reranker_model,
            reranker_top_k=bench_config.reranker_top_k if bench_config.reranker_model else None,
        )

        # Save to disk
        _save_result_to_disk(full_result, config_label)

        # Log to MLflow
        try:
            log_benchmark_run(full_result)
        except Exception as exc:
            logger.warning("MLflow logging failed: %s", exc)

        # Build compact result for agent state
        metrics: dict[str, float] = {}
        for key in ("faithfulness", "answer_relevancy", "answer_correctness",
                     "context_precision", "context_recall"):
            v = ragas_means.get(key)
            if v is not None:
                metrics[f"ragas_{key}"] = v
        if custom_means:
            metrics.update(custom_means)
        metrics["avg_ttft_seconds"] = full_result.avg_ttft_seconds
        metrics["avg_tokens_per_second"] = full_result.avg_tokens_per_second

        return full_result, ExplorationResult(
            config=config,
            metrics=metrics,
            total_time_seconds=total_time,
            num_chunks=len(chunks),
            error=eval_result.error,
        )

    except Exception as exc:
        logger.exception("Pipeline failed for %s", config_label)
        return None, ExplorationResult(
            config=config,
            metrics={},
            total_time_seconds=time.perf_counter() - run_start,
            num_chunks=0,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_model_id(model_string: str) -> tuple[str, str]:
    """Parse provider:model_name. Defaults to ollama."""
    from benchmark.providers import parse_model_id
    return parse_model_id(model_string)


def _save_result_to_disk(result: BenchmarkResultExtended, config_label: str) -> None:
    """Save a single config result to the agent results directory."""
    base = Path("results")
    # Find or create the latest agent run directory
    agent_dir = _ensure_agent_run_dir(base)

    safe_name = config_label.replace(":", "_").replace("/", "_")
    path = agent_dir / "configs" / f"{safe_name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_result_to_dict(result), indent=2, default=str))
    console.print(f"[dim]  Saved to {path}[/dim]")


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
