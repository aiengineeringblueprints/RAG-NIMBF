import json
import logging
import time
from datetime import datetime
from pathlib import Path

from rich.console import Console

from config import BenchmarkConfig, get_all_combinations
from benchmark.dataset import load_benchmark_data
from benchmark.chunking import get_chunker, chunk_documents
from benchmark.retrieval import (
    build_vector_store,
    clear_cache,
    retrieve,
    _cache_key,
)
from benchmark.generation import get_llm, generate_answer, GenerationResult
from benchmark.prompt_templates import get_template
from benchmark.evaluation import evaluate_results
from benchmark.reranker import get_reranker
from benchmark.reporting import generate_report
from benchmark.reporting.exports import _result_to_dict
from benchmark.tracking import setup_mlflow, log_benchmark_run
from benchmark.reporting.models import (
    BenchmarkResultExtended,
    PerSampleResult,
    compute_stats,
)

console = Console()
logger = logging.getLogger(__name__)


def run_single_benchmark(
    config: BenchmarkConfig, data: list[dict], run_dir: Path | None = None,
) -> BenchmarkResultExtended:
    run_start = time.perf_counter()
    console.print(f"\n[bold yellow]>>> Starting: {config.name}[/bold yellow]")
    collection_name = config.name.replace(":", "_").replace("/", "_")

    # Prepare QA log path if run_dir is provided
    qa_log: list[dict] = []
    qa_log_path: Path | None = None
    if run_dir is not None:
        configs_dir = run_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config.name.replace(":", "_").replace("/", "_")
        qa_log_path = configs_dir / f"{safe_name}_qa.json"

    # 1. Chunk
    chunker = get_chunker(config.chunking_strategy, config.chunk_size, config.chunk_overlap)
    chunks = chunk_documents(chunker, data)
    console.print(f"  [dim]Chunked into {len(chunks)} pieces[/dim]")

    # 2. Embed + build vector store (with caching to avoid redundant embedding)
    cache_k = _cache_key(
        config.embedding_model,
        config.chunk_size,
        config.chunk_overlap,
        config.chunking_strategy,
    )
    vector_store = build_vector_store(
        chunks,
        config.embedding_model,
        collection_name,
        ollama_base_url=config.embedding_base_url(),
        ollama_api_key=config.embedding_api_key(),
        cache_key=cache_k,
        embedding_provider=config.embedding_provider,
    )
    console.print(f"  [dim]Vector store built[/dim]")

    # 3. Generate answers
    llm = get_llm(
        provider=config.llm_provider,
        model_name=config.llm_model,
        base_url=config.llm_base_url(),
        api_key=config.llm_api_key(),
        max_new_tokens=config.max_new_tokens,
    )
    reranker = get_reranker(config.reranker_model)
    prompt_tmpl = get_template(config.prompt_template)
    questions: list[str] = []
    ground_truths: list[str] = []
    all_contexts: list[list[str]] = []
    gen_results: list[GenerationResult] = []

    for i, sample in enumerate(data):
        console.print(f"  [cyan]({i + 1}/{len(data)})[/cyan] {sample['question'][:80]}{'...' if len(sample['question']) > 80 else ''}")

        retrieved_docs = retrieve(vector_store, sample["question"], config.retrieval_top_k)

        if reranker is not None:
            retrieved_docs = reranker.rerank(
                sample["question"], retrieved_docs, config.reranker_top_k,
            )

        context_texts = [doc.page_content for doc in retrieved_docs]

        result = generate_answer(
            llm, sample["question"], context_texts,
            system_prompt=prompt_tmpl.system_prompt,
            human_template=prompt_tmpl.human_template,
        )

        questions.append(sample["question"])
        ground_truths.append(sample["ground_truth"])
        all_contexts.append(context_texts)
        gen_results.append(result)

        # Stream QA pair to log file after each answer
        if qa_log_path is not None:
            qa_log.append({
                "index": i,
                "question": sample["question"],
                "answer": result.answer,
            })
            qa_log_path.write_text(json.dumps(qa_log, indent=2, ensure_ascii=False))

    console.print(f"  [dim]Generated {len(gen_results)} answers[/dim]       ")

    # 4. Evaluate with RAGAS using a separate critic model
    console.print(f"  [dim]Running RAGAS evaluation (critic: {config.eval_critic_llm})...[/dim]")
    eval_result = evaluate_results(
        questions, ground_truths,
        [r.answer for r in gen_results],
        all_contexts,
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        critic_llm_model=config.eval_critic_llm,
        critic_embedding_model=config.eval_critic_embedding,
        ollama_base_url=config.ollama_base_url,
        ollama_api_key=config.ollama_api_key,
        openai_compat_base_url=config.openai_compat_base_url,
        openai_compat_api_key=config.openai_compat_api_key,
        critic_ollama_base_url=config.eval_critic_ollama_base_url,
        critic_ollama_api_key=config.eval_critic_ollama_api_key,
        critic_openai_compat_base_url=config.eval_critic_openai_compat_base_url,
        critic_openai_compat_api_key=config.eval_critic_openai_compat_api_key,
        critic_max_tokens=config.eval_critic_max_tokens,
    )

    if eval_result.error:
        console.print(f"  [red]RAGAS evaluation failed: {eval_result.error}[/red]")
    else:
        console.print(f"  [dim]RAGAS evaluation complete[/dim]")

    ragas_means = eval_result.metric_means
    per_sample_ragas = eval_result.per_sample_scores

    total_time = time.perf_counter() - run_start
    console.print(f"[bold green]<<< Finished: {config.name} in {total_time:.1f}s[/bold green]")

    # 5. Build per-sample results
    per_sample = tuple(
        PerSampleResult(
            question=q,
            ground_truth=gt,
            answer=gr.answer,
            contexts=tuple(ctx),
            ttft_seconds=gr.ttft_seconds,
            total_seconds=gr.total_seconds,
            token_count=gr.token_count,
            tokens_per_second=gr.tokens_per_second,
            gpu_usage=gr.gpu_usage,
            ragas_scores=per_sample_ragas[i] if i < len(per_sample_ragas) else {},
        )
        for i, (q, gt, ctx, gr) in enumerate(
            zip(questions, ground_truths, all_contexts, gen_results)
        )
    )

    # 6. Aggregate metrics
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

    def _ragas_stat_samples(key: str) -> list[float]:
        vals = []
        for s in per_sample:
            v = s.ragas_scores.get(key)
            if v is not None:
                vals.append(v)
        return vals

    return BenchmarkResultExtended(
        config_name=config.name,
        llm_model=config.llm_model,
        embedding_model=config.embedding_model,
        prompt_template=config.prompt_template,
        chunking_strategy=config.chunking_strategy,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
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
        ragas_faithfulness_stats=compute_stats(_ragas_stat_samples("faithfulness")),
        ragas_answer_relevancy_stats=compute_stats(_ragas_stat_samples("answer_relevancy")),
        ragas_answer_correctness_stats=compute_stats(_ragas_stat_samples("answer_correctness")),
        ragas_context_precision_stats=compute_stats(_ragas_stat_samples("context_precision")),
        ragas_context_recall_stats=compute_stats(_ragas_stat_samples("context_recall")),
        evaluation_error=eval_result.error,
        ragas_valid_sample_counts=eval_result.samples_with_valid_scores or None,
        reranker_model=config.reranker_model,
        reranker_top_k=config.reranker_top_k if config.reranker_model else None,
    )


def _next_run_dir(base: Path = Path("results")) -> Path:
    """Return the next available runN subdirectory (e.g. results/run3)."""
    base.mkdir(parents=True, exist_ok=True)
    max_run = 0
    for child in base.iterdir():
        if child.is_dir() and child.name.startswith("run"):
            try:
                n = int(child.name[3:])
                max_run = max(max_run, n)
            except ValueError:
                pass
    run_dir = base / f"run{max_run + 1}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _save_config_result(result: BenchmarkResultExtended, run_dir: Path) -> None:
    """Persist a single config result to disk immediately."""
    config_dir = run_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.config_name.replace(":", "_").replace("/", "_")
    path = config_dir / f"{safe_name}.json"
    path.write_text(json.dumps(_result_to_dict(result), indent=2, default=str))
    console.print(f"[dim]  Saved config result to {path}[/dim]")


def run_all_benchmarks() -> list[BenchmarkResultExtended]:
    configs = get_all_combinations()
    console.print(f"[bold]Running {len(configs)} benchmark configuration(s)[/bold]")

    data = load_benchmark_data(
        dataset_name=configs[0].dataset_name,
        subset=configs[0].dataset_subset or None,
        sample_size=configs[0].dataset_sample_size,
    )

    # Create run directory up front so results are saved even if a
    # later step (MLflow, cleanup, etc.) crashes.
    run_dir = _next_run_dir()
    wall_start = time.perf_counter()

    # Run sequentially: local models typically share a single GPU and process
    # requests serially. Parallel execution causes GPU memory thrashing,
    # request queuing, and timeouts that produce *lower* throughput.
    results: list[BenchmarkResultExtended] = []
    for i, config in enumerate(configs):
        console.print(f"\n[bold cyan]Config {i + 1}/{len(configs)}[/bold cyan]")
        result = run_single_benchmark(config, data, run_dir=run_dir)

        # Save to disk immediately (survives MLflow crashes)
        _save_config_result(result, run_dir)

        log_benchmark_run(result)
        results.append(result)

    # Clear embedding cache and all ChromaDB collections after all runs complete
    console.print("\n[bold]Cleaning up vector store collections...[/bold]")
    clear_cache()
    console.print("[green]All collections deleted.[/green]")

    wall_time = time.perf_counter() - wall_start
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    console.print(f"[dim]Saving aggregated results to {run_dir}[/dim]")

    generate_report(
        results,
        results_dir=run_dir,
        timestamp=timestamp,
        dataset_name=configs[0].dataset_name,
        dataset_subset=configs[0].dataset_subset,
        dataset_sample_size=configs[0].dataset_sample_size,
        total_time=wall_time,
    )

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    console.print("[bold]RAG Benchmarking Framework[/bold]")
    console.print("=" * 50)
    tracking_uri = setup_mlflow()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    run_all_benchmarks()


if __name__ == "__main__":
    main()
