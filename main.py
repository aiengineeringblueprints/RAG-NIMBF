import json
import logging
import time
import hashlib
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from datetime import datetime
from pathlib import Path

import mlflow
from rich.console import Console

from config import BenchmarkConfig, get_all_combinations
from benchmark.dataset import load_benchmark_data, load_corpus_and_questions
from benchmark.chunking import get_chunker, chunk_documents
from benchmark.retrieval import (
    build_vector_store,
    retrieve,
    expand_query_with_hyde,
    _cache_key,
)
from benchmark.generation import get_llm, generate_answer, GenerationResult
from benchmark.prompt_templates import get_template
from benchmark.evaluation import evaluate_results
from benchmark.custom_metrics import CustomMetricsResult, compute_custom_metrics
from benchmark.gold_retrieval_metrics import compute_gold_doc_retrieval_metrics
from benchmark.reranker import get_reranker
from benchmark.reporting import generate_report
from benchmark.reporting.exports import _result_to_dict
from benchmark.tracking import (
    setup_mlflow,
    log_benchmark_run,
    log_aggregate_artifacts_to_mlflow,
)
from benchmark.tracing import setup_tracing
from benchmark.adapters import get_rag_adapter
from benchmark.reproducibility import write_reproducibility_bundle
from benchmark.resource_monitor import (
    ResourceMonitor,
    enabled_from_env as resource_monitor_enabled,
    gpu_index_from_env as resource_monitor_gpu_index,
    interval_from_env as resource_monitor_interval,
)
from benchmark.reporting.models import (
    BenchmarkResultExtended,
    PerSampleResult,
    compute_stats,
)

console = Console()
logger = logging.getLogger(__name__)


@contextmanager
def _stage_timer(
    stage_timings: dict[str, float],
    name: str,
    resource_monitor: ResourceMonitor | None = None,
):
    started = time.perf_counter()
    if resource_monitor is not None:
        resource_monitor.stage_start(name)
    try:
        yield
    finally:
        stage_timings[name] = stage_timings.get(name, 0.0) + (
            time.perf_counter() - started
        )
        if resource_monitor is not None:
            resource_monitor.stage_end(name)


def _content_fingerprint(items: list[dict]) -> str:
    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("id", "")).encode())
        digest.update(b"\0")
        digest.update(str(item.get("question", "")).encode())
        digest.update(b"\0")
        digest.update(str(item.get("context", "")).encode())
        digest.update(b"\0")
        digest.update(str(item.get("ground_truth", "")).encode())
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=2)
def _get_bert_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device="cpu")


def _build_internal_retrieval_index(
    config: BenchmarkConfig,
    data: list[dict],
    corpus: list[dict] | None,
    stage_timings: dict[str, float],
    resource_monitor: ResourceMonitor | None,
) -> tuple[list, object]:
    with _stage_timer(stage_timings, "chunk", resource_monitor):
        chunker_kwargs: dict = {}
        if config.chunking_strategy == "semantic":
            from benchmark.embedding import get_embedding_model

            semantic_emb = get_embedding_model(
                config.embedding_model,
                config.embedding_base_url(),
                config.embedding_api_key(),
                provider=config.embedding_provider,
            )
            chunker_kwargs = dict(
                embeddings=semantic_emb,
                breakpoint_threshold_type=config.semantic_breakpoint_type,
                breakpoint_threshold_amount=config.semantic_breakpoint_amount,
            )
        chunker = get_chunker(
            config.chunking_strategy,
            config.chunk_size or 0,
            config.chunk_overlap or 0,
            **chunker_kwargs,
        )

        # Use shared corpus if available, otherwise chunk per-question data
        chunk_source = corpus if corpus else data
        chunks = chunk_documents(chunker, chunk_source)
    console.print(f"  [dim]Chunked into {len(chunks)} pieces[/dim]")

    corpus_fingerprint = _content_fingerprint(chunk_source)
    cache_k = _cache_key(
        config.embedding_model,
        config.chunk_size,
        config.chunk_overlap,
        config.chunking_strategy,
        dataset_name=config.dataset_name,
        embedding_provider=config.embedding_provider,
        dataset_subset=config.dataset_subset,
        dataset_sample_size=config.dataset_sample_size,
        corpus_fingerprint=corpus_fingerprint,
        vector_db_backend=config.vector_db_backend,
    )
    collection_name = f"rag_{config.vector_db_backend}_{cache_k[:24]}"
    with _stage_timer(stage_timings, "index", resource_monitor):
        vector_store = build_vector_store(
            chunks,
            config.embedding_model,
            collection_name,
            ollama_base_url=config.embedding_base_url(),
            ollama_api_key=config.embedding_api_key(),
            cache_key=cache_k,
            embedding_provider=config.embedding_provider,
            vector_db_backend=config.vector_db_backend,
            lancedb_path=config.lancedb_path,
            create_if_missing=config.benchmark_stage != "query",
        )
    console.print(f"  [dim]Vector store built[/dim]")
    return chunks, vector_store


def _generation_result_from_adapter(adapter_result) -> GenerationResult:
    return GenerationResult(
        answer=adapter_result.answer,
        ttft_seconds=adapter_result.ttft_seconds,
        total_seconds=adapter_result.total_seconds,
        token_count=adapter_result.token_count,
        tokens_per_second=adapter_result.tokens_per_second,
        gpu_usage=adapter_result.gpu_usage,
        raw_content=adapter_result.raw_content,
        raw_reasoning=adapter_result.raw_reasoning,
        answer_valid=adapter_result.answer_valid,
    )


def _metadata_doc_ids(metadata_list: list[dict]) -> tuple[str, ...]:
    doc_ids: list[str] = []
    for metadata in metadata_list:
        doc_id = metadata.get("doc_id") if metadata else None
        if doc_id is not None:
            doc_ids.append(str(doc_id))
    return tuple(doc_ids)


def run_single_benchmark(
    config: BenchmarkConfig, data: list[dict], run_dir: Path | None = None,
    corpus: list[dict] | None = None,
    load_data_seconds: float | None = None,
    resource_monitor: ResourceMonitor | None = None,
) -> BenchmarkResultExtended:
    run_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    if load_data_seconds is not None:
        stage_timings["load_data"] = load_data_seconds
    console.print(f"\n[bold yellow]>>> Starting: {config.name}[/bold yellow]")
    chunks = []
    rag_adapter = get_rag_adapter(config)

    # Prepare QA log path if run_dir is provided
    qa_log: list[dict] = []
    qa_log_path: Path | None = None
    if run_dir is not None:
        configs_dir = run_dir / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        safe_name = config.name.replace(":", "_").replace("/", "_")
        qa_log_path = configs_dir / f"{safe_name}_qa.json"

    # 1. Chunk + 2. Embed (only in retrieval mode)
    vector_store = None
    if rag_adapter is not None:
        with _stage_timer(stage_timings, "adapter_prepare", resource_monitor):
            rag_adapter.prepare(config, data, corpus=corpus)
        console.print(f"  [dim]Using external RAG adapter: {rag_adapter.name}[/dim]")
    elif config.retrieval_mode == "retrieval":
        chunks, vector_store = _build_internal_retrieval_index(
            config,
            data,
            corpus,
            stage_timings,
            resource_monitor,
        )
    else:
        console.print(f"  [dim]Direct mode — skipping chunking/retrieval[/dim]")

    if config.benchmark_stage == "index":
        total_time = time.perf_counter() - run_start
        stage_timings["total"] = total_time
        console.print(
            f"[bold green]<<< Indexed: {config.name} in {total_time:.1f}s[/bold green]"
        )
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
            avg_ttft_seconds=0,
            avg_tokens_per_second=0,
            avg_gpu_utilization_pct=None,
            avg_gpu_memory_used_mb=None,
            ragas_faithfulness=None,
            ragas_answer_relevancy=None,
            ragas_answer_correctness=None,
            ragas_context_precision=None,
            ragas_context_recall=None,
            ragas_semantic_similarity=None,
            total_time_seconds=total_time,
            per_sample=(),
            ttft_stats=None,
            tps_stats=None,
            gpu_util_stats=None,
            gpu_mem_stats=None,
            ragas_faithfulness_stats=None,
            ragas_answer_relevancy_stats=None,
            ragas_answer_correctness_stats=None,
            ragas_context_precision_stats=None,
            ragas_context_recall_stats=None,
            ragas_semantic_similarity_stats=None,
            retrieval_strategy=config.retrieval_strategy,
            retrieval_top_k=config.retrieval_top_k,
            dataset_name=config.dataset_name,
            dataset_sample_size=config.dataset_sample_size,
            stage_timings=stage_timings,
            vector_db_backend=config.vector_db_backend,
        )

    # 3. Generate answers
    llm = None
    reranker = None
    prompt_tmpl = None
    if rag_adapter is None:
        with _stage_timer(stage_timings, "load_models", resource_monitor):
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
    all_retrieved_metadata: list[list[dict]] = []
    gold_doc_ids: list[str | None] = []
    gen_results: list[GenerationResult] = []

    for i, sample in enumerate(data):
        console.print(f"  [cyan]({i + 1}/{len(data)})[/cyan] {sample['question'][:80]}{'...' if len(sample['question']) > 80 else ''}")

        if rag_adapter is not None:
            with _stage_timer(stage_timings, "external_rag", resource_monitor):
                adapter_result = rag_adapter.answer(sample, config)
            context_texts = adapter_result.contexts
            retrieved_metadata = adapter_result.metadata
            result = _generation_result_from_adapter(adapter_result)
        else:
            if llm is None or prompt_tmpl is None:
                raise RuntimeError("Internal RAG pipeline was not initialized.")

            # HyDE query expansion: replace the raw question with a hypothetical answer
            query = sample["question"]
            if config.retrieval_mode == "direct":
                context_texts = [sample["context"]]
                retrieved_metadata = []
            else:
                if config.retrieval_use_hyde:
                    with _stage_timer(stage_timings, "hyde", resource_monitor):
                        query = expand_query_with_hyde(llm, sample["question"])

                if vector_store is None:
                    raise RuntimeError("Vector store missing in retrieval mode.")
                with _stage_timer(stage_timings, "retrieve", resource_monitor):
                    retrieved_docs = retrieve(
                        vector_store, query, config.retrieval_top_k,
                        retrieval_strategy=config.retrieval_strategy,
                        fetch_k=config.retrieval_fetch_k,
                        mmr_lambda=config.retrieval_mmr_lambda,
                    )

                if reranker is not None:
                    with _stage_timer(stage_timings, "rerank", resource_monitor):
                        retrieved_docs = reranker.rerank(
                            sample["question"], retrieved_docs, config.reranker_top_k,
                        )

                context_texts = [doc.page_content for doc in retrieved_docs]
                retrieved_metadata = [dict(doc.metadata) for doc in retrieved_docs]

            with _stage_timer(stage_timings, "generate", resource_monitor):
                result = generate_answer(
                    llm, sample["question"], context_texts,
                    system_prompt=prompt_tmpl.system_prompt,
                    human_template=prompt_tmpl.human_template,
                    strip_mode=config.llm_answer_strip_mode,
                    value_fallback=config.llm_answer_value_fallback,
                    ground_truth=sample["ground_truth"],
                    prompt_template_name=config.prompt_template,
                )

        questions.append(sample["question"])
        ground_truths.append(sample["ground_truth"])
        all_contexts.append(context_texts)
        all_retrieved_metadata.append(retrieved_metadata)
        gold_doc_ids.append(
            sample.get("metadata", {}).get("gold_doc_id")
            if config.retrieval_mode == "retrieval"
            else None
        )
        gen_results.append(result)

        # Stream QA pair to log file after each answer
        if qa_log_path is not None:
            qa_log.append({
                "index": i,
                "question": sample["question"],
                "answer": result.answer,
                "raw_content": result.raw_content,
                "raw_reasoning": result.raw_reasoning,
            })
            qa_log_path.write_text(json.dumps(qa_log, indent=2, ensure_ascii=False))

    console.print(f"  [dim]Generated {len(gen_results)} answers[/dim]       ")

    # 4. Evaluate with RAGAS using a separate critic model
    console.print(f"  [dim]Running RAGAS evaluation (critic: {config.eval_critic_llm})...[/dim]")
    with _stage_timer(stage_timings, "ragas_eval", resource_monitor):
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

    # 4b. Compute custom (non-RAGAS) metrics
    console.print("  [dim]Computing custom metrics (IR + NLG)...[/dim]")
    
    # Reuse or create embedding model for IR relevance detection + context_relevance
    from benchmark.embedding import get_embedding_model
    with _stage_timer(stage_timings, "custom_metrics", resource_monitor):
        _emb_model = get_embedding_model(
            config.embedding_model,
            config.embedding_base_url(),
            config.embedding_api_key(),
            provider=config.embedding_provider,
        )
        _embed_fn = _emb_model.embed_query

        _bert_model = (
            _get_bert_model(config.custom_metrics_bert_model)
            if config.custom_metrics_bert_model
            else None
        )

        custom_result = compute_custom_metrics(
            questions, ground_truths,
            [r.answer for r in gen_results],
            all_contexts,
            embed_fn=_embed_fn,
            bert_model=_bert_model,
        )
        if config.custom_retrieval_metrics_mode == "gold_doc":
            custom_result = _with_gold_doc_retrieval_metrics(
                custom_result,
                gold_doc_ids,
                all_retrieved_metadata,
            )
    if custom_result.error:
        console.print(f"  [yellow]Custom metrics error: {custom_result.error}[/yellow]")
    else:
        console.print(f"  [dim]Custom metrics computed: {', '.join(custom_result.metric_means.keys())}[/dim]")

    total_time = time.perf_counter() - run_start
    stage_timings["total"] = total_time
    console.print(f"[bold green]<<< Finished: {config.name} in {total_time:.1f}s[/bold green]")

    # 5. Build per-sample results
    per_sample_custom = custom_result.per_sample
    all_retrieved_doc_ids = [
        _metadata_doc_ids(metadata_list)
        for metadata_list in all_retrieved_metadata
    ]
    all_ground_truth_doc_ids = [
        (str(doc_id),) if doc_id else ()
        for doc_id in gold_doc_ids
    ]
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
            custom_scores=per_sample_custom[i] if i < len(per_sample_custom) else {},
            answer_valid=gr.answer_valid,
            retrieved_doc_ids=all_retrieved_doc_ids[i]
            if i < len(all_retrieved_doc_ids)
            else (),
            ground_truth_doc_ids=all_ground_truth_doc_ids[i]
            if i < len(all_ground_truth_doc_ids)
            else (),
        )
        for i, (q, gt, ctx, gr) in enumerate(
            zip(questions, ground_truths, all_contexts, gen_results)
        )
    )

    invalid_count = sum(1 for s in per_sample if not s.answer_valid)
    if invalid_count:
        console.print(f"  [yellow]{invalid_count}/{len(per_sample)} answers were empty/invalid[/yellow]")

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

    def _custom_stat_samples(key: str) -> list[float]:
        vals = []
        for s in per_sample:
            if s.custom_scores:
                v = s.custom_scores.get(key)
                if v is not None:
                    vals.append(v)
        return vals

    custom_means = custom_result.metric_means
    custom_stat_summaries = {
        key: compute_stats(_custom_stat_samples(key))
        for key in custom_means
    }

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
        ragas_semantic_similarity=ragas_means.get("semantic_similarity"),
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
        ragas_semantic_similarity_stats=compute_stats(_ragas_stat_samples("semantic_similarity")),
        evaluation_error=eval_result.error,
        ragas_valid_sample_counts=eval_result.samples_with_valid_scores or None,
        custom_metric_means=custom_means if custom_means else None,
        custom_stats=custom_stat_summaries if custom_stat_summaries else None,
        reranker_model=config.reranker_model,
        reranker_top_k=config.reranker_top_k if config.reranker_model else None,
        retrieval_strategy=config.retrieval_strategy,
        retrieval_top_k=config.retrieval_top_k,
        dataset_name=config.dataset_name,
        dataset_sample_size=config.dataset_sample_size,
        stage_timings=stage_timings,
        vector_db_backend=config.vector_db_backend,
    )


def _with_gold_doc_retrieval_metrics(
    custom_result: CustomMetricsResult,
    gold_doc_ids: list[str | None],
    retrieved_metadata: list[list[dict]],
) -> CustomMetricsResult:
    gold_result = compute_gold_doc_retrieval_metrics(
        gold_doc_ids,
        retrieved_metadata,
    )
    metric_means = dict(custom_result.metric_means)
    metric_means.update(gold_result.metric_means)

    samples_with_valid_scores = dict(custom_result.samples_with_valid_scores)
    samples_with_valid_scores.update(gold_result.samples_with_valid_scores)

    per_sample: list[dict[str, float | None]] = []
    max_len = max(len(custom_result.per_sample), len(gold_result.per_sample))
    for i in range(max_len):
        scores: dict[str, float | None] = {}
        if i < len(custom_result.per_sample):
            scores.update(custom_result.per_sample[i])
        if i < len(gold_result.per_sample):
            scores.update(gold_result.per_sample[i])
        per_sample.append(scores)

    error = custom_result.error
    if gold_result.skipped_samples:
        msg = (
            f"Gold-doc retrieval metrics skipped {gold_result.skipped_samples} "
            "sample(s) without gold_doc_id."
        )
        error = f"{error}; {msg}" if error else msg

    return CustomMetricsResult(
        metric_means=metric_means,
        per_sample=per_sample,
        samples_with_valid_scores=samples_with_valid_scores,
        error=error,
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
    console.print(
        f"[bold]Running {len(configs)} benchmark configuration(s) "
        f"(stage: {configs[0].benchmark_stage}, vector DB: {configs[0].vector_db_backend})[/bold]"
    )

    # Load data — use shared corpus for datasets that support it
    load_stage: dict[str, float] = {}
    with _stage_timer(load_stage, "load_data"):
        from benchmark.dataset_adapters import get_adapter
        adapter = get_adapter(configs[0].dataset_name)
        corpus: list[dict] | None = None
        if adapter.has_shared_corpus:
            corpus, data = load_corpus_and_questions(
                dataset_name=configs[0].dataset_name,
                subset=configs[0].dataset_subset or None,
                sample_size=configs[0].dataset_sample_size,
            )
        else:
            data = load_benchmark_data(
                dataset_name=configs[0].dataset_name,
                subset=configs[0].dataset_subset or None,
                sample_size=configs[0].dataset_sample_size,
            )
    load_data_seconds = load_stage.get("load_data")

    # Create run directory up front so results are saved even if a
    # later step (MLflow, cleanup, etc.) crashes.
    run_dir = _next_run_dir()
    reproducibility_dir = write_reproducibility_bundle(run_dir, configs)
    console.print(f"[dim]Reproducibility bundle: {reproducibility_dir}[/dim]")
    wall_start = time.perf_counter()

    # Run sequentially: local models typically share a single GPU and process
    # requests serially. Parallel execution causes GPU memory thrashing,
    # request queuing, and timeouts that produce *lower* throughput.
    results: list[BenchmarkResultExtended] = []
    parent_context = mlflow.start_run(
        run_name=f"benchmark_env_matrix_{run_dir.name}",
        tags={
            "type": "benchmark_parent",
            "experiment_name": "env-matrix",
            "results_dir": str(run_dir),
            "num_configs": str(len(configs)),
        },
    ) if configs else nullcontext()
    with parent_context:
        for i, config in enumerate(configs):
            console.print(f"\n[bold cyan]Config {i + 1}/{len(configs)}[/bold cyan]")
            monitor = None
            if resource_monitor_enabled():
                safe_name = config.name.replace(":", "_").replace("/", "_")
                trace_dir = run_dir / "resource_traces"
                monitor = ResourceMonitor(
                    trace_dir / f"{safe_name}.csv",
                    trace_dir / f"{safe_name}_markers.csv",
                    interval_seconds=resource_monitor_interval(),
                    gpu_index=resource_monitor_gpu_index(),
                )

            if monitor is None:
                result = run_single_benchmark(
                    config,
                    data,
                    run_dir=run_dir,
                    corpus=corpus,
                    load_data_seconds=load_data_seconds,
                )
            else:
                with monitor:
                    result = run_single_benchmark(
                        config,
                        data,
                        run_dir=run_dir,
                        corpus=corpus,
                        load_data_seconds=load_data_seconds,
                        resource_monitor=monitor,
                    )
                console.print(f"[dim]  Resource trace: {monitor.trace_path}[/dim]")

            # Save to disk immediately (survives MLflow crashes)
            _save_config_result(result, run_dir)

            log_benchmark_run(
                result,
                reproducibility_dir=reproducibility_dir,
                nested=True,
            )
            results.append(result)

        store_path = ".chroma/" if configs[0].vector_db_backend == "chroma" else configs[0].lancedb_path
        console.print(f"\n[dim]Vector stores persisted in {store_path}[/dim]")

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

        # Log aggregate tables, reports, plots, and rerun metadata to MLflow.
        log_aggregate_artifacts_to_mlflow(
            run_dir,
            run_name=f"summary_{run_dir.name}_{timestamp}",
            reproducibility_dir=reproducibility_dir,
        )

    return results


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    console.print("[bold]RAG Benchmarking Framework[/bold]")
    console.print("=" * 50)
    tracking_uri = setup_tracing()
    console.print(f"[dim]MLflow tracking: {tracking_uri}[/dim]")
    setup_mlflow()
    run_all_benchmarks()


if __name__ == "__main__":
    main()
