"""Glue layer that runs a workload against the live retrieve + generate path.

``main.py`` builds the vector store, generator LLM, reranker, and prompt
template exactly as in the sequential path, then hands them to
``run_workload_phase`` here. We assemble a ``query_callable`` that the
runner uses for every Query op and return populated result lists that
drop straight into the existing Ragas / per-sample aggregation.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.documents import Document

from benchmark.generation import (
    GenerationResult,
    generate_answer,
)
from benchmark.retrieval import retrieve
from benchmark.reranker import get_reranker
from benchmark.workload.generator import WorkloadConfig, WorkloadGenerator
from benchmark.workload.operations import QueryOp
from benchmark.workload.report import mlflow_metrics, summarize, write_summary_json
from benchmark.workload.runner import WorkloadRunner

logger = logging.getLogger(__name__)


@dataclass
class WorkloadPhaseResult:
    """Container handed back to main.py after the workload runs."""

    questions: list[str]
    ground_truths: list[str]
    contexts: list[list[str]]
    retrieved_metadata: list[list[dict]]
    gen_results: list[GenerationResult]
    gold_doc_ids: list[str | None]
    sample_metadata: list[dict]
    adapter_diagnostics: list[dict]
    mlflow_metrics: dict[str, float]
    summary_dict: dict[str, Any]
    summary_path: str | None


def _build_query_callable(
    *,
    vector_store: Any,
    llm: Any,
    prompt_tmpl: Any,
    config: Any,
    reranker: Any = None,
) -> Callable[[str], dict]:
    """Wrap retrieve -> generate in a callable for the runner."""
    def _query(question: str) -> dict:
        query_text = question
        if config.retrieval_use_hyde:
            from benchmark.retrieval import expand_query_with_hyde

            query_text = expand_query_with_hyde(llm, question)
        retrieved_docs = retrieve(
            vector_store,
            query_text,
            config.retrieval_top_k,
            retrieval_strategy=config.retrieval_strategy,
            fetch_k=config.retrieval_fetch_k,
            mmr_lambda=config.retrieval_mmr_lambda,
        )
        if reranker is not None:
            retrieved_docs = reranker.rerank(
                question, retrieved_docs, config.reranker_top_k,
            )
        context_texts = [doc.page_content for doc in retrieved_docs]
        result = generate_answer(
            llm,
            question,
            context_texts,
            system_prompt=prompt_tmpl.system_prompt,
            human_template=prompt_tmpl.human_template,
            strip_mode=config.llm_answer_strip_mode,
            value_fallback=config.llm_answer_value_fallback,
            ground_truth="",
            prompt_template_name=config.prompt_template,
            cost_model_name=config.llm_model,
        )
        return {
            "answer": result.answer,
            "contexts": context_texts,
            "metadata": [dict(doc.metadata) for doc in retrieved_docs],
            "result": result,
        }

    return _query


def _workload_config_from_benchmark(config: Any) -> WorkloadConfig:
    return WorkloadConfig(
        total_ops=config.workload_total_ops,
        target_qps=config.workload_target_qps,
        concurrency=config.workload_concurrency,
        distribution=config.workload_distribution,
        zipf_theta=config.workload_zipf_theta,
        op_mix_query=config.workload_op_mix_query,
        op_mix_insert=config.workload_op_mix_insert,
        op_mix_update=config.workload_op_mix_update,
        op_mix_remove=config.workload_op_mix_remove,
    )


def run_workload_phase(
    *,
    config: Any,
    data: list[dict],
    corpus: list[dict] | None,
    vector_store: Any,
    llm: Any,
    prompt_tmpl: Any,
    reranker: Any = None,
    run_dir: Any = None,
) -> WorkloadPhaseResult:
    """Run the configured workload and return populated result lists."""
    from rich.console import Console

    console = Console()
    if corpus is None:
        corpus = []
    if not corpus:
        raise ValueError(
            "Workload mode requires a non-empty corpus for Update/Remove sampling"
        )
    if not data:
        raise ValueError("Workload mode requires a non-empty question pool")

    workload_cfg = _workload_config_from_benchmark(config)
    generator = WorkloadGenerator(workload_cfg, data, corpus)
    query_callable = _build_query_callable(
        vector_store=vector_store,
        llm=llm,
        prompt_tmpl=prompt_tmpl,
        config=config,
        reranker=reranker,
    )

    runner = WorkloadRunner(
        config=workload_cfg,
        generator=generator,
        vector_store=vector_store,
        query_callable=query_callable,
    )
    console.print(
        f"  [bold magenta]Workload:[/bold magenta] {workload_cfg.total_ops} ops "
        f"at {workload_cfg.target_qps} QPS "
        f"(mix q/i/u/r = "
        f"{workload_cfg.op_mix_query}/{workload_cfg.op_mix_insert}/"
        f"{workload_cfg.op_mix_update}/{workload_cfg.op_mix_remove}, "
        f"{workload_cfg.distribution}, "
        f"concurrency={workload_cfg.concurrency})"
    )
    summary = runner.run()

    summary_dict = summarize(summary)
    metrics = mlflow_metrics(summary)
    summary_path = None
    if run_dir is not None:
        out = write_summary_json(summary, run_dir / "workload_summary.json")
        summary_path = str(out)

    # Map query records back into the aggregate-collection shape that the
    # rest of main.py expects. Non-query ops are summarised but do not
    # contribute to Ragas scoring.
    questions: list[str] = []
    ground_truths: list[str] = []
    all_contexts: list[list[str]] = []
    all_retrieved_metadata: list[list[dict]] = []
    gen_results: list[GenerationResult] = []
    sample_metadata: list[dict] = []
    adapter_diagnostics: list[dict] = []
    gold_doc_ids: list[str | None] = []

    for record in summary.records:
        if record.kind != "query" or record.outcome != "ok":
            continue
        questions.append(record.question or "")
        ground_truths.append(record.ground_truth or "")
        all_contexts.append(list(record.contexts or []))
        # The query_callable returned metadata inside its dict, but the
        # record only stores contexts. Recover metadata via empty fallback.
        all_retrieved_metadata.append([])
        sample_metadata.append({})
        adapter_diagnostics.append({})
        gold_doc_ids.append(None)
        gen_results.append(
            GenerationResult(
                answer=record.answer or "",
                ttft_seconds=0.0,
                total_seconds=record.latency_seconds,
                token_count=0,
                tokens_per_second=0.0,
                gpu_usage=None,
                answer_valid=bool(record.answer),
            )
        )

    console.print(
        f"  [dim]Workload done: {summary.total_completed} ops "
        f"({summary.total_errors} errors, {summary.total_stale_check} stale-check) "
        f"in {summary.wall_seconds:.1f}s, observed QPS={summary.observed_qps:.2f}[/dim]"
    )

    return WorkloadPhaseResult(
        questions=questions,
        ground_truths=ground_truths,
        contexts=all_contexts,
        retrieved_metadata=all_retrieved_metadata,
        gen_results=gen_results,
        gold_doc_ids=gold_doc_ids,
        sample_metadata=sample_metadata,
        adapter_diagnostics=adapter_diagnostics,
        mlflow_metrics=metrics,
        summary_dict=summary_dict,
        summary_path=summary_path,
    )
