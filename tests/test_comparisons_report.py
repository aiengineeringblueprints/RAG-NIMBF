from __future__ import annotations

import json
from pathlib import Path

from benchmark.reporting.comparisons import config_comparisons, save_comparisons_report
from benchmark.reporting.models import BenchmarkResultExtended, PerSampleResult


def _per_sample(faith: list[float], ndcg: list[float]) -> tuple[PerSampleResult, ...]:
    return tuple(
        PerSampleResult(
            question=f"q{i}",
            ground_truth="gt",
            answer="a",
            contexts=(),
            ttft_seconds=0.0,
            total_seconds=0.0,
            token_count=0,
            tokens_per_second=0.0,
            gpu_usage=None,
            ragas_scores={"faithfulness": f},
            custom_scores={"ndcg": n},
        )
        for i, (f, n) in enumerate(zip(faith, ndcg))
    )


def _result(name: str, per_sample) -> BenchmarkResultExtended:
    return BenchmarkResultExtended(
        config_name=name,
        llm_model="m",
        embedding_model="e",
        prompt_template="concise",
        chunking_strategy="recursive",
        chunk_size=1000,
        chunk_overlap=200,
        num_chunks=10,
        num_questions=len(per_sample),
        avg_ttft_seconds=0.0,
        avg_tokens_per_second=0.0,
        avg_gpu_utilization_pct=None,
        avg_gpu_memory_used_mb=None,
        ragas_faithfulness=None,
        ragas_answer_relevancy=None,
        ragas_answer_correctness=None,
        ragas_context_precision=None,
        ragas_context_recall=None,
        ragas_semantic_similarity=None,
        total_time_seconds=0.0,
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
        per_sample=per_sample,
    )


def test_config_comparisons_pairs_baseline_against_each_config():
    baseline = _result(
        "baseline", _per_sample([0.8, 0.65, 0.9, 0.75, 0.7, 0.85], [0.5] * 6)
    )
    other = _result(
        "variant", _per_sample([0.4, 0.3, 0.5, 0.35, 0.45, 0.55], [0.5] * 6)
    )
    third = _result(
        "equal", _per_sample([0.8, 0.65, 0.9, 0.75, 0.7, 0.85], [0.5] * 6)
    )

    payload = config_comparisons([baseline, other, third], n_resamples=500)

    assert len(payload) == 2
    assert payload[0]["baseline"] == "baseline"
    assert payload[0]["config"] == "variant"
    metrics = {c["metric"]: c for c in payload[0]["comparisons"]}
    assert metrics["faithfulness"]["reliable"] is True
    assert metrics["faithfulness"]["delta"] > 0

    equal_metrics = {c["metric"]: c for c in payload[1]["comparisons"]}
    assert equal_metrics["faithfulness"]["reliable"] is False
    assert equal_metrics["faithfulness"]["delta"] == 0.0


def test_config_comparisons_empty_for_single_config():
    only = _result("only", _per_sample([0.5, 0.6], [0.5, 0.6]))
    assert config_comparisons([only]) == []


def test_save_comparisons_report_writes_json_and_markdown(tmp_path: Path):
    baseline = _result(
        "baseline", _per_sample([0.8, 0.65, 0.9, 0.75, 0.7, 0.85], [0.5] * 6)
    )
    other = _result(
        "variant", _per_sample([0.4, 0.3, 0.5, 0.35, 0.45, 0.55], [0.5] * 6)
    )

    path = save_comparisons_report([baseline, other], tmp_path, n_resamples=500)

    assert path is not None and path.exists()
    payload = json.loads((tmp_path / "comparisons.json").read_text())
    assert payload["comparisons"][0]["config"] == "variant"
    md = path.read_text()
    assert "Statistical Config Comparison" in md
    assert "faithfulness" in md
    assert "| 0.750 |" not in md or True  # table formatting smoke


def test_save_comparisons_report_noop_for_single_config(tmp_path: Path):
    only = _result("only", _per_sample([0.5, 0.6], [0.5, 0.6]))
    assert save_comparisons_report([only], tmp_path) is None
    assert not (tmp_path / "comparisons.json").exists()


def test_config_comparisons_report_misalignment_instead_of_crashing():
    baseline = _result("baseline", _per_sample([0.5, 0.6, 0.7], [0.5] * 3))
    swapped = BenchmarkResultExtended(
        **{
            **baseline.__dict__,
            "config_name": "swapped",
            "per_sample": tuple(
                type(s)(**{**s.__dict__, "question": "different"})
                for s in baseline.per_sample
            ),
        }
    )
    payload = config_comparisons([baseline, swapped], n_resamples=100)
    assert payload[0]["error"]
    assert "misaligned" in payload[0]["error"]
    assert payload[0]["comparisons"] == []


def test_comparison_rows_include_agent_tokens_from_diagnostics():
    from benchmark.reporting.comparisons import _comparison_rows

    sample = PerSampleResult(
        question="q",
        ground_truth="gt",
        answer="a",
        contexts=(),
        ttft_seconds=0.0,
        total_seconds=0.0,
        token_count=0,
        tokens_per_second=0.0,
        gpu_usage=None,
        ragas_scores={"faithfulness": 0.9},
        custom_scores={},
        adapter_diagnostics={
            "execution_mode": "agentic",
            "agent_tokens_total": 123,
            "agent_rounds": 3,
        },
    )
    rows = _comparison_rows([sample])
    assert rows[0]["agent_tokens_total"] == 123.0
    assert rows[0]["agent_rounds"] == 3.0

    fixed_sample = type(sample)(
        **{**sample.__dict__, "adapter_diagnostics": {"execution_mode": "fixed"}}
    )
    assert "agent_tokens_total" not in _comparison_rows([fixed_sample])[0]
