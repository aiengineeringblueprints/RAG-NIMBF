"""Tests for the dashboard data loader."""

import json

import pytest

from dashboard import loader
from dashboard.loader import (
    RunData,
    config_metrics,
    load_run,
    per_sample_metric_values,
    scan_runs,
    summarize_config,
)

QUALITY_KEYS = [
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "ragas_answer_correctness",
    "ragas_context_precision",
    "ragas_context_recall",
    "ragas_semantic_similarity",
]


def _write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _sample_config(config_name="cfg_a", custom_means=None):
    return {
        "config_name": config_name,
        "llm_model": "qwen2.5:0.5b",
        "embedding_model": "nomic-embed-text:latest",
        "prompt_template": "concise",
        "chunking_strategy": "recursive",
        "chunk_size": 256,
        "chunk_overlap": 32,
        "num_chunks": 121,
        "num_questions": 10,
        "ragas_faithfulness": 0.9,
        "ragas_answer_relevancy": None,
        "ragas_context_recall": 1.0,
        "avg_ttft_seconds": 0.2,
        "avg_tokens_per_second": 60.0,
        "total_time_seconds": 6.5,
        "total_tokens": 3711,
        "stage_timings": {"retrieve": 0.22, "generate": 3.0, "total": 6.5},
        "stage_latency": {
            "chunking": {"count": 1, "mean_s": 0.08, "p95_s": 0.08, "total_s": 0.08},
            "indexing": {"count": 1, "mean_s": 1.2, "p95_s": 1.2, "total_s": 1.2},
        },
        "custom_metric_means": custom_means
        or {"context_relevance": 0.65, "hit@1": 0.74, "rouge_l": 0.5},
        "per_sample": [
            {
                "question": "Q?",
                "ground_truth": "GT",
                "answer": "A",
                "ttft_seconds": 0.1,
                "total_seconds": 0.5,
                "token_count": 21,
                "output_tokens": 21,
                "input_tokens": 396,
                "answer_valid": True,
                "ragas_scores": {"ragas_faithfulness": 0.8, "ragas_context_recall": 1.0},
                "custom_scores": {
                    "context_relevance": 0.6,
                    "hit@1": 1.0,
                    "trace_utilization": 0.4,
                },
            }
        ],
    }


@pytest.fixture
def results_dir(tmp_path):
    return tmp_path / "results"


def _build_completed_run(results_dir, run_name="run1"):
    run_dir = results_dir / run_name
    _write_json(
        run_dir / "benchmark_20260101_000000.json",
        {
            "timestamp": "20260101_000000",
            "num_configs": 1,
            "dataset": {
                "name": "ragbench_covidqa",
                "subset": "distractor",
                "sample_size": 500,
            },
            "system_info": {"python": "3.12.3", "os": "Linux", "gpu": "RTX 3050"},
            "results": [_sample_config()],
        },
    )
    return run_dir


def _build_running_run(results_dir, run_name="run2"):
    run_dir = results_dir / run_name
    _write_json(
        run_dir / "progress.json",
        {
            "configs": {
                "cfg_a": {"status": "completed", "completed_at": "2026-01-01T00:01:00"},
                "cfg_b": {"status": "running", "started_at": "2026-01-01T00:02:00"},
                "cfg_c": {"status": "failed", "error": "boom"},
            }
        },
    )
    _write_json(
        run_dir / "configs" / "cfg_a_qa.json",
        [
            {
                "question": "Q?",
                "answer": "A",
                "ground_truth": "GT",
                "ttft_seconds": 0.1,
                "total_seconds": 0.5,
                "token_count": 21,
                "output_tokens": 21,
                "input_tokens": 396,
            }
        ],
    )
    return run_dir


class TestScanRuns:
    def test_empty_dir(self, results_dir):
        assert scan_runs(results_dir) == []

    def test_missing_dir(self, tmp_path):
        assert scan_runs(tmp_path / "nope") == []

    def test_completed_run(self, results_dir):
        _build_completed_run(results_dir)
        runs = scan_runs(results_dir)
        assert len(runs) == 1
        info = runs[0]
        assert info.status == "completed"
        assert info.run_name == "run1"
        assert info.dataset_name == "ragbench_covidqa"
        assert info.num_configs == 1
        assert info.num_done == 1
        assert info.timestamp == "20260101_000000"

    def test_running_run(self, results_dir):
        _build_running_run(results_dir)
        runs = scan_runs(results_dir)
        assert runs[0].status == "running"
        assert runs[0].num_configs == 3
        assert runs[0].num_done == 1
        assert runs[0].num_failed == 1

    def test_sorted_newest_first(self, results_dir):
        _build_completed_run(results_dir, "run1")
        _build_completed_run(results_dir, "run2")
        runs = scan_runs(results_dir)
        assert [r.run_name for r in runs] == ["run2", "run1"]


class TestLoadRun:
    def test_completed(self, results_dir):
        _build_completed_run(results_dir)
        data = load_run(results_dir / "run1")
        assert isinstance(data, RunData)
        assert len(data.configs) == 1
        assert data.configs[0]["config_name"] == "cfg_a"
        assert data.info.status == "completed"

    def test_running_synthesizes_configs(self, results_dir):
        _build_running_run(results_dir)
        data = load_run(results_dir / "run2")
        assert data.info.status == "running"
        assert len(data.live_configs) == 3
        statuses = {lc.name: lc.status for lc in data.live_configs}
        assert statuses["cfg_b"] == "running"
        assert statuses["cfg_c"] == "failed"
        # cfg_a has a QA log -> synthesized config present.
        names = [c["config_name"] for c in data.configs]
        assert "cfg_a" in names
        cfg_a = next(c for c in data.configs if c["config_name"] == "cfg_a")
        assert cfg_a["_live"] is True
        assert cfg_a["num_questions"] == 1
        assert cfg_a["total_time_seconds"] == 0.5
        # QA log exposed for live inspection.
        assert "cfg_a" in data.qa_logs
        assert data.qa_logs["cfg_a"][0]["question"] == "Q?"

    def test_running_unknown_config_from_qa_log(self, results_dir):
        run_dir = _build_running_run(results_dir)
        # QA log for a config that progress.json does not know about (main.py mode).
        _write_json(
            run_dir / "configs" / "orphan_qa.json",
            [{"question": "X", "answer": "Y", "ground_truth": "Z", "total_seconds": 1.0}],
        )
        data = load_run(run_dir)
        names = [lc.name for lc in data.live_configs]
        assert "orphan" in names


class TestConfigMetrics:
    def test_normalizes_ragas_and_custom(self):
        cfg = _sample_config()
        metrics = config_metrics(cfg)
        for key in QUALITY_KEYS:
            assert key in metrics
        assert metrics["ragas_faithfulness"] == 0.9
        assert metrics["custom_context_relevance"] == 0.65
        # Legacy "@" keys are normalized to _at_.
        assert metrics["custom_hit_at_1"] == 0.74
        assert metrics["custom_rouge_l"] == 0.5
        assert metrics["avg_ttft_seconds"] == 0.2
        assert metrics["total_tokens"] == 3711

    def test_legacy_flat_custom_keys(self):
        cfg = _sample_config()
        cfg.pop("custom_metric_means")
        cfg.update({"hit@1": 0.7, "ndcg@1": 0.7, "rouge_l": 0.4})
        metrics = config_metrics(cfg)
        assert metrics["custom_hit_at_1"] == 0.7
        assert metrics["custom_ndcg_at_1"] == 0.7
        assert metrics["custom_rouge_l"] == 0.4

    def test_llm_performance_metrics(self):
        cfg = _sample_config()
        cfg["llm_performance_metrics"] = {"generation_sequential_latency_p95_s": 1.3}
        metrics = config_metrics(cfg)
        assert metrics["llm_perf_generation_sequential_latency_p95_s"] == 1.3

    def test_missing_values_none(self):
        cfg = _sample_config()
        cfg["ragas_faithfulness"] = None
        metrics = config_metrics(cfg)
        assert metrics["ragas_faithfulness"] is None


class TestPerSampleMetricValues:
    def test_extracts_ragas_and_custom(self):
        cfg = _sample_config()
        vals = per_sample_metric_values(cfg, "ragas_faithfulness")
        assert vals == [0.8]
        vals = per_sample_metric_values(cfg, "custom_context_relevance")
        assert vals == [0.6]
        vals = per_sample_metric_values(cfg, "custom_trace_utilization")
        assert vals == [0.4]

    def test_ragas_short_keys_in_samples(self):
        cfg = _sample_config()
        cfg["per_sample"][0]["ragas_scores"] = {
            "faithfulness": 0.75,
            "context_recall": 0.9,
        }
        assert per_sample_metric_values(cfg, "ragas_faithfulness") == [0.75]
        assert per_sample_metric_values(cfg, "ragas_context_recall") == [0.9]

    def test_flat_latency_field(self):
        cfg = _sample_config()
        vals = per_sample_metric_values(cfg, "ttft_seconds")
        assert vals == [0.1]


class TestSummarizeConfig:
    def test_shape(self):
        row = summarize_config(_sample_config())
        assert row["config_name"] == "cfg_a"
        assert row["llm_model"] == "qwen2.5:0.5b"
        assert row["ragas_faithfulness"] == 0.9


class TestMetricLabel:
    def test_known_and_unknown(self):
        assert loader.metric_label("ragas_faithfulness") == "Faithfulness"
        assert loader.metric_label("totally_new_metric") == "totally_new_metric"