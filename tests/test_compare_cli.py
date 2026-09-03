from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmark.compare import collect_configs, cross_run_comparisons


def _write_run(
    run_dir: Path,
    configs: list[tuple[str, list[dict]]],
    dataset: str = "squad",
    sample_size: int = 6,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset_name": dataset,
        "dataset_subset": None,
        "dataset_sample_size": sample_size,
        "results": [
            {"config_name": name, "per_sample": rows} for name, rows in configs
        ],
    }
    (run_dir / "benchmark_20260101-000000.json").write_text(json.dumps(report))


def _rows(base: float) -> list[dict]:
    return [
        {
            "question": f"q{i}",
            "ragas_scores": {"faithfulness": base + i * 0.01},
            "custom_scores": {},
        }
        for i in range(6)
    ]


def test_collect_configs_merges_runs_and_checks_dataset_parity(tmp_path: Path):
    _write_run(tmp_path / "run1", [("a", _rows(0.5))])
    _write_run(tmp_path / "run2", [("b", _rows(0.7))])

    label, configs = collect_configs([tmp_path / "run1", tmp_path / "run2"])

    assert label == "squad/-@6"
    assert sorted(configs) == ["a", "b"]

    _write_run(tmp_path / "run3", [("c", _rows(0.1))], dataset="hotpotqa")
    with pytest.raises(ValueError, match="different datasets"):
        collect_configs([tmp_path / "run1", tmp_path / "run3"])


def test_cross_run_comparisons_produces_reliable_and_noise_rows(tmp_path: Path):
    _write_run(tmp_path / "run1", [("internal", _rows(0.5))])
    _write_run(
        tmp_path / "run2",
        [
            ("much_better", [{"question": f"q{i}", "ragas_scores": {"faithfulness": 0.9}, "custom_scores": {}} for i in range(6)]),
            ("same", _rows(0.5)),
        ],
    )

    payload = cross_run_comparisons([tmp_path / "run1", tmp_path / "run2"], n_resamples=500)

    by_config = {entry["config"]: entry for entry in payload}
    reliable = {c["metric"]: c for c in by_config["much_better"]["comparisons"]}
    assert reliable["faithfulness"]["reliable"] is True
    noise = {c["metric"]: c for c in by_config["same"]["comparisons"]}
    assert noise["faithfulness"]["reliable"] is False


def test_cross_run_comparisons_explicit_and_missing_baseline(tmp_path: Path):
    _write_run(tmp_path / "run1", [("a", _rows(0.5)), ("b", _rows(0.8))])
    _write_run(tmp_path / "run2", [("c", _rows(0.6))])

    payload = cross_run_comparisons(
        [tmp_path / "run1", tmp_path / "run2"], baseline="b", n_resamples=200
    )
    assert {entry["baseline"] for entry in payload} == {"b"}
    assert {entry["config"] for entry in payload} == {"a", "c"}

    with pytest.raises(ValueError, match="not among configs"):
        cross_run_comparisons([tmp_path / "run1", tmp_path / "run2"], baseline="nope")

    _write_run(tmp_path / "single", [("only", _rows(0.5))])
    with pytest.raises(ValueError, match="at least 2 distinct configs"):
        cross_run_comparisons([tmp_path / "single"], baseline="only")


def test_cross_run_comparisons_surfaces_misalignment(tmp_path: Path):
    _write_run(tmp_path / "run1", [("a", _rows(0.5))])
    bad_rows = [
        {"question": f"other{i}", "ragas_scores": {"faithfulness": 0.9}, "custom_scores": {}}
        for i in range(6)
    ]
    _write_run(tmp_path / "run2", [("b", bad_rows)])

    payload = cross_run_comparisons([tmp_path / "run1", tmp_path / "run2"], n_resamples=100)
    assert "misaligned" in payload[0]["error"]
