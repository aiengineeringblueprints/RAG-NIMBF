"""Stage-gate: experiment manifests for the three v1 OCR GT datasets (OCR-08).

The manifests must expand into a valid parsing-stage config matrix and carry
the dataset license flags pinned by the dataset adapters.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.orchestration.matrix import (
    build_configs_from_spec,
    load_experiment_spec,
)

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"

# manifest → (dataset name, license, research_only)
OCR_MANIFESTS = {
    "ocr_parser_omnidocbench.yaml": ("omnidocbench", "research-only", True),
    "ocr_parser_dp_bench.yaml": ("dp-bench", "MIT", False),
    "ocr_parser_olmocr_bench.yaml": ("olmocr-bench", "Apache-2.0", False),
}


@pytest.mark.parametrize("manifest", sorted(OCR_MANIFESTS))
def test_ocr_manifest_expands_to_parsing_matrix(manifest):
    dataset_name, license_name, research_only = OCR_MANIFESTS[manifest]

    spec = load_experiment_spec(EXPERIMENTS_DIR / manifest)
    assert spec.dataset["name"] == dataset_name

    configs = build_configs_from_spec(spec)

    assert configs, "manifest must expand to at least one config cell"
    for config in configs:
        assert config.benchmark_stage == "parsing"
        assert config.parser_adapter, "parsing cells need a parser adapter"
        assert config.dataset_license == license_name
        assert config.dataset_research_only is research_only
        # Evaluator LLMs must stay off for parsing-only runs.
        assert config.ragas_enabled is False
        assert config.custom_metrics_enabled is False
