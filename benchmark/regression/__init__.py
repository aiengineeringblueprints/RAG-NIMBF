"""Fixed regression benchmark for iterative Blueprint evaluation (issue #64).

Three steps, each a pure function so they can be tested without a model:

- :func:`freeze_question_set` turns loaded benchmark samples into a versioned
  dev split (the shared comparison set) and a disjoint holdout split (final
  confirmation only), with a content-hashed manifest.
- :func:`summarize_run` reduces one run's per-sample results to the issue's
  evaluation dimensions, reporting a dimension as ``unavailable`` instead of
  guessing when its labels or scores are missing.
- :func:`compare_runs` pairs a candidate with a reference run question by
  question and judges it against explicit improvement / trade-off criteria.

CLI (guide: ``doc/blueprint-regression-benchmark.md``)::

    python -m benchmark.regression freeze --dataset ragbench --subset techqa \
        --split train --dev-size 500 --holdout-size 200 \
        --out datasets/blueprint_regression/v1
    python -m benchmark.regression verify --question-set datasets/blueprint_regression/v1
    python -m benchmark.regression compare --question-set datasets/blueprint_regression/v1 \
        --criteria experiments/blueprint-regression/criteria.yaml \
        --baseline <report.json> --best <report.json> --candidate <report.json> \
        --out results/<run>/regression

Modules: ``question_set`` (freeze, keys, hashes), ``metrics`` (per-sample
metrics, run summary), ``criteria`` (typed decision policy), ``compare``
(paired statistics, verdict), ``runs`` (result loading, config consistency),
``report`` (Markdown), ``cli``.
"""

from benchmark.regression.cli import main
from benchmark.regression.compare import Paired, compare_runs
from benchmark.regression.criteria import Guardrail, RegressionCriteria
from benchmark.regression.metrics import LOWER_IS_BETTER, sample_metrics, summarize_run
from benchmark.regression.question_set import (
    SPLITS,
    FrozenQuestionSet,
    SampleKey,
    freeze_question_set,
    jsonl_lines,
    question_categories,
    read_jsonl,
    sample_keys,
    split_status,
)
from benchmark.regression.report import render_report
from benchmark.regression.runs import EVALUATION_FIELDS, check_run_configs, find_run_config, load_run_results

__all__ = [
    "EVALUATION_FIELDS", "FrozenQuestionSet", "Guardrail", "LOWER_IS_BETTER", "Paired", "RegressionCriteria",
    "SPLITS", "SampleKey", "check_run_configs", "compare_runs", "find_run_config", "freeze_question_set",
    "jsonl_lines", "load_run_results", "main", "question_categories", "read_jsonl", "render_report",
    "sample_keys", "sample_metrics", "split_status", "summarize_run",
]
