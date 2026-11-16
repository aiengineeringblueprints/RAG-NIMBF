"""Composite RAGAS ranking score across runs with different metric sets."""

import math

import pandas as pd
import pytest

from benchmark.reporting.run_tracker import composite_ragas_score

WEIGHTS = {"ragas_faithfulness": 0.5, "ragas_context_precision": 0.5}


def test_skipped_metric_does_not_count_as_zero():
    df = pd.DataFrame({
        "ragas_faithfulness": [0.8, 0.8],
        "ragas_context_precision": [0.6, None],  # second run skipped it
    })

    scores = composite_ragas_score(df, WEIGHTS)

    assert scores.tolist() == pytest.approx([0.7, 0.8])


def test_row_without_any_ragas_metric_has_no_score():
    df = pd.DataFrame({
        "ragas_faithfulness": [None],
        "ragas_context_precision": [None],
    })

    assert math.isnan(composite_ragas_score(df, WEIGHTS).iloc[0])
