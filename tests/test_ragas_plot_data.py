"""Report charts show only RAGAS metrics that ran; gaps stay NaN, not 0."""

import math

import pandas as pd

from benchmark.reporting.ngen_ai_charts import metric_mean_and_error
from benchmark.reporting.visualization import ragas_plot_matrix


class _Result:
    def __init__(self, **ragas):
        for key in (
            "faithfulness", "semantic_similarity", "answer_relevancy",
            "context_precision", "context_recall",
        ):
            setattr(self, f"ragas_{key}", ragas.get(key))


def test_metrics_no_config_ran_are_left_out():
    keys, data = ragas_plot_matrix([_Result(faithfulness=0.8)])

    assert keys == ["faithfulness"]
    assert data.tolist() == [[0.8]]


def test_metric_missing_for_one_config_is_nan_not_zero():
    keys, data = ragas_plot_matrix([
        _Result(faithfulness=0.8, context_precision=0.6),
        _Result(faithfulness=0.7),
    ])

    assert keys == ["faithfulness", "context_precision"]
    assert data[0].tolist() == [0.8, 0.6]
    assert data[1][0] == 0.7
    assert math.isnan(data[1][1])


def test_ngen_mean_and_error_without_values_is_nan():
    df = pd.DataFrame({"ragas_faithfulness": [None, None]})

    mean, err = metric_mean_and_error(df, "ragas_faithfulness")

    assert math.isnan(mean)
    assert err == 0


def test_ngen_mean_and_error_for_missing_column_is_nan():
    mean, err = metric_mean_and_error(pd.DataFrame({"x": [1]}), "ragas_faithfulness")

    assert math.isnan(mean)
    assert err == 0


def test_ngen_mean_and_error_ignores_zero_ragas_placeholders():
    df = pd.DataFrame({"ragas_faithfulness": [0.0, 0.6, 0.8]})

    mean, err = metric_mean_and_error(df, "ragas_faithfulness")

    assert mean == 0.7
    assert round(err, 4) == 0.1414
