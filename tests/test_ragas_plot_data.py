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


def test_ngen_mean_and_error_keeps_genuine_zero_scores():
    df = pd.DataFrame({"ragas_faithfulness": [None, 0.0, 0.8]})

    mean, err = metric_mean_and_error(df, "ragas_faithfulness")

    assert mean == 0.4
    assert round(err, 4) == 0.5657


# --- rendered charts -------------------------------------------------------

import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402

from benchmark.reporting import visualization  # noqa: E402


class _Run(_Result):
    def __init__(self, name, llm="m1", faithfulness=None, context_precision=None):
        super().__init__(faithfulness=faithfulness, context_precision=context_precision)
        self.config_name = name
        self.llm_model = llm
        self.custom_metric_means = None
        for key in visualization._METRIC_KEYS:
            setattr(self, f"ragas_{key}_stats", None)


@pytest.fixture
def saved_figures(monkeypatch):
    figures = []
    monkeypatch.setattr(
        matplotlib.figure.Figure, "savefig", lambda fig, *a, **k: figures.append(fig)
    )
    return figures


MIXED = [
    _Run("a", faithfulness=0.8, context_precision=0.6),
    _Run("b", faithfulness=0.7),
]


def test_comparison_bars_draw_nothing_for_missing_metric(saved_figures, tmp_path):
    visualization._plot_ragas_comparison(MIXED, tmp_path)

    ax = saved_figures[0].axes[0]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    heights = [bar.get_height() for bar in ax.patches]
    assert legend == ["Faithfulness", "Context Precision"]
    assert heights[:3] == [0.8, 0.7, 0.6]
    assert math.isnan(heights[3])


def test_heatmap_masks_missing_cell(saved_figures, tmp_path):
    visualization._plot_heatmap(MIXED, tmp_path)

    image = saved_figures[0].axes[0].get_images()[0].get_array()
    assert image.shape == (2, 2)
    assert image.mask.tolist() == [[False, False], [False, True]]


def test_interactive_radar_averages_only_measured_values():
    runs = [
        _Run("a", llm="m1", faithfulness=0.8),
        _Run("b", llm="m1"),  # RAGAS skipped
        _Run("c", llm="m2", faithfulness=0.6),
    ]

    fig = visualization.llm_comparison_radar_figure(runs)

    by_llm = {trace.name: list(trace.r) for trace in fig.data}
    assert by_llm["m1"] == [pytest.approx(0.8), pytest.approx(0.8)]
    assert by_llm["m2"] == [pytest.approx(0.6), pytest.approx(0.6)]


def test_interactive_radar_leaves_gap_for_metric_an_llm_lacks():
    runs = [_Run("a", llm="m1", faithfulness=0.8), _Run("b", llm="m2")]
    runs[1].custom_metric_means = {"rouge_l": 0.5}
    runs[0].custom_metric_means = {"rouge_l": 0.4}

    fig = visualization.llm_comparison_radar_figure(runs)

    by_llm = {trace.name: list(trace.r) for trace in fig.data}
    assert by_llm["m2"][0] is None  # Faithfulness, not 0
