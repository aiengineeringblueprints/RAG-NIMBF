"""Smoke tests for the Streamlit dashboard app (skip when streamlit missing)."""

import json

import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = pytest.mark.dashboard


@pytest.fixture
def live_results(tmp_path):
    """A small results tree with one completed run and one running run."""
    results = tmp_path / "results"
    done = results / "run1"
    done.mkdir(parents=True)
    (done / "benchmark_20260101_000000.json").write_text(
        json.dumps(
            {
                "timestamp": "20260101_000000",
                "num_configs": 1,
                "dataset": {"name": "ragbench_covidqa", "subset": "distractor", "sample_size": 10},
                "results": [
                    {
                        "config_name": "cfg_a",
                        "llm_model": "qwen2.5:0.5b",
                        "ragas_faithfulness": 0.9,
                        "avg_ttft_seconds": 0.2,
                        "per_sample": [
                            {
                                "question": "Q",
                                "answer": "A",
                                "ground_truth": "GT",
                                "ttft_seconds": 0.2,
                                "total_seconds": 0.5,
                                "token_count": 5,
                                "ragas_scores": {"faithfulness": 0.9},
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    running = results / "run2"
    running.mkdir(parents=True)
    (running / "progress.json").write_text(
        json.dumps(
            {
                "configs": {
                    "cfg_x": {"status": "completed", "completed_at": "2026-01-01T00:01:00"},
                    "cfg_y": {"status": "running", "started_at": "2026-01-01T00:02:00"},
                }
            }
        ),
        encoding="utf-8",
    )
    (running / "configs").mkdir()
    (running / "configs" / "cfg_x_qa.json").write_text(
        json.dumps(
            [
                {
                    "question": "Live Q",
                    "answer": "Live A",
                    "ground_truth": "GT",
                    "ttft_seconds": 0.1,
                    "total_seconds": 0.4,
                    "token_count": 3,
                }
            ]
        ),
        encoding="utf-8",
    )
    return results


def test_app_boots_without_exceptions(tmp_path, live_results):
    import os

    os.environ["DASHBOARD_RESULTS_DIR"] = str(live_results)
    app_path = __file__.rsplit("tests", 1)[0] + "dashboard/app.py"
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [(m.label, m.value) for m in at.metric]
    assert ("Runs gesamt", "2") in labels
    assert ("Aktiv (laufend)", "1") in labels


def test_live_tab_shows_progress(live_results):
    import os

    os.environ["DASHBOARD_RESULTS_DIR"] = str(live_results)
    app_path = __file__.rsplit("tests", 1)[0] + "dashboard/app.py"
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run()
    labels = [(m.label, m.value) for m in at.metric]
    assert ("Status", "running") in labels
    assert ("Configs gesamt", "2") in labels
    assert ("Fertig", "1") in labels


def test_no_runs_dir_shows_info(tmp_path):
    import os

    os.environ["DASHBOARD_RESULTS_DIR"] = str(tmp_path / "does-not-exist")
    app_path = __file__.rsplit("tests", 1)[0] + "dashboard/app.py"
    at = AppTest.from_file(app_path, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(i.value and "Keine Run-Verzeichnisse" in i.value for i in at.info)