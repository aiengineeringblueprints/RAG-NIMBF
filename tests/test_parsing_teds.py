"""TEDS table metric (OCR-04): vendored PubTabNet implementation."""

from __future__ import annotations

import pytest

from benchmark.parsing.teds import (
    TEDS_ALGORITHM_LICENSE,
    TEDS_ALGORITHM_NAME,
    TEDS_ALGORITHM_SOURCE,
    TEDS_ALGORITHM_VERSION,
    teds,
    teds_run_metadata,
    teds_structure_only,
)

GT = (
    "<html><body><table><thead><tr><th>A</th><th>B</th></tr></thead>"
    "<tbody><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr>"
    "</tbody></table></body></html>"
)
PRED_SAME = GT
PRED_CHANGED_CELL = (
    "<html><body><table><thead><tr><th>A</th><th>B</th></tr></thead>"
    "<tbody><tr><td>1</td><td>X</td></tr><tr><td>3</td><td>4</td></tr>"
    "</tbody></table></body></html>"
)
PRED_MISSING_ROW = (
    "<html><body><table><thead><tr><th>A</th><th>B</th></tr></thead>"
    "<tbody><tr><td>1</td><td>2</td></tr></tbody></table></body></html>"
)

# Reference values computed with the pristine upstream PubTabNet
# implementation (ibm-aur-nlp/PubTabNet src/metric.py, Apache-2.0),
# pinned here so the vendored copy must reproduce them exactly.
UPSTREAM_FIXTURES = {
    ("same", False): 1.0,
    ("changed_cell", False): 0.9090909090909091,
    ("missing_row", False): 0.7272727272727273,
    ("changed_cell", True): 1.0,
}


@pytest.mark.parametrize(
    ("pred", "structure_only", "expected"),
    [
        (PRED_SAME, False, UPSTREAM_FIXTURES[("same", False)]),
        (PRED_CHANGED_CELL, False, UPSTREAM_FIXTURES[("changed_cell", False)]),
        (PRED_MISSING_ROW, False, UPSTREAM_FIXTURES[("missing_row", False)]),
        (PRED_CHANGED_CELL, True, UPSTREAM_FIXTURES[("changed_cell", True)]),
    ],
)
def test_teds_matches_upstream_reference_fixture(pred, structure_only, expected):
    assert teds(pred, GT, structure_only=structure_only) == pytest.approx(expected)


def test_teds_empty_prediction_scores_zero():
    assert teds("", GT) == 0.0
    assert teds(None, GT) == 0.0


def test_teds_no_table_in_document_scores_zero():
    assert teds("<html><body><p>no table</p></body></html>", GT) == 0.0


def test_teds_run_metadata_records_provenance():
    meta = teds_run_metadata()
    assert meta["algorithm"] == TEDS_ALGORITHM_NAME
    assert meta["license"] == TEDS_ALGORITHM_LICENSE == "Apache-2.0"
    assert meta["source"] == TEDS_ALGORITHM_SOURCE
    assert "PubTabNet" in TEDS_ALGORITHM_VERSION
    assert teds_structure_only.__doc__ is not None
