"""NLG metrics skip only real refusals, not genuine answers (issue #57)."""

import pytest

from benchmark.custom_metrics import compute_custom_metrics

# False positives from run8 (Enterprise RAG Blueprint, TechQA): genuine answers
# that contain words of the old broad refusal substring list.
GENUINE_ANSWERS = [
    "The error is typically caused by insufficient ulimit settings for the "
    "WebSphere process; raise the open files limit to 8192.",
    "Based on the provided context, WebSphere Application Server 8.5.5.6 "
    "addresses CVE-2015-0226 in Apache WSS4J.",
    # run7: short answer with "error" / "does not exist", which declines()
    # counts as a decline for the MCP trust set.
    "According to dataset-doc-250, this error occurs on Z/OS because the DB2 "
    "SYSIBMADM.DBCFG table view does not exist. The document explicitly states "
    "that this error message can be safely ignored.",
    # run8: the answer comes first, a caveat about the sources follows.
    "To resolve the issue, upgrade to Firefox 38.5 ESR, downgrade to Firefox "
    "38.3 ESR, or use Internet Explorer as a workaround. The provided documents "
    "do not contain a specific fix for the crash.",
]


def _rouge_l(answer: str, ground_truth: str) -> float:
    result = compute_custom_metrics(["q"], [ground_truth], [answer], [["ctx"]])
    return result.per_sample[0]["rouge_l"]


@pytest.mark.parametrize("answer", GENUINE_ANSWERS)
def test_genuine_answers_keep_their_nlg_scores(answer):
    assert _rouge_l(answer, answer) == pytest.approx(1.0)


@pytest.mark.parametrize("answer", [
    "",
    "I cannot answer this question based on the provided context.",
    "The provided context does not contain information about the fix pack.",
    "The provided context does not contain specific steps for the upgrade. "
    "The documents discuss unrelated SSD driver issues.",
])
def test_refusals_score_zero(answer):
    assert _rouge_l(answer, "Apply fix pack 8.5.5.6.") == 0.0
