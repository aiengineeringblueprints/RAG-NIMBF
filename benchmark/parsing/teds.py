"""TEDS table-structure metric, vendored from PubTabNet (OCR-04).

Vendors the TEDS / TEDS-S implementation of IBM's PubTabNet evaluation code
(https://github.com/ibm-aur-nlp/PubTabNet, file ``src/metric.py``, Copyright
2020 IBM, Author peter.zhong@au1.ibm.com, licensed under the Apache License,
Version 2.0). OmniDocBench reuses this same implementation for its table
TEDS metric.

Upstream behavior retained exactly (validated against reference fixtures
computed with the pristine upstream code):

- HTML is parsed with lxml; the first ``<table>`` under ``body`` is compared
- the tree edit distance (APTED) is normalized by the larger node count of
  the two tables; ``1 - distance/n_nodes`` is the TEDS score
- cell content comparison is a character-level normalized Levenshtein
- ``structure_only=True`` gives TEDS-S: cell text is ignored, only the tag
  / colspan / rowspan skeleton is compared

Local deviations (score-neutral):

- the ``distance`` package is replaced by RapidFuzz's Levenshtein
  (identical integer edit distances)
- batch evaluation / parallelism helpers are not vendored; the per-table
  batching lives in :mod:`benchmark.parsing.tables`
"""

from __future__ import annotations

from apted import APTED, Config, helpers
from lxml import etree, html

# ── Vendored-algorithm identity (recorded into run metadata) ─────────

TEDS_ALGORITHM_NAME = "teds"
TEDS_ALGORITHM_VERSION = "PubTabNet TEDS (vendored, OmniDocBench table metric)"
TEDS_ALGORITHM_LICENSE = "Apache-2.0"
TEDS_ALGORITHM_SOURCE = "https://github.com/ibm-aur-nlp/PubTabNet"

__all__ = [
    "TEDS",
    "TEDS_ALGORITHM_LICENSE",
    "TEDS_ALGORITHM_NAME",
    "TEDS_ALGORITHM_SOURCE",
    "TEDS_ALGORITHM_VERSION",
    "teds",
    "teds_run_metadata",
    "teds_structure_only",
]


def teds_run_metadata() -> dict[str, str]:
    """Metric provenance for run metadata / reports."""
    return {
        "algorithm": TEDS_ALGORITHM_NAME,
        "version": TEDS_ALGORITHM_VERSION,
        "license": TEDS_ALGORITHM_LICENSE,
        "source": TEDS_ALGORITHM_SOURCE,
    }


# ── Vendored PubTabNet implementation ────────────────────────────────


class TableTree(helpers.Tree):
    """APTED tree node for one HTML table element (upstream)."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation (upstream)."""
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag,
                self.colspan,
                self.rowspan,
                self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """APTED cost model over table trees (upstream)."""

    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value."""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1."""
        from rapidfuzz.distance import Levenshtein

        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees."""
        if (
            (node1.tag != node2.tag)
            or (node1.colspan != node2.colspan)
            or (node1.rowspan != node2.rowspan)
        ):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity (upstream)."""

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (
            n_jobs >= 1
        ), "n_jobs must be an integer greather than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells."""
        self.__tokens__.append("<%s>" % node.tag)
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
            )
        else:
            new_node = TableTree(node.tag, None, None, None)
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth."""
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


def teds(
    pred_html: str | None,
    gt_html: str | None,
    structure_only: bool = False,
) -> float:
    """Score one predicted table against its ground truth (TEDS or TEDS-S)."""
    return TEDS(structure_only=structure_only).evaluate(pred_html, gt_html)


def teds_structure_only(pred_html: str | None, gt_html: str | None) -> float:
    """TEDS-S: structure-only variant (cell text ignored)."""
    return teds(pred_html, gt_html, structure_only=True)
