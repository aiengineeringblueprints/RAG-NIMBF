"""GT↔prediction alignment via vendored OmniDocBench ``quick_match`` (OCR-03).

Vendors the text-line semantics of OmniDocBench's ``match_gt2pred_quick``
(https://github.com/opendatalab/OmniDocBench, file
``src/core/matching/match_quick.py``, Copyright opendatalab, licensed under
the Apache License, Version 2.0): both sides are
segmented into paragraphs, a normalized-Levenshtein cost matrix is built,
and truncated/merged predictions are recovered by adjacency search before a
Hungarian assignment produces the final aligned pairs.

Upstream behavior retained here:

- certain matches: normalized edit distance < 0.25
- truncation/merge recovery: an unmatched GT line greedily absorbs adjacent
  unmatched prediction lines while a fuzzy sub-match (threshold 0.6) keeps
  improving; overlapping merge candidates resolve by lowest average cost
- post-assignment filter: assigned pairs with cost > 0.7 fall back to
  unmatched (edit = 1)
- residual unmatched GT/prediction lines are paired up greedily, each with
  edit = 1

The formula/category-specific paths of upstream (LaTeX environments,
category ignores, position tracking) are out of scope for plain-text page
scoring and intentionally not ported.
"""

from __future__ import annotations

import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment

# ── Vendored-algorithm identity (recorded into run metadata) ─────────

MATCH_ALGORITHM_NAME = "quick_match"
MATCH_ALGORITHM_VERSION = "OmniDocBench v1.5 quick_match (vendored)"
MATCH_ALGORITHM_LICENSE = "Apache-2.0"
MATCH_ALGORITHM_SOURCE = "https://github.com/opendatalab/OmniDocBench"

CERTAIN_MATCH_THRESHOLD = 0.25
UNMATCHED_THRESHOLD = 0.7
FUZZY_MERGE_THRESHOLD = 0.6
MAX_TRUNCATED_PRED_MERGE = 160


def normalize_text(text: str | None) -> str:
    """Fold text the way OmniDocBench-style OCR evaluation expects.

    Applies NFKC unicode normalization (fullwidth -> ASCII, ligature
    expansion), accent stripping, lowercasing, and whitespace collapsing to
    single spaces.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(
        ch for ch in decomposed if not unicodedata.combining(ch)
    )
    folded = unicodedata.normalize("NFKC", stripped)
    return " ".join(folded.casefold().split())


def alignment_run_metadata() -> dict[str, str]:
    """Matching-algorithm provenance for run metadata / reports."""
    return {
        "algorithm": MATCH_ALGORITHM_NAME,
        "version": MATCH_ALGORITHM_VERSION,
        "license": MATCH_ALGORITHM_LICENSE,
        "source": MATCH_ALGORITHM_SOURCE,
    }


# ── Segmentation ─────────────────────────────────────────────────────


def split_paragraphs(text: str | None) -> list[str]:
    """Segment page text into paragraphs, OmniDocBench-style.

    Blank lines separate paragraphs; single newlines are treated as soft
    line wraps inside a paragraph. A single block that only contains single
    newlines falls back to one line per entry so line-level granularity is
    preserved for plain-text sources.
    """
    if not text:
        return []
    blocks = [block.strip() for block in text.split("\n\n")]
    blocks = [block for block in blocks if block]
    if len(blocks) <= 1:
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]
        if len(lines) > len(blocks):
            return lines
    return blocks


# ── Alignment result ─────────────────────────────────────────────────


@dataclass(frozen=True)
class AlignedPair:
    """One aligned GT↔prediction pair produced by :func:`quick_match`.

    ``matched`` is False for truncation-style misses: the pair exists but
    counts as a full error (edit = 1). ``gt_indices``/``pred_indices`` refer
    to positions in the segmented inputs; empty-GT pairs carry no GT side
    and vice versa.
    """

    gt: str
    pred: str
    edit: float
    matched: bool
    gt_indices: tuple[int, ...] = ()
    pred_indices: tuple[int, ...] = ()
    extra: dict = field(default_factory=dict)


# ── Normalized edit-distance primitives (upstream) ───────────────────


def _lev(gt: str, pred: str) -> float:
    return Levenshtein.distance(gt, pred)


def _normalized(a: str, b: str) -> float:
    try:
        return _lev(a, b) / max(len(a), len(b))
    except ZeroDivisionError:
        return 1.0


def _cost_matrix(gt_lines: list[str], pred_lines: list[str]) -> np.ndarray:
    return np.array(
        [
            [_normalized(g, p) for p in pred_lines]
            for g in gt_lines
        ]
    ).reshape(len(gt_lines), len(pred_lines))


def _sub_pred_fuzzy_matching(gt: str, pred: str) -> float | bool:
    """Best distance of ``pred`` as a sliding-window sub-match inside ``gt``."""
    gt_len, pred_len = len(gt), len(pred)
    if gt_len >= pred_len > 0:
        min_d = float("inf")
        for i in range(gt_len - pred_len + 1):
            dist = _lev(gt[i : i + pred_len], pred) / pred_len
            min_d = min(min_d, dist)
        return min_d
    return False


def _judge_pred_merge(
    gt: str, pred_lines: list[str], threshold: float = FUZZY_MERGE_THRESHOLD
) -> tuple[bool, bool]:
    """Should the next prediction line be merged into the running chunk?"""
    if len(pred_lines) == 1:
        return False, False

    cur_pred = " ".join(pred_lines[:-1])
    merged_pred = " ".join(pred_lines)

    cur_dist = _normalized(gt, cur_pred)
    merged_dist = _normalized(gt, merged_pred)
    if merged_dist > cur_dist:
        return False, False

    for line in pred_lines[:-1]:
        dist = _sub_pred_fuzzy_matching(gt, line)
        if dist is False or dist > threshold:
            return False, False

    add_dist = _sub_pred_fuzzy_matching(gt, pred_lines[-1])
    if add_dist is False:
        return False, False

    merged_flag = add_dist < threshold
    continue_flag = len(merged_pred) <= len(gt)
    return merged_flag, continue_flag


def _merge_lists_with_sublists(main_list, sub_lists):
    main_final = list(main_list)
    for sub_list in sub_lists:
        pop_idx = main_final.index(sub_list[0])
        for _ in sub_list:
            main_final.pop(pop_idx)
        main_final.insert(pop_idx, sub_list)
    return main_final


def _get_final_subset(subsets: list[list[int]], costs: list[float]):
    """Resolve overlapping merge candidates, keeping cheapest consistent paths."""
    if not subsets:
        return []

    ordered = sorted(zip(subsets, costs), key=lambda x: x[0][0])
    groups: dict[int, list] = defaultdict(list)
    group_idx = 0
    groups[group_idx].append(ordered[0])

    for item in ordered[1:]:
        overlap = any(
            idx in existing[0]
            for existing in groups[group_idx]
            for idx in item[0]
        )
        if overlap:
            groups[group_idx].append(item)
        else:
            group_idx += 1
            groups[group_idx].append(item)

    final: list[list[int]] = []
    for group in groups.values():
        if len(group) == 1:
            final.append(group[0][0])
            continue
        paths: dict[int, list] = defaultdict(list)
        paths[0].append(group[0])
        for subset in group[1:]:
            placed = False
            for path_items in paths.values():
                duplicate = any(item[0] == subset[0] for item in path_items)
                shares = any(
                    n1 == n2
                    for item in path_items
                    if item[0] != subset[0]
                    for n1 in item[0]
                    for n2 in subset[0]
                )
                if duplicate:
                    best = min(
                        [i for i in path_items if i[0] == subset[0]]
                        + [subset],
                        key=lambda x: x[1],
                    )
                    path_items.remove(
                        next(i for i in path_items if i[0] == subset[0])
                    )
                    path_items.append(best)
                    placed = True
                elif not shares:
                    path_items.append(subset)
                    placed = True
            if not placed:
                paths[len(paths)].append(subset)
        best_path = min(paths.values(), key=lambda p: sum(i[1] for i in p) / len(p))
        final.extend(item[0] for item in best_path)
    return final


def _deal_with_truncated(cost_matrix, gt_lines, pred_lines):
    """Adjacency-search merge of truncated/merged prediction lines."""
    matched_first = np.argwhere(cost_matrix < CERTAIN_MATCH_THRESHOLD)
    masked_gt_idx = {int(i) for i, _ in matched_first}
    masked_pred_idx = {int(j) for _, j in matched_first}
    unmasked_gt_idx = [i for i in range(cost_matrix.shape[0]) if i not in masked_gt_idx]
    unmasked_pred_idx = [j for j in range(cost_matrix.shape[1]) if j not in masked_pred_idx]

    merge_info: dict[int, dict] = {}
    for gt_idx in unmasked_gt_idx:
        check_merge_subsets: list[list[int]] = []
        merged_dists: list[float] = []

        for pred_idx in unmasked_pred_idx:
            step = 1
            merged_pred = [pred_lines[pred_idx]]
            while True:
                if step >= MAX_TRUNCATED_PRED_MERGE:
                    break
                nxt = pred_idx + step
                if nxt in masked_pred_idx or nxt >= len(pred_lines):
                    break
                merged_pred.append(pred_lines[nxt])
                merged_flag, continue_flag = _judge_pred_merge(
                    gt_lines[gt_idx], merged_pred
                )
                if not merged_flag:
                    break
                step += 1
                if not continue_flag:
                    break

            check_merge_subsets.append(list(range(pred_idx, pred_idx + step)))
            merged_line = " ".join(pred_lines[pred_idx : pred_idx + step])
            merged_dists.append(_normalized(gt_lines[gt_idx], merged_line))

        if not merged_dists:
            merge_info[gt_idx] = {"subset": [], "cost": float("inf")}
        else:
            min_idx = merged_dists.index(min(merged_dists))
            merge_info[gt_idx] = {
                "subset": check_merge_subsets[min_idx],
                "cost": merged_dists[min_idx],
            }

    subsets = [m["subset"] for m in merge_info.values() if m["subset"]]
    costs = [m["cost"] for m in merge_info.values() if m["subset"]]
    final_subsets = _get_final_subset(subsets, costs)

    if not final_subsets:
        return cost_matrix, list(range(len(pred_lines)))

    final_pred_idx_list = _merge_lists_with_sublists(
        list(range(len(pred_lines))), final_subsets
    )
    final_pred_lines = [
        " ".join(pred_lines[idx_list[0] : idx_list[-1] + 1])
        if isinstance(idx_list, list)
        else pred_lines[idx_list]
        for idx_list in final_pred_idx_list
    ]
    new_cost_matrix = _cost_matrix(gt_lines, final_pred_lines)
    return new_cost_matrix, final_pred_idx_list


# ── Main entry point ─────────────────────────────────────────────────


def quick_match(
    gt_paragraphs: list[str], pred_paragraphs: list[str]
) -> list[AlignedPair]:
    """Align segmented GT paragraphs against segmented prediction paragraphs.

    Returns one :class:`AlignedPair` per GT paragraph plus one pair per
    prediction paragraph with no GT counterpart. Matched pairs carry the
    normalized edit distance of the aligned texts; unmatched pairs carry
    edit = 1.
    """
    gt_original = list(gt_paragraphs)
    pred_original = list(pred_paragraphs)
    gt_lines = [normalize_text(p) for p in gt_original]
    pred_lines = [normalize_text(p) for p in pred_original]

    if not gt_lines and not pred_lines:
        return []
    if not gt_lines:
        return [
            AlignedPair(
                gt="", pred=pred_original[i], edit=1.0, matched=False,
                pred_indices=(i,),
            )
            for i in range(len(pred_lines))
        ]
    if not pred_lines:
        return [
            AlignedPair(
                gt=gt_original[i], pred="", edit=1.0, matched=False,
                gt_indices=(i,),
            )
            for i in range(len(gt_lines))
        ]
    if len(gt_lines) == 1 and len(pred_lines) == 1:
        edit = _normalized(gt_lines[0], pred_lines[0])
        return [
            AlignedPair(
                gt=gt_original[0],
                pred=pred_original[0],
                edit=edit,
                matched=edit <= UNMATCHED_THRESHOLD,
                gt_indices=(0,),
                pred_indices=(0,),
            )
        ]

    cost_matrix = _cost_matrix(gt_lines, pred_lines)
    new_cost_matrix, final_pred_idx_list = _deal_with_truncated(
        cost_matrix, gt_lines, pred_lines
    )
    row_ind, col_ind = linear_sum_assignment(new_cost_matrix)

    # group GT rows by assigned prediction key
    assigned: dict[tuple[int, ...], dict] = {}
    unmatched_gt: list[int] = []
    consumed_preds: set[int] = set()

    for r, c in zip(row_ind, col_ind):
        edit = new_cost_matrix[r][c]
        pred_key = final_pred_idx_list[c]
        pred_span = pred_key if isinstance(pred_key, list) else [pred_key]
        if edit > UNMATCHED_THRESHOLD:
            unmatched_gt.append(int(r))
            continue
        key = tuple(sorted(pred_span))
        assigned.setdefault(key, {"gt_indices": [], "edit": float(edit)})
        assigned[key]["gt_indices"].append(int(r))
        consumed_preds.update(pred_span)

    pairs: list[AlignedPair] = []
    matched_gt: set[int] = set()
    for pred_key, info in assigned.items():
        gt_idxs = sorted(set(info["gt_indices"]))
        matched_gt.update(gt_idxs)
        merged_gt = "".join(gt_lines[i] for i in gt_idxs)
        pred_text = " ".join(pred_lines[i] for i in pred_key)
        if len(gt_idxs) > 1:
            edit = _normalized(merged_gt, pred_text)
        else:
            edit = info["edit"]
        pairs.append(
            AlignedPair(
                gt=" ".join(gt_original[i] for i in gt_idxs),
                pred=" ".join(pred_original[i] for i in pred_key),
                edit=edit,
                matched=True,
                gt_indices=tuple(gt_idxs),
                pred_indices=tuple(pred_key),
            )
        )

    # residual unmatched GT / prediction lines: pair greedily, edit = 1
    residual_gt = [i for i in range(len(gt_lines)) if i not in matched_gt]
    residual_pred = [i for i in range(len(pred_lines)) if i not in consumed_preds]

    if residual_gt and residual_pred:
        dist = np.array(
            [
                [_normalized(gt_lines[g], pred_lines[p]) for p in residual_pred]
                for g in residual_gt
            ]
        )
        rows, cols = linear_sum_assignment(dist)
        used_gt = set()
        used_pred = set()
        for r, c in zip(rows, cols):
            used_gt.add(residual_gt[r])
            used_pred.add(residual_pred[c])
            pairs.append(
                AlignedPair(
                    gt=gt_original[residual_gt[r]],
                    pred=pred_original[residual_pred[c]],
                    edit=1.0,
                    matched=False,
                    gt_indices=(residual_gt[r],),
                    pred_indices=(residual_pred[c],),
                )
            )
        residual_gt = [g for g in residual_gt if g not in used_gt]
        residual_pred = [p for p in residual_pred if p not in used_pred]

    for g in residual_gt:
        pairs.append(
            AlignedPair(
                gt=gt_original[g], pred="", edit=1.0, matched=False,
                gt_indices=(g,),
            )
        )
    for p in residual_pred:
        pairs.append(
            AlignedPair(
                gt="", pred=pred_original[p], edit=1.0, matched=False,
                pred_indices=(p,),
            )
        )

    return pairs


def format_alignment_summary(alignment: dict) -> str:
    """Human-readable one-liner for reports, e.g. log headers and summaries."""
    return (
        f"{alignment.get('algorithm', MATCH_ALGORITHM_NAME)} "
        f"({alignment.get('version', MATCH_ALGORITHM_VERSION)}): "
        f"{alignment.get('matched_pairs', 0)} matched, "
        f"{alignment.get('unmatched_pairs', 0)} unmatched pairs"
    )
