"""Frozen question sets: versioned dev/holdout splits and their identity."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SPLITS = ("dev", "holdout")

# Loader-stamped document IDs. They are re-derived from the row position when
# the frozen file is loaded again (``dataset-doc-{index}``), so storing them
# would only risk a stale value.
_LOADER_METADATA_KEYS = ("doc_id", "gold_doc_id")


@dataclass(frozen=True)
class FrozenQuestionSet:
    dev: list[dict[str, Any]]
    holdout: list[dict[str, Any]]
    manifest: dict[str, Any]


def _category(metadata: Mapping[str, Any]) -> str:
    answerable = metadata.get("answerable")
    if answerable is None:
        return "unlabeled"
    return "answerable" if answerable else "unanswerable"


def _frozen_row(sample: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {
        key: value
        for key, value in (sample.get("metadata") or {}).items()
        if key not in _LOADER_METADATA_KEYS
    }
    metadata["category"] = _category(metadata)
    return {
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "context": sample["context"],
        "metadata": metadata,
    }


def jsonl_lines(rows: Sequence[Mapping[str, Any]]) -> str:
    """Canonical JSONL text of ``rows``; the hashed and written form."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def _split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "size": len(rows),
        "categories": dict(Counter(row["metadata"]["category"] for row in rows)),
        "sha256": hashlib.sha256(jsonl_lines(rows).encode("utf-8")).hexdigest(),
    }


def freeze_question_set(
    samples: Sequence[Mapping[str, Any]],
    *,
    dev_size: int,
    holdout_size: int,
    source: Mapping[str, Any],
) -> FrozenQuestionSet:
    """Split loaded samples into a fixed dev set and a disjoint holdout set.

    ``dev`` is the first ``dev_size`` samples, so a frozen set built from a
    seeded shuffle keeps the questions earlier runs with the same seed and
    sample size already used. ``holdout`` takes the following samples whose
    question text does not occur in ``dev``.
    """
    if len(samples) < dev_size:
        raise ValueError(f"need {dev_size} dev samples, source has {len(samples)}")
    dev = [_frozen_row(sample) for sample in samples[:dev_size]]
    dev_questions = {row["question"] for row in dev}

    holdout: list[dict[str, Any]] = []
    for sample in samples[dev_size:]:
        if len(holdout) == holdout_size:
            break
        if sample["question"] in dev_questions:
            continue
        holdout.append(_frozen_row(sample))
    if len(holdout) < holdout_size:
        raise ValueError(
            f"need {holdout_size} holdout samples disjoint from dev, "
            f"source has {len(holdout)}; load more samples"
        )

    manifest = {
        "source": dict(source),
        "splits": {"dev": _split_manifest(dev), "holdout": _split_manifest(holdout)},
    }
    return FrozenQuestionSet(dev=dev, holdout=holdout, manifest=manifest)


SampleKey = tuple[str, str, int]


def sample_keys(records: Sequence[Mapping[str, Any]]) -> list[SampleKey]:
    """Identity of each record: question, reference answer, occurrence.

    The question text alone is not unique: RAGBench TechQA lists every
    question twice with the same documents but a different reference answer
    (and sometimes a different answerable label). The occurrence number
    keeps exact duplicates apart. Works for frozen rows and result records.
    """
    seen: Counter[tuple[str, str]] = Counter()
    keys = []
    for record in records:
        base = (record["question"], str(record.get("ground_truth") or ""))
        keys.append((*base, seen[base]))
        seen[base] += 1
    return keys


def question_categories(question_set: Sequence[Mapping[str, Any]] | None) -> dict[SampleKey, str]:
    """Map each frozen sample to its category.

    ``answerable`` / ``unanswerable`` / ``unlabeled`` come from the frozen
    row. A question listed several times with *conflicting* answerable labels
    (47 TechQA questions) becomes ``ambiguous`` on every copy: a refusal on it
    is neither clearly right nor clearly wrong, so abstention scoring skips it.
    """
    rows = list(question_set or ())
    recorded = [(row.get("metadata") or {}).get("category", "unlabeled") for row in rows]
    labels: dict[str, set[str]] = {}
    for row, category in zip(rows, recorded):
        if category in ("answerable", "unanswerable"):
            labels.setdefault(row["question"], set()).add(category)
    return {
        key: "ambiguous" if len(labels.get(row["question"], ())) > 1 else category
        for key, row, category in zip(sample_keys(rows), rows, recorded)
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def split_status(root: str | Path, split: str) -> tuple[str, str | None]:
    """Check one split file against the manifest hash.

    Returns ``(status, actual_sha256)`` with status ``ok``, ``MODIFIED``,
    ``MISSING`` (split file absent) or ``NO MANIFEST``.
    """
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return "NO MANIFEST", None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / f"{split}.jsonl"
    if not path.exists():
        return "MISSING", None
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = manifest["splits"][split]["sha256"]
    return ("ok" if actual == expected else "MODIFIED"), actual
