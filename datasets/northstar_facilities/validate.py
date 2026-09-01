#!/usr/bin/env python3
"""Validate schema, referential integrity, evidence, and reproducibility."""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED = {
    "question",
    "ground_truth",
    "context",
    "metadata",
    "case_id",
    "reference_answer",
    "relevant_source_ids",
    "gold_chunk_ids",
    "evidence",
}


def main() -> int:
    rows = [
        json.loads(line)
        for line in (ROOT / "questions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    errors: list[str] = []
    ids = [row.get("case_id") for row in rows]
    questions = [row.get("question") for row in rows]
    if len(rows) < 300:
        errors.append(f"expected at least 300 cases, found {len(rows)}")
    if len(ids) != len(set(ids)):
        errors.append("case_id values are not unique")
    if len(questions) != len(set(questions)):
        errors.append("question values are not unique")
    for number, row in enumerate(rows, 1):
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"row {number}: missing {sorted(missing)}")
            continue
        if row["ground_truth"] != row["reference_answer"]:
            errors.append(f"row {number}: answer aliases disagree")
        if not str(row["question"]).strip() or not str(row["ground_truth"]).strip():
            errors.append(f"row {number}: question or answer is empty")
        meta = row["metadata"]
        if meta.get("case_id") != row["case_id"] or meta.get("language") != "en":
            errors.append(f"row {number}: invalid metadata identity/language")
        if meta.get("relevant_source_ids") != row["relevant_source_ids"]:
            errors.append(f"row {number}: source IDs disagree")
        if meta.get("gold_chunk_ids") != row["gold_chunk_ids"]:
            errors.append(f"row {number}: chunk IDs disagree")
        expected_contexts = [
            (ROOT / "corpus" / f"{source_id}.md").read_text(encoding="utf-8")
            for source_id in row["relevant_source_ids"]
            if (ROOT / "corpus" / f"{source_id}.md").is_file()
        ]
        actual_contexts = (
            row["context"] if isinstance(row["context"], list) else [row["context"]]
        )
        if actual_contexts != expected_contexts:
            errors.append(
                f"row {number}: context does not exactly match referenced sources"
            )
        for item in row["evidence"]:
            source = ROOT / "corpus" / f"{item['source_id']}.md"
            if not source.is_file():
                errors.append(f"row {number}: missing source {source.name}")
                continue
            lines = source.read_text(encoding="utf-8").splitlines()
            start, end = int(item["line_start"]), int(item["line_end"])
            if (
                start < 1
                or end > len(lines)
                or "\n".join(lines[start - 1 : end]) != item["quote"]
            ):
                errors.append(
                    f"row {number}: evidence span/quote mismatch in {source.name}"
                )
            if item["chunk_id"] not in row["gold_chunk_ids"]:
                errors.append(f"row {number}: evidence chunk is not gold")
        evidence_text = " ".join(str(item["quote"]) for item in row["evidence"])
        for token in str(row["ground_truth"]).replace(";", " ").split():
            if (
                any(ch.isdigit() for ch in token)
                and token.strip(";,.") not in evidence_text
            ):
                errors.append(
                    f"row {number}: numeric answer token {token!r} absent from evidence"
                )
    check = subprocess.run(
        [sys.executable, str(ROOT / "generate.py"), "--check"],
        capture_output=True,
        text=True,
    )
    if check.returncode:
        errors.append(
            check.stderr.strip()
            or check.stdout.strip()
            or "reproducibility check failed"
        )
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    question_bytes = (ROOT / "questions.jsonl").read_bytes()
    if manifest.get("questions_sha256") != hashlib.sha256(question_bytes).hexdigest():
        errors.append("manifest question checksum mismatch")
    if manifest.get("question_count") != len(rows):
        errors.append("manifest question count mismatch")
    expected_sources = set(manifest.get("source_sha256", {}))
    actual_sources = {
        str(path.relative_to(ROOT)) for path in (ROOT / "corpus").glob("*.md")
    }
    if manifest.get("document_count") != len(actual_sources):
        errors.append("manifest document count mismatch")
    if actual_sources != expected_sources:
        missing = sorted(expected_sources - actual_sources)
        extra = sorted(actual_sources - expected_sources)
        errors.append(f"manifest corpus set mismatch: missing={missing}, extra={extra}")
    for relative, expected_digest in manifest.get("source_sha256", {}).items():
        path = ROOT / relative
        if not path.is_file():
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_digest:
            errors.append(f"manifest checksum mismatch: {relative}")
    stats = {
        "status": "fail" if errors else "pass",
        "question_count": len(rows),
        "document_count": len(list((ROOT / "corpus").glob("*.md"))),
        "unique_case_ids": len(set(ids)),
        "unique_questions": len(set(questions)),
        "questions_with_evidence": sum(bool(row.get("evidence")) for row in rows),
        "multi_source_questions": sum(
            len(row.get("relevant_source_ids", [])) > 1 for row in rows
        ),
        "difficulty": dict(
            sorted(
                Counter(
                    row.get("metadata", {}).get("difficulty") for row in rows
                ).items()
            )
        ),
        "question_type": dict(
            sorted(
                Counter(
                    row.get("metadata", {}).get("question_type") for row in rows
                ).items()
            )
        ),
        "errors": errors,
    }
    (ROOT / "validation_report.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(stats, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
