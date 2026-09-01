#!/usr/bin/env python3
"""Generate the deterministic Northstar Facilities RAG benchmark.

This is a synthetic integration/regression corpus, not a claim about real
facilities.  Gold answers and evidence are rendered from the same immutable
records, so every answer is directly auditable against the generated sources.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORPUS = ROOT / "corpus"
QUESTIONS = ROOT / "questions.jsonl"
MANIFEST = ROOT / "manifest.json"
SEED = "northstar-facilities-v1"

REGIONS = (
    "Alder Coast",
    "Brindle Plain",
    "Cobalt Ridge",
    "Dawn Valley",
    "Ember Isle",
    "Frost Basin",
)
SENSORS = ("Aurora-7", "Beacon-X2", "Cirrus-4", "DeltaSense-9", "EchoGrid-3")
PROTOCOLS = ("Quartz Link", "Nimbus Relay", "Harbor Sync", "Pine Mesh")
POWER = (
    "lithium battery bank",
    "hydrogen fuel cell",
    "flywheel array",
    "thermal salt reserve",
)
STEWARDS = ("Atlas Unit", "Birch Team", "Cedar Group", "Dune Crew", "Elm Office")
WINDOWS = (
    "Monday 02:00-04:00 UTC",
    "Tuesday 05:00-07:00 UTC",
    "Wednesday 08:00-10:00 UTC",
    "Thursday 11:00-13:00 UTC",
    "Friday 14:00-16:00 UTC",
)

SECTION_NOTES = {
    "identity": (
        "Catalog identifiers are used on work orders, inspection forms, and "
        "handover records. Regional labels support routing and reporting, while "
        "the registry code remains the authoritative join key when names are "
        "abbreviated. Operators verify both fields before merging records from "
        "separate maintenance systems."
    ),
    "operations": (
        "The operations team reviews output limits during the morning handover. "
        "Temporary derating events are recorded separately from the catalog "
        "rating, so short-lived dispatch changes do not overwrite the baseline. "
        "Commissioning records are retained as historical attributes rather than "
        "being inferred from later refurbishment dates."
    ),
    "resilience": (
        "Emergency readiness is checked with a staged transfer exercise and a "
        "documented restoration sequence. The exercise records start latency, "
        "operator acknowledgement, and recovery observations. Test observations "
        "belong in the incident log; they do not replace the designated backup "
        "technology in the facility register."
    ),
    "instrumentation": (
        "Sensor work follows a controlled procedure with pre-check, reference "
        "comparison, adjustment, and sign-off. A missed interval creates a "
        "maintenance exception for review. Replacement devices inherit neither "
        "an old serial number nor an old certificate, although they follow the "
        "same documented calibration policy."
    ),
    "governance": (
        "Stewardship identifies the group accountable for approving configuration "
        "changes and resolving ownership questions. The telemetry protocol names "
        "the logical exchange convention, not a network address or credential. "
        "Access keys are deliberately excluded from this fictional catalog and "
        "must never be copied into benchmark fixtures."
    ),
    "data-policy": (
        "Retention and upload cadence are independent controls. Retention governs "
        "how long accepted telemetry remains available, while cadence describes "
        "the normal transfer schedule. Backfill after an outage is recorded as an "
        "exception and does not change either catalog value. Audits compare both "
        "controls against the approved record."
    ),
    "maintenance": (
        "Planners use the recurring window to coordinate technicians and notify "
        "dependent teams. Emergency work can occur outside that window, but it is "
        "tracked as an exception with its own approval trail. Times in this "
        "catalog use UTC to avoid seasonal ambiguity across operating regions and "
        "remote support teams."
    ),
}

SUPPLEMENTARY_NOTES = (
    "Quality reviewers sample closed work orders each month. They compare the "
    "facility code, event timestamp, reviewer identity, and closure rationale, "
    "then record discrepancies without altering the underlying catalog facts. "
    "This separation keeps operational evidence auditable during later reviews.",
    "Training exercises use synthetic event identifiers and do not connect to "
    "production control channels. Participants practice escalation, handover, and "
    "recovery documentation. Exercise results may improve procedures, but they do "
    "not silently modify ownership, retention, capacity, or maintenance records.",
    "Change proposals receive a scope check, peer review, and scheduled approval. "
    "The reviewer confirms that narrative notes agree with structured fields and "
    "that no credential or personal data entered the record. Rejected proposals "
    "remain visible in the audit trail with a concise reason.",
)


def record(index: int) -> dict[str, object]:
    """Return one deterministic fictional facility record."""
    code = f"NF-{index + 1:03d}"
    return {
        "code": code,
        "name": f"Northstar Station {index + 1:03d}",
        "region": REGIONS[index % len(REGIONS)],
        "commissioned": 1998 + (index * 7) % 27,
        "capacity_mw": 40 + (index * 13) % 161,
        "backup": POWER[(index * 3) % len(POWER)],
        "sensor": SENSORS[(index * 2) % len(SENSORS)],
        "calibration_days": 14 + (index * 11) % 77,
        "steward": STEWARDS[(index * 4) % len(STEWARDS)],
        "protocol": PROTOCOLS[(index * 3) % len(PROTOCOLS)],
        "retention_months": 12 + (index * 5) % 49,
        "upload_minutes": 5 + (index * 7) % 56,
        "window": WINDOWS[(index * 2) % len(WINDOWS)],
    }


def render_source(r: dict[str, object]) -> tuple[str, dict[str, tuple[int, str]]]:
    """Render a source document and its stable line-level evidence map."""
    fields = [
        (
            "identity",
            f"Facility {r['name']} has registry code {r['code']} and operates in {r['region']}.",
        ),
        (
            "operations",
            f"It was commissioned in {r['commissioned']} and has a rated capacity of {r['capacity_mw']} MW.",
        ),
        ("resilience", f"Its emergency backup system is a {r['backup']}."),
        (
            "instrumentation",
            f"The primary sensor is {r['sensor']}; calibration is required every {r['calibration_days']} days.",
        ),
        (
            "governance",
            f"The responsible steward is {r['steward']}, and telemetry uses the {r['protocol']} protocol.",
        ),
        (
            "data-policy",
            f"Telemetry is retained for {r['retention_months']} months and uploaded every {r['upload_minutes']} minutes.",
        ),
        ("maintenance", f"The scheduled maintenance window is {r['window']}."),
    ]
    lines = [
        f"# {r['name']}",
        "",
        "This record describes a fictional facility in the Northstar test catalog.",
        "",
    ]
    evidence: dict[str, tuple[int, str]] = {}
    for key, sentence in fields:
        lines.extend((f"## {key.replace('-', ' ').title()}", sentence))
        evidence[key] = (len(lines), sentence)
        lines.extend(("", SECTION_NOTES[key], ""))

    # Vary source length so chunk-size sweeps exercise different boundary
    # counts instead of producing one tiny chunk for every document.
    supplement_count = 1 + (int(str(r["code"]).split("-")[-1]) % 3)
    lines.append("## Supplementary Procedures")
    for note_index in range(supplement_count):
        lines.extend(
            (
                f"### Procedure {note_index + 1}",
                SUPPLEMENTARY_NOTES[note_index],
                "",
            )
        )
    return "\n".join(lines).rstrip() + "\n", evidence


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def case(
    *,
    case_id: str,
    question: str,
    answer: str,
    contexts: list[str],
    sources: list[str],
    chunks: list[str],
    evidence: list[dict[str, object]],
    topic: str,
    difficulty: str,
    question_type: str,
) -> dict[str, object]:
    metadata = {
        "case_id": case_id,
        "source_dataset": "northstar-facilities-v1",
        "gold_doc_id": sources[0] if len(sources) == 1 else None,
        "gold_doc_ids": sources,
        "gold_chunk_ids": chunks,
        "relevant_source_ids": sources,
        "evidence": evidence,
        "topic": topic,
        "difficulty": difficulty,
        "question_type": question_type,
        "language": "en",
        "synthetic": True,
    }
    return {
        "question": question,
        "ground_truth": answer,
        "context": contexts[0] if len(contexts) == 1 else contexts,
        "metadata": metadata,
        "case_id": case_id,
        "reference_answer": answer,
        "relevant_source_ids": sources,
        "gold_chunk_ids": chunks,
        "evidence": evidence,
    }


def evidence_item(
    source_id: str, section: str, entry: tuple[int, str]
) -> dict[str, object]:
    line, quote = entry
    return {
        "source_id": source_id,
        "chunk_id": f"{source_id}#{section}",
        "section": section,
        "line_start": line,
        "line_end": line,
        "quote": quote,
    }


def build() -> tuple[list[dict[str, object]], dict[str, str]]:
    records = [record(i) for i in range(60)]
    rendered: dict[str, tuple[str, dict[str, tuple[int, str]]]] = {}
    checksums: dict[str, str] = {}
    for r in records:
        source_id = str(r["code"])
        text, evidence = render_source(r)
        rendered[source_id] = (text, evidence)
        checksums[f"corpus/{source_id}.md"] = sha256(text)

    cases: list[dict[str, object]] = []
    for i, r in enumerate(records):
        sid = str(r["code"])
        text, ev = rendered[sid]
        specs = [
            (
                "capacity",
                f"What is the rated capacity of {r['name']}?",
                f"{r['capacity_mw']} MW",
                "operations",
                "operations",
                "easy",
                "factoid",
            ),
            (
                "calibration",
                f"How often must the primary sensor at {r['code']} be calibrated?",
                f"Every {r['calibration_days']} days",
                "instrumentation",
                "instrumentation",
                "easy",
                "paraphrase",
            ),
            (
                "backup",
                f"Which emergency power technology protects {r['name']}?",
                str(r["backup"]),
                "resilience",
                "resilience",
                "easy",
                "paraphrase",
            ),
            (
                "governance",
                f"Who stewards {r['code']}, and which telemetry protocol does it use?",
                f"{r['steward']}; {r['protocol']}",
                "governance",
                "governance",
                "medium",
                "multi-fact",
            ),
            (
                "data-policy",
                f"State the retention period and upload interval for {r['name']}.",
                f"{r['retention_months']} months; every {r['upload_minutes']} minutes",
                "data-policy",
                "data-policy",
                "medium",
                "multi-fact",
            ),
        ]
        for suffix, question, answer, section, topic, difficulty, qtype in specs:
            item = evidence_item(sid, section, ev[section])
            cases.append(
                case(
                    case_id=f"nsf-{i + 1:03d}-{suffix}",
                    question=question,
                    answer=answer,
                    contexts=[text],
                    sources=[sid],
                    chunks=[str(item["chunk_id"])],
                    evidence=[item],
                    topic=topic,
                    difficulty=difficulty,
                    question_type=qtype,
                )
            )

        other = records[(i + 1) % len(records)]
        oid = str(other["code"])
        other_text, other_ev = rendered[oid]
        first_evidence = evidence_item(sid, "operations", ev["operations"])
        second_evidence = evidence_item(oid, "maintenance", other_ev["maintenance"])
        cases.append(
            case(
                case_id=f"nsf-{i + 1:03d}-cross-station",
                question=(
                    f"What is the commissioned year of {r['name']}, and what is the "
                    f"maintenance window of {other['name']}?"
                ),
                answer=f"{r['commissioned']}; {other['window']}",
                contexts=[text, other_text],
                sources=[sid, oid],
                chunks=[
                    str(first_evidence["chunk_id"]),
                    str(second_evidence["chunk_id"]),
                ],
                evidence=[first_evidence, second_evidence],
                topic="cross-facility",
                difficulty="hard",
                question_type="multi-hop",
            )
        )
    return cases, checksums


def write_dataset() -> None:
    cases, checksums = build()
    CORPUS.mkdir(parents=True, exist_ok=True)
    for i in range(60):
        r = record(i)
        text, _ = render_source(r)
        (CORPUS / f"{r['code']}.md").write_text(text, encoding="utf-8", newline="\n")
    QUESTIONS.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in cases
        ),
        encoding="utf-8",
        newline="\n",
    )
    distribution = {
        key: dict(sorted(Counter(str(row["metadata"][key]) for row in cases).items()))
        for key in ("difficulty", "question_type", "topic", "language")
    }
    manifest = {
        "dataset_id": "northstar-facilities-v1",
        "generator_seed": SEED,
        "license": "CC0-1.0",
        "synthetic": True,
        "intended_use": "RAG adapter integration, retrieval regression, and pipeline smoke testing",
        "limitations": [
            "Template-generated questions are less linguistically diverse than human-authored questions.",
            "The fictional corpus must not be used to measure real-world domain knowledge.",
            "Gold chunk IDs name semantic sections; provider-generated chunk IDs may differ after ingestion.",
        ],
        "document_count": 60,
        "question_count": len(cases),
        "distribution": distribution,
        "source_sha256": checksums,
        "questions_sha256": sha256(
            "".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in cases
            )
        ),
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true", help="fail if generated files differ"
    )
    args = parser.parse_args()
    if args.check:
        expected, checksums = build()
        expected_questions = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in expected
        )
        if (
            not QUESTIONS.exists()
            or QUESTIONS.read_text(encoding="utf-8") != expected_questions
        ):
            raise SystemExit(
                "questions.jsonl is missing or not reproducible; run generate.py"
            )
        for relative, digest in checksums.items():
            path = ROOT / relative
            if not path.exists() or sha256(path.read_text(encoding="utf-8")) != digest:
                raise SystemExit(f"source mismatch: {relative}")
        print(
            f"reproducibility check passed: {len(expected)} questions, {len(checksums)} sources"
        )
    else:
        write_dataset()
        print(
            f"generated {QUESTIONS} and {len(list(CORPUS.glob('*.md')))} source documents"
        )
