"""Batch synthesis CLI / API for dynamic ground-truth generation.

Two entry points:

* Programmatic: :func:`synthesize_jsonl` reads input JSONL of chunks and
  writes output JSONL of updates, returning a small summary dict.
* CLI: ``python -m benchmark.dynamic_groundtruth.batch <input> <output>``.

JSONL input schema (one record per line)::

    {"chunk_id": "...", "text": "..."}      # or
    {"id": "...", "content": "..."}         # or
    "raw text of chunk"                     # chunk_id auto-assigned

JSONL output schema (per spec §3.2)::

    {"chunk_id": "...", "original": "...", "updated": "...",
     "masked_token_original": "1995", "masked_token_new": "2003",
     "question": "When was X founded?", "answer": "2003"}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator

from .synthesizer import (
    DynamicGroundTruthSynthesizer,
    SynthesisReport,
    UpdateWithGroundTruth,
    update_to_json_dict,
)

logger = logging.getLogger(__name__)


# ── Streaming JSONL I/O ───────────────────────────────────────────────

def _iter_jsonl(path: Path) -> Iterator[dict | str]:
    """Yield records from a JSONL file.  Bare strings are also accepted."""
    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s:%d — %s", path, line_no, exc)


def _write_jsonl(path: Path, updates: Iterable[UpdateWithGroundTruth]) -> int:
    """Write updates to *path* as JSONL.  Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for upd in updates:
            fh.write(json.dumps(update_to_json_dict(upd), ensure_ascii=False))
            fh.write("\n")
            count += 1
    return count


# ── Public API ────────────────────────────────────────────────────────

def synthesize_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    device: int = -1,
    dedup_questions: bool = True,
    synthesizer: DynamicGroundTruthSynthesizer | None = None,
) -> dict:
    """Read input JSONL → synthesise → write output JSONL.

    Returns a small summary dict::

        {"input_count": ..., "output_count": ..., "skipped_count": ...,
         "success_rate": ...}
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    chunks = list(_iter_jsonl(input_path))
    synth = synthesizer or DynamicGroundTruthSynthesizer(
        device=device,
        dedup_questions=dedup_questions,
    )
    report: SynthesisReport = synth.synthesize_batch(chunks)

    written = _write_jsonl(output_path, report.updates)

    summary = {
        "input_count": len(chunks),
        "output_count": written,
        "skipped_count": len(report.skipped),
        "success_rate": report.success_rate,
    }
    logger.info(
        "Synthesised %d/%d chunks (%.1f%%); %d skipped → %s",
        written, len(chunks), 100.0 * summary["success_rate"],
        len(report.skipped), output_path,
    )
    return summary


# ── CLI ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmark.dynamic_groundtruth.batch",
        description="Synthesise dynamic ground-truth QA pairs for RAG updates.",
    )
    p.add_argument("input", type=Path, help="Input JSONL of chunks.")
    p.add_argument("output", type=Path, help="Output JSONL of updates.")
    p.add_argument(
        "--device", type=int, default=-1,
        help="Device: -1 (CPU, default), 0 (first GPU), ...",
    )
    p.add_argument(
        "--no-dedup", dest="dedup", action="store_false",
        help="Disable question deduplication.",
    )
    p.add_argument(
        "--report-path", type=Path, default=None,
        help="Optional path to write a JSON skip report.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.  Returns process exit code."""
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.input.exists():
        print(f"error: input file not found: {args.input}", file=sys.stderr)
        return 2

    synth = DynamicGroundTruthSynthesizer(
        device=args.device,
        dedup_questions=args.dedup,
    )

    try:
        chunks = list(_iter_jsonl(args.input))
    except OSError as exc:
        print(f"error: cannot read input: {exc}", file=sys.stderr)
        return 2

    report = synth.synthesize_batch(chunks)
    written = _write_jsonl(args.output, report.updates)

    print(
        f"input={len(chunks)} written={written} skipped={len(report.skipped)} "
        f"success_rate={report.success_rate:.2%} → {args.output}"
    )

    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        with args.report_path.open("w", encoding="utf-8") as fh:
            for skip in report.skipped:
                fh.write(json.dumps({
                    "chunk_id": skip.chunk_id,
                    "reason": skip.reason,
                    "detail": skip.detail,
                }, ensure_ascii=False))
                fh.write("\n")

    return 0 if written > 0 else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
