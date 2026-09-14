"""Per-question checkpoint persistence for resumable benchmark runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from benchmark.generation import GenerationResult

logger = logging.getLogger(__name__)


class CheckpointStore:
    """Append-style JSON checkpoint keyed by question index."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._entries: dict[int, dict[str, Any]] | None = None

    def load(self) -> dict[int, dict[str, Any]]:
        if self._entries is not None:
            return self._entries
        if not self.path.exists():
            self._entries = {}
            return self._entries
        try:
            raw = json.loads(self.path.read_text())
            self._entries = {int(k): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("Ignoring corrupt checkpoint %s: %s", self.path, exc)
            self._entries = {}
        return self._entries

    def save(self, record: dict[str, Any]) -> None:
        entries = self.load()
        entries[int(record["index"])] = record
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    def entry_for(self, index: int, question: str) -> dict[str, Any] | None:
        entry = self.load().get(int(index))
        if entry is not None and entry.get("question") == question:
            return entry
        return None

    @staticmethod
    def build_record(
        *,
        index: int,
        question: str,
        contexts: list[str],
        retrieved_metadata: list[dict],
        gold_doc_id: str | None,
        sample_metadata: dict,
        adapter_diagnostics: dict,
        result: GenerationResult,
    ) -> dict[str, Any]:
        return {
            "index": index,
            "question": question,
            "contexts": contexts,
            "retrieved_metadata": retrieved_metadata,
            "gold_doc_id": gold_doc_id,
            "sample_metadata": sample_metadata,
            "adapter_diagnostics": adapter_diagnostics,
            "gen": {
                "answer": result.answer,
                "ttft_seconds": result.ttft_seconds,
                "total_seconds": result.total_seconds,
                "token_count": result.token_count,
                "tokens_per_second": result.tokens_per_second,
                "gpu_usage": result.gpu_usage,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "total_tokens": result.total_tokens,
                "estimated_cost_usd": result.estimated_cost_usd,
                "raw_content": result.raw_content,
                "raw_reasoning": result.raw_reasoning,
                "answer_valid": result.answer_valid,
            },
        }


def generation_result_from_record(record: dict[str, Any]) -> GenerationResult:
    gen = record["gen"]
    return GenerationResult(
        answer=gen["answer"],
        ttft_seconds=gen["ttft_seconds"],
        total_seconds=gen["total_seconds"],
        token_count=gen["token_count"],
        tokens_per_second=gen["tokens_per_second"],
        gpu_usage=gen["gpu_usage"],
        input_tokens=gen["input_tokens"],
        output_tokens=gen["output_tokens"],
        total_tokens=gen["total_tokens"],
        estimated_cost_usd=gen["estimated_cost_usd"],
        raw_content=gen["raw_content"],
        raw_reasoning=gen["raw_reasoning"],
        answer_valid=gen["answer_valid"],
    )
