"""Tests for per-question checkpointing so interrupted runs can resume."""

import json
from pathlib import Path

from benchmark.checkpoint import (
    CheckpointStore,
    generation_result_from_record,
)
from benchmark.generation import GenerationResult


def _sample_result() -> GenerationResult:
    return GenerationResult(
        answer="42",
        ttft_seconds=0.12,
        total_seconds=1.5,
        token_count=10,
        tokens_per_second=6.67,
        gpu_usage={"gpu_utilization_pct": 50.0},
        input_tokens=100,
        output_tokens=10,
        total_tokens=110,
        estimated_cost_usd=0.001,
        raw_content="<answer>42</answer>",
        raw_reasoning="thinking...",
        answer_valid=True,
    )


class TestCheckpointStoreRoundtrip:
    def test_saved_entry_roundtrips_to_generation_result(self, tmp_path: Path):
        store = CheckpointStore(tmp_path / "cp.json")
        record = store.build_record(
            index=0,
            question="What is X?",
            contexts=["ctx1", "ctx2"],
            retrieved_metadata=[{"doc_id": "d1"}],
            gold_doc_id="d1",
            sample_metadata={"source": "hotpot"},
            adapter_diagnostics={"connection_seconds": 0.1},
            result=_sample_result(),
        )
        store.save(record)

        loaded = store.load()
        assert 0 in loaded
        entry = loaded[0]
        assert entry["question"] == "What is X?"
        assert entry["contexts"] == ["ctx1", "ctx2"]
        assert entry["retrieved_metadata"] == [{"doc_id": "d1"}]
        assert entry["gold_doc_id"] == "d1"
        assert entry["sample_metadata"] == {"source": "hotpot"}
        assert entry["adapter_diagnostics"] == {"connection_seconds": 0.1}

        result = generation_result_from_record(entry)
        assert result == _sample_result()

    def test_persists_across_store_instances(self, tmp_path: Path):
        path = tmp_path / "cp.json"
        store = CheckpointStore(path)
        store.save(store.build_record(
            index=3, question="q3", contexts=["c"], retrieved_metadata=[],
            gold_doc_id=None, sample_metadata={}, adapter_diagnostics={},
            result=_sample_result(),
        ))

        reloaded = CheckpointStore(path).load()
        assert 3 in reloaded

    def test_load_missing_file_returns_empty(self, tmp_path: Path):
        assert CheckpointStore(tmp_path / "nope.json").load() == {}

    def test_load_corrupt_file_returns_empty(self, tmp_path: Path):
        path = tmp_path / "cp.json"
        path.write_text("{not json")
        assert CheckpointStore(path).load() == {}

    def test_partial_record_raises_on_reconstruction(self, tmp_path: Path):
        store = CheckpointStore(tmp_path / "cp.json")
        record = store.build_record(
            index=0, question="q", contexts=[], retrieved_metadata=[],
            gold_doc_id=None, sample_metadata={}, adapter_diagnostics={},
            result=_sample_result(),
        )
        del record["gen"]["answer"]
        with __import__("pytest").raises(KeyError):
            generation_result_from_record(record)


class TestQuestionMatchGuard:
    def test_matching_question_accepted(self, tmp_path: Path):
        store = CheckpointStore(tmp_path / "cp.json")
        store.save(store.build_record(
            index=0, question="q1", contexts=[], retrieved_metadata=[],
            gold_doc_id=None, sample_metadata={}, adapter_diagnostics={},
            result=_sample_result(),
        ))
        entry = store.entry_for(0, "q1")
        assert entry is not None

    def test_mismatched_question_rejected(self, tmp_path: Path):
        store = CheckpointStore(tmp_path / "cp.json")
        store.save(store.build_record(
            index=0, question="q1", contexts=[], retrieved_metadata=[],
            gold_doc_id=None, sample_metadata={}, adapter_diagnostics={},
            result=_sample_result(),
        ))
        assert store.entry_for(0, "different question") is None
        assert store.entry_for(99, "q1") is None
