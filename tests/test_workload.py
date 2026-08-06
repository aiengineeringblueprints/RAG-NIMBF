"""Tests for benchmark.workload — distributions, generator, and runner."""

from __future__ import annotations

import math
import random
import threading
from collections import Counter
from unittest.mock import MagicMock

import pytest

from benchmark.workload import (
    InsertOp,
    QueryOp,
    RemoveOp,
    UpdateOp,
    WorkloadConfig,
    WorkloadGenerator,
    WorkloadRunner,
    uniform_sample,
    zipfian_sample,
)
from benchmark.workload.distributions import entropy
from benchmark.workload.report import mlflow_metrics, summarize


# ----------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------


class TestDistributions:
    def test_uniform_returns_member_of_set(self):
        rng = random.Random(0)
        ids = ["a", "b", "c", "d"]
        for _ in range(20):
            assert uniform_sample(ids, rng) in ids

    def test_uniform_is_roughly_uniform(self):
        rng = random.Random(42)
        ids = ["a", "b", "c", "d"]
        counts = Counter(uniform_sample(ids, rng) for _ in range(4000))
        # Each id should be near 1000 (+/- generous tolerance).
        for identifier in ids:
            assert 700 < counts[identifier] < 1300, counts

    def test_uniform_empty_raises(self):
        with pytest.raises(ValueError):
            uniform_sample([])

    def test_zipfian_concentrates_on_first_ids(self):
        rng = random.Random(7)
        ids = [f"id_{i}" for i in range(20)]
        counts = Counter(zipfian_sample(ids, theta=1.2, rng=rng) for _ in range(5000))
        # The hottest id should be sampled far more than the median id.
        hottest = counts[ids[0]]
        median = counts[ids[10]]
        assert hottest > median * 4, (hottest, median)

    def test_zipfian_has_lower_entropy_than_uniform(self):
        rng_zipf = random.Random(1)
        rng_uni = random.Random(1)
        ids = [f"id_{i}" for i in range(50)]
        n = 5000
        zipf_counts = Counter(zipfian_sample(ids, theta=1.5, rng=rng_zipf) for _ in range(n))
        uni_counts = Counter(uniform_sample(ids, rng=rng_uni) for _ in range(n))
        zipf_entropy = entropy([c / n for c in zipf_counts.values()])
        uni_entropy = entropy([c / n for c in uni_counts.values()])
        assert zipf_entropy < uni_entropy

    def test_zipfian_empty_raises(self):
        with pytest.raises(ValueError):
            zipfian_sample([])

    def test_zipfian_theta_zero_is_uniform(self):
        rng = random.Random(2)
        ids = [f"id_{i}" for i in range(10)]
        # theta <= 0 should degenerate to uniform choice.
        result = zipfian_sample(ids, theta=0.0, rng=rng)
        assert result in ids


# ----------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------


def _make_questions(n: int = 10) -> list[dict]:
    return [
        {"question": f"Question {i}?", "ground_truth": f"Answer {i}", "context": ""}
        for i in range(n)
    ]


def _make_corpus(n: int = 12) -> list[dict]:
    return [
        {"context": f"Body {i}", "metadata": {"doc_id": f"doc_{i}"}}
        for i in range(n)
    ]


class TestWorkloadConfig:
    def test_normalised_mix_sums_to_one(self):
        cfg = WorkloadConfig(
            op_mix_query=0.7,
            op_mix_insert=0.15,
            op_mix_update=0.1,
            op_mix_remove=0.05,
        )
        mix = cfg.normalised_mix()
        assert math.isclose(sum(mix.values()), 1.0)

    def test_normalised_mix_handles_unnormalised_input(self):
        cfg = WorkloadConfig(op_mix_query=7, op_mix_insert=1, op_mix_update=1, op_mix_remove=1)
        mix = cfg.normalised_mix()
        assert math.isclose(sum(mix.values()), 1.0)
        assert mix["query"] > mix["insert"]

    def test_validate_rejects_zero_mix(self):
        cfg = WorkloadConfig(
            op_mix_query=0, op_mix_insert=0, op_mix_update=0, op_mix_remove=0
        )
        with pytest.raises(ValueError):
            cfg.validate()


class TestWorkloadGenerator:
    def test_emits_expected_total_ops(self):
        cfg = WorkloadConfig(total_ops=50, target_qps=10)
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        ops = gen.materialise()
        assert len(ops) == 50

    def test_arrival_offsets_are_monotonic(self):
        cfg = WorkloadConfig(total_ops=20, target_qps=5)
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        offsets = [op.arrive_at for op in gen]
        assert offsets == sorted(offsets)
        assert offsets[0] >= 0

    def test_arrival_offsets_scale_with_qps(self):
        # Higher QPS should produce smaller inter-arrival gaps on average.
        slow = WorkloadGenerator(
            WorkloadConfig(total_ops=30, target_qps=2, seed=5),
            _make_questions(),
            _make_corpus(),
        )
        fast = WorkloadGenerator(
            WorkloadConfig(total_ops=30, target_qps=20, seed=5),
            _make_questions(),
            _make_corpus(),
        )
        slow_last = max(op.arrive_at for op in slow)
        fast_last = max(op.arrive_at for op in fast)
        assert fast_last < slow_last

    def test_seed_is_reproducible(self):
        cfg = WorkloadConfig(total_ops=30, seed=99)
        g1 = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        g2 = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        ops1 = [(op.kind, op.arrive_at) for op in g1]
        ops2 = [(op.kind, op.arrive_at) for op in g2]
        assert ops1 == ops2

    def test_op_mix_is_roughly_honoured(self):
        cfg = WorkloadConfig(
            total_ops=400,
            target_qps=20,
            op_mix_query=0.7,
            op_mix_insert=0.1,
            op_mix_update=0.1,
            op_mix_remove=0.1,
            seed=1,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        counts = Counter(op.kind for op in gen)
        # Generous tolerance — Poisson sampling + small N.
        assert counts["query"] > counts["insert"]
        assert counts["query"] > counts["update"]
        assert counts["query"] > counts["remove"]
        # No op kind should be entirely missing.
        for kind in ("query", "insert", "update", "remove"):
            assert counts[kind] > 0

    def test_query_ops_carry_question_and_ground_truth(self):
        cfg = WorkloadConfig(total_ops=30, op_mix_query=1.0, op_mix_insert=0, op_mix_update=0, op_mix_remove=0)
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        ops = list(gen)
        assert all(isinstance(op, QueryOp) for op in ops)
        assert all(op.question.startswith("Question ") for op in ops)
        assert all(op.ground_truth.startswith("Answer ") for op in ops)

    def test_update_remove_target_existing_doc_ids(self):
        cfg = WorkloadConfig(
            total_ops=80,
            op_mix_query=0,
            op_mix_insert=0,
            op_mix_update=0.5,
            op_mix_remove=0.5,
            seed=3,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        doc_ids = {f"doc_{i}" for i in range(12)}
        for op in gen:
            assert isinstance(op, (UpdateOp, RemoveOp))
            assert op.doc_id in doc_ids

    def test_insert_uses_pool_when_provided(self):
        cfg = WorkloadConfig(
            total_ops=20,
            op_mix_query=0,
            op_mix_insert=1.0,
            op_mix_update=0,
            op_mix_remove=0,
        )
        insert_pool = [
            {"doc_id": "fresh_1", "text": "first"},
            {"doc_id": "fresh_2", "text": "second"},
        ]
        gen = WorkloadGenerator(
            cfg, _make_questions(), _make_corpus(), insert_pool=insert_pool
        )
        ops = list(gen)
        assert all(isinstance(op, InsertOp) for op in ops)
        assert {op.doc_id for op in ops} == {"fresh_1", "fresh_2"}

    def test_zipf_skew_in_doc_sampling(self):
        cfg = WorkloadConfig(
            total_ops=600,
            op_mix_query=0,
            op_mix_insert=0,
            op_mix_update=1.0,
            op_mix_remove=0,
            distribution="zipfian",
            zipf_theta=1.4,
            seed=10,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus(n=20))
        counts = Counter(op.doc_id for op in gen)
        # Hottest doc should be much more sampled than median doc.
        hottest = counts.most_common(1)[0][1]
        median_value = sorted(counts.values())[len(counts) // 2]
        assert hottest > median_value * 3


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------


class _FakeVectorStore:
    """Records mutations made by the workload adapter helpers."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []  # (doc_id, text)
        self.deleted_ids: list[str] = []
        self.deletes_by_doc_id: list[str] = []
        # Track which doc_ids the helpers *think* exist via collection.get().
        self._doc_ids_by_collection_get: dict[str, list[str]] = {}
        self.lock = threading.Lock()
        # Simulate the LangChain Chroma collection.get(where=...) path by
        # recording calls and returning configured responses.
        self._collection = MagicMock()
        self._collection.get.side_effect = lambda where: {
            "ids": self._doc_ids_by_collection_get.get(where.get("doc_id", ""), [])
        }

    def add_documents(self, documents):
        with self.lock:
            for doc in documents:
                self.added.append((doc.metadata.get("doc_id", "?"), doc.page_content))

    def delete(self, ids):
        with self.lock:
            self.deleted_ids.extend(ids)

    def set_existing_chunks(self, doc_id: str, chunk_ids: list[str]) -> None:
        self._doc_ids_by_collection_get[doc_id] = list(chunk_ids)


def _make_query_callable(answer_template: str = "answer-{i}"):
    """Return a deterministic, slow-ish query callable for runner tests."""
    counter = {"i": 0}
    lock = threading.Lock()

    def _query(_question: str) -> dict:
        with lock:
            counter["i"] += 1
            index = counter["i"]
        return {"answer": answer_template.format(i=index), "contexts": ["ctx"]}

    return _query


class TestWorkloadRunner:
    def test_runs_query_only_workload(self):
        cfg = WorkloadConfig(
            total_ops=20,
            target_qps=50,
            concurrency=4,
            op_mix_query=1.0,
            op_mix_insert=0,
            op_mix_update=0,
            op_mix_remove=0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        store = _FakeVectorStore()
        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=store,
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        assert summary.total_completed == 20
        assert summary.counts_by_kind.get("query") == 20
        assert summary.total_errors == 0
        assert summary.observed_qps > 0
        # Records carry answers and ground truth.
        for record in summary.records:
            assert record.outcome == "ok"
            assert record.answer.startswith("answer-")
            assert record.ground_truth.startswith("Answer ")

    def test_runs_insert_only_workload(self):
        cfg = WorkloadConfig(
            total_ops=10,
            target_qps=20,
            concurrency=2,
            op_mix_query=0,
            op_mix_insert=1.0,
            op_mix_update=0,
            op_mix_remove=0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        store = _FakeVectorStore()
        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=store,
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        assert summary.counts_by_kind.get("insert") == 10
        assert len(store.added) == 10
        assert summary.total_errors == 0

    def test_runs_remove_only_workload(self):
        cfg = WorkloadConfig(
            total_ops=10,
            target_qps=20,
            concurrency=2,
            op_mix_query=0,
            op_mix_insert=0,
            op_mix_update=0,
            op_mix_remove=1.0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        store = _FakeVectorStore()
        # Pre-populate every doc with one chunk so remove finds something.
        for i in range(12):
            store.set_existing_chunks(f"doc_{i}", [f"chunk_doc_{i}_0"])

        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=store,
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        assert summary.counts_by_kind.get("remove") == 10
        # Every remove op should have hit at least one chunk.
        assert summary.total_errors == 0

    def test_runs_update_only_workload_marks_stale_check(self):
        cfg = WorkloadConfig(
            total_ops=8,
            target_qps=20,
            concurrency=2,
            op_mix_query=0,
            op_mix_insert=0,
            op_mix_update=1.0,
            op_mix_remove=0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        store = _FakeVectorStore()
        for i in range(12):
            store.set_existing_chunks(f"doc_{i}", [f"chunk_doc_{i}_0"])

        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=store,
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        assert summary.counts_by_kind.get("update") == 8
        assert summary.total_stale_check == 8
        # Every stale-check should be tagged with the doc id.
        assert len(summary.stale_check_doc_ids) == 8

    def test_runs_mixed_workload(self):
        cfg = WorkloadConfig(
            total_ops=120,
            target_qps=50,
            concurrency=8,
            op_mix_query=0.7,
            op_mix_insert=0.1,
            op_mix_update=0.1,
            op_mix_remove=0.1,
            seed=11,
        )
        gen = WorkloadGenerator(cfg, _make_questions(n=40), _make_corpus(n=40))
        store = _FakeVectorStore()
        for i in range(40):
            store.set_existing_chunks(f"doc_{i}", [f"chunk_doc_{i}_0"])

        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=store,
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        assert summary.total_completed == 120
        # All four op kinds should be present.
        assert set(summary.counts_by_kind.keys()) == {"query", "insert", "update", "remove"}
        # Update ops should flag stale-check; others ok.
        assert summary.total_stale_check == summary.counts_by_kind.get("update", 0)

    def test_query_callable_failure_is_recorded(self):
        def _failing(_q: str) -> dict:
            raise RuntimeError("boom")

        cfg = WorkloadConfig(
            total_ops=4,
            target_qps=20,
            concurrency=2,
            op_mix_query=1.0,
            op_mix_insert=0,
            op_mix_update=0,
            op_mix_remove=0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=_FakeVectorStore(),
            query_callable=_failing,
        )
        summary = runner.run()
        assert summary.total_errors == 4
        for record in summary.records:
            assert record.outcome == "error"
            assert "boom" in (record.error or "")

    def test_summarise_and_mlflow_metrics(self):
        cfg = WorkloadConfig(
            total_ops=10,
            target_qps=50,
            concurrency=2,
            op_mix_query=1.0,
            op_mix_insert=0,
            op_mix_update=0,
            op_mix_remove=0,
        )
        gen = WorkloadGenerator(cfg, _make_questions(), _make_corpus())
        runner = WorkloadRunner(
            config=cfg,
            generator=gen,
            vector_store=_FakeVectorStore(),
            query_callable=_make_query_callable(),
        )
        summary = runner.run()
        agg = summarize(summary)
        assert agg["total_submitted"] == 10
        assert agg["counts_by_kind"]["query"] == 10
        assert "p95" in agg["latency_by_kind"]["query"]
        # MLflow metrics should all carry the workload_ prefix.
        metrics = mlflow_metrics(summary)
        assert all(k.startswith("workload_") for k in metrics)
        assert "workload_observed_qps" in metrics
        assert "workload_query_p95_seconds" in metrics
        assert "workload_error_rate" in metrics
