"""Workload generator: turn a config + corpus + question pool into a
deterministic stream of operations.

The generator is **open-loop**: it pre-computes arrival times using a
Poisson process derived from ``target_qps`` and stamps each op with the
offset at which the runner should release it. This means observed
throughput under load is allowed to fall below the offered rate — the
property RAGPerf §3.2 measures.

Distribution and op-mix decisions are driven by a seeded ``random.Random``
so a given workload is reproducible across runs.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from benchmark.workload.distributions import uniform_sample, zipfian_sample
from benchmark.workload.operations import (
    InsertOp,
    Operation,
    QueryOp,
    RemoveOp,
    UpdateOp,
)


@dataclass(frozen=True)
class WorkloadConfig:
    """User-facing workload knobs.

    The four ``op_mix_*`` values are renormalised at generation time so
    callers do not have to ensure they sum to exactly 1.0.
    """

    total_ops: int = 500
    target_qps: float = 5.0
    concurrency: int = 8
    distribution: str = "uniform"  # uniform | zipfian
    zipf_theta: float = 0.8
    op_mix_query: float = 0.7
    op_mix_insert: float = 0.15
    op_mix_update: float = 0.1
    op_mix_remove: float = 0.05
    seed: int = 42

    def normalised_mix(self) -> dict[str, float]:
        raw = {
            "query": max(0.0, self.op_mix_query),
            "insert": max(0.0, self.op_mix_insert),
            "update": max(0.0, self.op_mix_update),
            "remove": max(0.0, self.op_mix_remove),
        }
        total = sum(raw.values())
        if total <= 0:
            raise ValueError("workload op_mix must sum to a positive value")
        return {key: value / total for key, value in raw.items()}

    def validate(self) -> "WorkloadConfig":
        if self.total_ops <= 0:
            raise ValueError("workload_total_ops must be positive")
        if self.target_qps <= 0:
            raise ValueError("workload_target_qps must be positive")
        if self.concurrency <= 0:
            raise ValueError("workload_concurrency must be positive")
        if self.distribution not in ("uniform", "zipfian"):
            raise ValueError(
                "workload_distribution must be 'uniform' or 'zipfian'"
            )
        if self.zipf_theta <= 0:
            raise ValueError("workload_zipf_theta must be positive")
        self.normalised_mix()  # raises on all-zero mix
        return self


@dataclass
class WorkloadGenerator:
    """Yield a deterministic stream of operations.

    Parameters
    ----------
    config:
        Knobs for mix, rate, and arrival pattern.
    questions:
        Pool of question dicts (``{"question": ..., "ground_truth": ...}``).
        Sampled for Query ops.
    corpus:
        Pool of corpus dicts (``{"context": ..., "metadata": {"doc_id": ...}}``).
        ``doc_id`` is sampled for Update/Remove ops.
    insert_pool:
        Optional list of ``{"text": ..., "doc_id": ...}`` dicts to draw
        fresh Insert content from. If empty (default), the generator
        synthesises short placeholder documents so the workload can run
        even without a curated insert pool.
    rng:
        Pre-seeded RNG; useful for tests. A fresh RNG seeded from
        ``config.seed`` is created when this is None.
    """

    config: WorkloadConfig
    questions: Sequence[dict]
    corpus: Sequence[dict]
    insert_pool: Sequence[dict] = field(default_factory=list)
    rng: random.Random | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        if not self.questions:
            raise ValueError("WorkloadGenerator needs at least one question")
        if not self.corpus:
            raise ValueError("WorkloadGenerator needs at least one corpus doc")
        if self.rng is None:
            self.rng = random.Random(self.config.seed)

    # ------------------------------------------------------------------ #
    # Sampling helpers
    # ------------------------------------------------------------------ #

    def _doc_ids(self) -> list[str]:
        ids: list[str] = []
        for doc in self.corpus:
            metadata = doc.get("metadata") or {}
            doc_id = metadata.get("doc_id")
            if doc_id is None:
                # Fall back to a positional ID so malformed corpora still run.
                doc_id = f"doc_{len(ids)}"
            ids.append(str(doc_id))
        # Deduplicate while preserving order — corpus may share doc_ids
        # across chunks after deduplication upstream.
        seen: set[str] = set()
        unique: list[str] = []
        for identifier in ids:
            if identifier not in seen:
                seen.add(identifier)
                unique.append(identifier)
        return unique or ["doc_0"]

    def _sample_doc_id(self) -> str:
        ids = self._doc_ids()
        if self.config.distribution == "zipfian":
            return zipfian_sample(ids, self.config.zipf_theta, self.rng)
        return uniform_sample(ids, self.rng)

    def _sample_question(self) -> dict:
        if self.config.distribution == "zipfian":
            keys = [str(i) for i in range(len(self.questions))]
            idx = int(zipfian_sample(keys, self.config.zipf_theta, self.rng))
        else:
            idx = self.rng.randrange(len(self.questions))
        return self.questions[idx]

    def _sample_insert(self, op_id: int) -> tuple[str, str]:
        """Return (doc_id, text) for an Insert op."""
        if self.insert_pool:
            chosen = self.rng.choice(list(self.insert_pool))
            text = str(chosen.get("text") or chosen.get("context") or "")
            doc_id = str(
                chosen.get("doc_id")
                or (chosen.get("metadata") or {}).get("doc_id")
                or f"insert_{op_id}"
            )
            return doc_id, text
        # Synthesised content keeps the workload self-sufficient.
        return f"insert_{op_id}", (
            f"Inserted document {op_id} for workload benchmark; "
            "tangential to the original corpus so gold answers do not apply."
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def _arrivals(self) -> list[float]:
        """Open-loop Poisson arrivals (seconds from start)."""
        # Inter-arrival ~ Exp(target_qps). Cumulative sum gives offsets.
        offsets: list[float] = []
        cumulative = 0.0
        for _ in range(self.config.total_ops):
            draw = self.rng.random()
            # Guard against draw == 0 to skip ln(0).
            gap = -math.log(max(draw, 1e-12)) / self.config.target_qps
            cumulative += gap
            offsets.append(cumulative)
        return offsets

    def _pick_kind(self) -> str:
        mix = self.config.normalised_mix()
        threshold = self.rng.random()
        running = 0.0
        for kind in ("query", "insert", "update", "remove"):
            running += mix[kind]
            if threshold <= running:
                return kind
        return "query"

    def __iter__(self) -> Iterator[Operation]:
        arrivals = self._arrivals()
        for op_id, arrive_at in enumerate(arrivals):
            kind = self._pick_kind()
            if kind == "query":
                sample = self._sample_question()
                yield QueryOp(
                    op_id=op_id,
                    arrive_at=arrive_at,
                    question=str(sample.get("question", "")),
                    ground_truth=str(sample.get("ground_truth", "")),
                    sample=dict(sample),
                )
            elif kind == "insert":
                doc_id, text = self._sample_insert(op_id)
                yield InsertOp(
                    op_id=op_id,
                    arrive_at=arrive_at,
                    doc_id=doc_id,
                    text=text,
                )
            elif kind == "update":
                doc_id = self._sample_doc_id()
                yield UpdateOp(
                    op_id=op_id,
                    arrive_at=arrive_at,
                    doc_id=doc_id,
                    new_text=(
                        f"[workload-update-{op_id}] Updated body for "
                        f"document {doc_id}."
                    ),
                )
            else:  # remove
                doc_id = self._sample_doc_id()
                yield RemoveOp(
                    op_id=op_id,
                    arrive_at=arrive_at,
                    doc_id=doc_id,
                )

    def materialise(self) -> list[Operation]:
        """Eagerly collect every op. Useful for tests and dry-run reports."""
        return list(iter(self))
