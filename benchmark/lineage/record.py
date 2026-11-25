"""Assemble the lineage record of one benchmark run (issue #65)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from benchmark.lineage.dataset import assign_sample_ids, dataset_identity, sample_category
from benchmark.lineage.experiment import parse_experiment_metadata, scorer_identity
from benchmark.lineage.provenance import resolve_system_provenance

LINEAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunLineage:
    experiment: dict[str, str]
    system: dict[str, Any]
    dataset: dict[str, Any]
    scorer: dict[str, Any]
    sample_ids: tuple[str, ...]
    categories: tuple[str | None, ...]

    def as_dict(self) -> dict[str, Any]:
        """Run-level record (per-sample IDs live on the samples themselves)."""
        return {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "experiment": dict(self.experiment),
            "system": dict(self.system),
            "dataset": dict(self.dataset),
            "scorer": dict(self.scorer),
        }


def _evaluated_corpus(data: Sequence[Mapping[str, Any]], corpus: Sequence[Any] | None):
    """Shared corpus if the dataset has one, else the per-sample contexts."""
    if corpus is not None:
        return corpus
    contexts = [sample.get("context") for sample in data if sample.get("context")]
    return [{"context": c} for c in contexts] or None


def build_run_lineage(
    adapter: Any,
    config: Any,
    data: Sequence[Mapping[str, Any]],
    corpus: Sequence[Any] | None = None,
) -> RunLineage:
    sample_ids = assign_sample_ids(data)
    return RunLineage(
        experiment=parse_experiment_metadata(getattr(config, "experiment_metadata_json", None)),
        system=resolve_system_provenance(adapter, config).as_dict(),
        dataset=dataset_identity(
            config, data, _evaluated_corpus(data, corpus), sample_ids=sample_ids
        ),
        scorer=scorer_identity(config),
        sample_ids=tuple(sample_ids),
        categories=tuple(sample_category(sample) for sample in data),
    )
