"""Assemble the lineage record of one benchmark run (issue #65)."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from benchmark.lineage.dataset import SampleIdentity, assign_sample_identities, dataset_identity
from benchmark.lineage.experiment import parse_experiment_metadata, scorer_identity
from benchmark.lineage.provenance import framework_provenance, resolve_system_provenance

LINEAGE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunLineage:
    experiment: dict[str, str]
    system: dict[str, Any]
    dataset: dict[str, Any]
    scorer: dict[str, Any]
    samples: tuple[SampleIdentity, ...]
    framework: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Run-level record (per-sample IDs live on the samples themselves)."""
        return {
            "schema_version": LINEAGE_SCHEMA_VERSION,
            "experiment": dict(self.experiment),
            "system": dict(self.system),
            "dataset": dict(self.dataset),
            "scorer": dict(self.scorer),
            "framework": dict(self.framework or {}),
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
    samples = assign_sample_identities(data)
    return RunLineage(
        experiment=parse_experiment_metadata(getattr(config, "experiment_metadata_json", None)),
        system=resolve_system_provenance(adapter, config).as_dict(),
        dataset=dataset_identity(
            config,
            data,
            _evaluated_corpus(data, corpus),
            sample_ids=[sample.sample_id for sample in samples],
        ),
        scorer=scorer_identity(config),
        samples=tuple(samples),
        framework=framework_provenance().as_dict(),
    )


def write_lineage_record(run_dir: Path, config_name: str, lineage: RunLineage) -> Path:
    """Add the run's lineage and sample IDs to the reproducibility bundle."""
    out = run_dir / "reproducibility" / "lineage" / f"{config_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **lineage.as_dict(),
        "samples": [dataclasses.asdict(sample) for sample in lineage.samples],
    }
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
