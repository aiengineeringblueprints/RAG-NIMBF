"""Loading run results and checking that compared runs are comparable."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Mapping

# Settings that define *how* a run is scored. Runs scored with different
# values are not comparable, whatever the candidate changes.
EVALUATION_FIELDS = (
    "dataset_name",
    "dataset_subset",
    "dataset_split",
    "dataset_path",
    "dataset_sample_size",
    "eval_critic_llm",
    "eval_critic_embedding",
    "custom_metrics_bert_model",
    "custom_retrieval_metrics_mode",
    "ragas_enabled",
    "ragas_metrics",
)

# Machine-local or secret fields: differences are noise, not a change of the
# evaluated system.
_NOISE_SUFFIXES = ("_api_key", "_base_url", "_url", "_path", "_dir", "_token")


def load_run_results(path: str | Path) -> list[dict[str, Any]]:
    """Per-sample records of a single-config run.

    Accepts the config report ``<config>.json`` or its
    ``<config>.samples.json.gz`` sidecar.
    """
    path = Path(path)
    if path.name.endswith(".samples.json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            per_result = json.load(fh)["results"]
    else:
        from benchmark.reporting.exports import load_json_report

        per_result = [r.get("per_sample") for r in load_json_report(path)["results"]]
    filled = [samples for samples in per_result if samples]
    if len(filled) != 1:
        raise ValueError(f"{path}: expected exactly one config with per-sample data, found {len(filled)}")
    return filled[0]


def find_run_config(report_path: str | Path) -> dict[str, Any] | None:
    """Full ``BenchmarkConfig`` of the run that wrote ``report_path``.

    Looks for ``reproducibility/manifest.json`` in the report's directory and
    its parents (reports sit in ``<run>/configs/``). Returns ``None`` when no
    manifest exists or it holds several configs (a matrix run), since the
    report cannot then be mapped to one config.
    """
    for directory in Path(report_path).resolve().parents:
        manifest = directory / "reproducibility" / "manifest.json"
        if manifest.exists():
            configs = json.loads(manifest.read_text(encoding="utf-8")).get("configs") or []
            return dict(configs[0]) if len(configs) == 1 else None
    return None


def _is_noise(field: str) -> bool:
    return field.endswith(_NOISE_SUFFIXES)


def check_run_configs(configs: Mapping[str, Mapping[str, Any] | None]) -> dict[str, Any]:
    """Check that runs share their evaluation setup.

    ``configs`` maps run name (``baseline``, ``best``, ``candidate``) to its
    recorded config or ``None``. Status:

    - ``invalid``: an :data:`EVALUATION_FIELDS` value differs; ``mismatches``
      lists it per run. The comparison must not be trusted.
    - ``unverified``: some run has no recorded config (``missing``).
    - ``ok``: all recorded and consistent.

    ``changed`` lists, per reference run, the system settings the candidate
    differs in — a single-component candidate should show exactly one
    component here.
    """
    known = {name: config for name, config in configs.items() if config is not None}
    missing = [name for name, config in configs.items() if config is None]

    mismatches: dict[str, dict[str, Any]] = {}
    for field in EVALUATION_FIELDS:
        values = {name: config.get(field) for name, config in known.items()}
        if len({json.dumps(v, sort_keys=True, default=str) for v in values.values()}) > 1:
            mismatches[field] = values

    changed: dict[str, dict[str, list[Any]]] = {}
    candidate = known.get("candidate")
    if candidate is not None:
        for ref, config in known.items():
            if ref == "candidate":
                continue
            changed[ref] = {
                field: [config.get(field), candidate.get(field)]
                for field in sorted(config.keys() | candidate.keys())
                if field not in EVALUATION_FIELDS
                and not _is_noise(field)
                and config.get(field) != candidate.get(field)
            }

    status = "invalid" if mismatches else "unverified" if missing else "ok"
    return {"status": status, "mismatches": mismatches, "missing": missing, "changed": changed}
