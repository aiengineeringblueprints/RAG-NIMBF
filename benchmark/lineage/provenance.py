"""Version record of the RAG system under test (issue #65).

The framework's own git state already lands in the reproducibility manifest.
This record is about the *system* a run measured — the built-in pipeline, a
Blueprint checkout, a container image, a hosted deployment — so two runs can
be compared by what actually changed.

Adapters may expose ``system_provenance(config)`` returning a
``SystemProvenance`` (or a mapping in its shape). When they cannot capture
anything, a manifest may declare the version (``system_provenance_json``);
that record is marked ``user_supplied``. Nothing here stores secrets: remote
URLs lose their credentials and uncommitted changes are reduced to a hash.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from benchmark.reproducibility import _run_git

log = logging.getLogger(__name__)

# Root of the benchmark framework checkout (provenance of the built-in pipeline).
FRAMEWORK_ROOT = Path(__file__).resolve().parents[2]

PROVENANCE_STATUSES = ("captured", "user_supplied", "not_available", "dirty")
VERSION_SOURCES = ("git", "container", "deployment", "user", "not_available")
SUPPLIED_FIELDS = ("system_id", "system_name", "version_source", "version", "branch", "dirty")

_URL_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")


@dataclass(frozen=True)
class SystemProvenance:
    system_id: str
    system_name: str | None = None
    version_source: str = "not_available"
    version: str | None = None
    branch: str | None = None
    remote: str | None = None
    dirty: bool | None = None
    # SHA-256 over the uncommitted change set; the diff itself never leaves
    # the machine (it may contain credentials or unreleased code).
    diff_hash: str | None = None
    status: str = "not_available"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "SystemProvenance":
        """Rebuild a record; a missing record (older runs) is ``not_available``."""
        if not data:
            return cls(system_id="unknown")
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in data.items() if k in known}
        values["system_id"] = values.get("system_id") or "unknown"
        return cls(**values)


def _strip_credentials(url: str | None) -> str | None:
    return _URL_CREDENTIALS.sub("", url) if url else url


def capture_git_provenance(
    path: str | Path, *, system_id: str, system_name: str | None = None
) -> SystemProvenance:
    """Version of the git checkout at ``path``; ``not_available`` outside git."""
    root = Path(path)
    commit = _run_git(["rev-parse", "HEAD"], cwd=root) if root.is_dir() else None
    if not commit:
        return SystemProvenance(system_id=system_id, system_name=system_name)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    status = _run_git(["status", "--porcelain"], cwd=root) or ""
    dirty = bool(status)
    diff_hash = None
    if dirty:
        diff = _run_git(["diff", "HEAD", "--binary"], cwd=root) or ""
        diff_hash = hashlib.sha256(f"{status}\n{diff}".encode("utf-8")).hexdigest()
    return SystemProvenance(
        system_id=system_id,
        system_name=system_name,
        version_source="git",
        version=commit,
        branch=None if branch in (None, "HEAD") else branch,
        remote=_strip_credentials(_run_git(["remote", "get-url", "origin"], cwd=root)),
        dirty=dirty,
        diff_hash=diff_hash,
        status="dirty" if dirty else "captured",
    )


def framework_provenance() -> SystemProvenance:
    """Provenance of the framework's built-in RAG pipeline."""
    return capture_git_provenance(
        FRAMEWORK_ROOT,
        system_id="benchmark_internal_pipeline",
        system_name="Benchmark built-in RAG pipeline",
    )


def parse_supplied_provenance(value: str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Validate a manifest-declared provenance object; ``None`` when unset."""
    if value is None or value == "":
        return None
    data = json.loads(value) if isinstance(value, str) else dict(value)
    if not isinstance(data, dict):
        raise ValueError("system_provenance_json must contain a JSON object")
    unknown = sorted(set(data) - set(SUPPLIED_FIELDS))
    if unknown:
        raise ValueError(
            f"Unknown system_provenance field(s): {', '.join(unknown)}; "
            f"allowed: {', '.join(SUPPLIED_FIELDS)}"
        )
    source = data.get("version_source", "user")
    if source not in VERSION_SOURCES:
        raise ValueError(
            f"system_provenance.version_source must be one of {', '.join(VERSION_SOURCES)}"
        )
    return data


def _supplied(config: Any, default_id: str) -> SystemProvenance | None:
    data = parse_supplied_provenance(getattr(config, "system_provenance_json", None))
    if data is None:
        return None
    return SystemProvenance(
        system_id=str(data.get("system_id") or default_id),
        system_name=data.get("system_name"),
        version_source=data.get("version_source", "user"),
        version=None if data.get("version") is None else str(data["version"]),
        branch=data.get("branch"),
        dirty=data.get("dirty"),
        status="user_supplied",
    )


def _coerce(value: Any, default_id: str) -> SystemProvenance | None:
    if value is None or isinstance(value, SystemProvenance):
        return value
    if isinstance(value, Mapping):
        record = SystemProvenance.from_dict({"system_id": default_id, **value})
        return record
    raise TypeError(f"system_provenance() returned {type(value).__name__}")


def resolve_system_provenance(adapter: Any, config: Any) -> SystemProvenance:
    """Provenance for one run: adapter capture, else manifest, else unknown."""
    default_id = str(
        getattr(adapter, "name", None) or getattr(config, "rag_system_adapter", None) or "unknown"
    )
    captured: SystemProvenance | None = None
    hook = getattr(adapter, "system_provenance", None)
    if callable(hook):
        try:
            captured = _coerce(hook(config), default_id)
        except Exception as exc:  # provenance must never fail a benchmark run
            log.warning("Adapter %s could not capture provenance: %s", default_id, exc)
    if captured is not None and captured.status != "not_available":
        return captured
    supplied = _supplied(config, captured.system_id if captured else default_id)
    if supplied is not None:
        return supplied
    return captured or SystemProvenance(system_id=default_id)
