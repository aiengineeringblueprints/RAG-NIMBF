"""System provenance capture for experiment lineage (issue #65)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmark.lineage import (
    SystemProvenance,
    capture_git_provenance,
    resolve_system_provenance,
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    path = tmp_path / "system"
    path.mkdir()
    _git(path, "init", "-q", "-b", "feature/x")
    _git(path, "config", "user.email", "t@example.com")
    _git(path, "config", "user.name", "t")
    _git(path, "remote", "add", "origin", "https://user:s3cret@example.com/org/system.git")
    (path / "app.py").write_text("print('v1')\n", encoding="utf-8")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "init")
    return path


def _config(**overrides):
    base = {"rag_system_adapter": "http", "system_provenance_json": None}
    base.update(overrides)
    return SimpleNamespace(**base)


def test_clean_checkout_is_captured_with_commit_and_branch(repo):
    provenance = capture_git_provenance(repo, system_id="sys", system_name="System")

    assert provenance.status == "captured"
    assert provenance.version_source == "git"
    assert provenance.version == _git(repo, "rev-parse", "HEAD")
    assert provenance.branch == "feature/x"
    assert provenance.dirty is False
    assert provenance.diff_hash is None


def test_remote_credentials_never_reach_the_record(repo):
    provenance = capture_git_provenance(repo, system_id="sys")

    assert "s3cret" not in json.dumps(provenance.as_dict())
    assert provenance.remote == "https://example.com/org/system.git"


def test_dirty_checkout_records_a_diff_hash_but_not_the_diff(repo):
    (repo / "app.py").write_text("print('secret change')\n", encoding="utf-8")

    provenance = capture_git_provenance(repo, system_id="sys")

    assert provenance.status == "dirty"
    assert provenance.dirty is True
    assert len(provenance.diff_hash) == 64
    assert "secret change" not in json.dumps(provenance.as_dict())


def test_diff_hash_changes_with_the_uncommitted_change(repo):
    (repo / "app.py").write_text("print('a')\n", encoding="utf-8")
    first = capture_git_provenance(repo, system_id="sys").diff_hash
    (repo / "app.py").write_text("print('b')\n", encoding="utf-8")
    second = capture_git_provenance(repo, system_id="sys").diff_hash

    assert first != second


def test_path_outside_git_is_not_available(tmp_path):
    provenance = capture_git_provenance(tmp_path, system_id="sys")

    assert provenance.status == "not_available"
    assert provenance.version is None


def test_adapter_without_provenance_and_no_manifest_is_not_available():
    adapter = SimpleNamespace(name="http")

    provenance = resolve_system_provenance(adapter, _config())

    assert provenance.status == "not_available"
    assert provenance.system_id == "http"


def test_manifest_supplied_provenance_is_marked_user_supplied():
    adapter = SimpleNamespace(name="http")
    supplied = {
        "system_id": "acme-rag",
        "system_name": "ACME RAG",
        "version_source": "container",
        "version": "registry.example.com/acme@sha256:abc",
    }

    provenance = resolve_system_provenance(
        adapter, _config(system_provenance_json=json.dumps(supplied))
    )

    assert provenance.status == "user_supplied"
    assert provenance.system_id == "acme-rag"
    assert provenance.version_source == "container"
    assert provenance.version == "registry.example.com/acme@sha256:abc"


def test_adapter_capture_wins_over_manifest(repo):
    class Adapter:
        name = "custom"

        def system_provenance(self, config):
            return capture_git_provenance(repo, system_id="custom")

    supplied = {"system_id": "custom", "version_source": "user", "version": "v9"}
    provenance = resolve_system_provenance(
        Adapter(), _config(system_provenance_json=json.dumps(supplied))
    )

    assert provenance.status == "captured"
    assert provenance.version == _git(repo, "rev-parse", "HEAD")


def test_manifest_fills_in_when_adapter_cannot_capture(tmp_path):
    class Adapter:
        name = "custom"

        def system_provenance(self, config):
            return capture_git_provenance(tmp_path, system_id="custom")

    supplied = {"system_id": "custom", "version_source": "deployment", "version": "prod-42"}
    provenance = resolve_system_provenance(
        Adapter(), _config(system_provenance_json=json.dumps(supplied))
    )

    assert provenance.status == "user_supplied"
    assert provenance.version == "prod-42"


def test_failing_adapter_capture_degrades_to_not_available():
    class Adapter:
        name = "custom"

        def system_provenance(self, config):
            raise RuntimeError("boom")

    provenance = resolve_system_provenance(Adapter(), _config())

    assert provenance.status == "not_available"


def test_blueprint_adapter_captures_its_own_checkout(repo):
    from examples.enterprise_rag_plugin.adapter import EnterpriseRagAdapter

    config = _config(rag_system_adapter="enterprise_rag", enterprise_rag_blueprint_path=str(repo))

    provenance = resolve_system_provenance(EnterpriseRagAdapter(config), config)

    assert provenance.system_id == "enterprise_rag_blueprint"
    assert provenance.status == "captured"
    assert provenance.version == _git(repo, "rev-parse", "HEAD")
    assert provenance.branch == "feature/x"


def test_blueprint_mcp_adapter_captures_its_own_checkout(repo):
    from examples.enterprise_rag_plugin.mcp_adapter import blueprint_provenance

    config = _config(enterprise_rag_blueprint_path=str(repo))

    provenance = blueprint_provenance(config)

    assert provenance.system_id == "enterprise_rag_blueprint"
    assert provenance.status == "captured"


def test_internal_adapter_reports_the_framework_checkout(repo, monkeypatch):
    from benchmark.adapters.internal import InternalRagAdapter

    import benchmark.lineage.provenance as provenance_module

    monkeypatch.setattr(provenance_module, "FRAMEWORK_ROOT", repo)
    provenance = resolve_system_provenance(InternalRagAdapter(), _config(rag_system_adapter="internal"))

    assert provenance.system_id == "benchmark_internal_pipeline"
    assert provenance.version == _git(repo, "rev-parse", "HEAD")


def test_round_trip_through_dict():
    record = SystemProvenance(system_id="s", status="user_supplied", version_source="user", version="1")

    assert SystemProvenance.from_dict(record.as_dict()) == record


def test_from_dict_of_missing_record_is_unknown():
    assert SystemProvenance.from_dict(None).status == "not_available"
