"""Guards for dependency floors that fix known evaluation bugs.

MLflow >= 3.18.0 fixes the DeepEval scorer returning another row's result
under parallel scoring (issue #41). These tests stop the requirements floor
or the lock file from silently drifting back below that version.
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
MLFLOW_MIN = Version("3.18.0")
MLFLOW_PACKAGES = ("mlflow", "mlflow-skinny", "mlflow-tracing")


def _requirement(name: str) -> Requirement:
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        req = Requirement(line)
        if req.name.lower() == name:
            return req
    raise AssertionError(f"{name} missing from requirements.txt")


def _locked_version(name: str) -> Version:
    lock = (REPO_ROOT / "requirements-lock.txt").read_text()
    match = re.search(rf"^{re.escape(name)}==(\S+)$", lock, re.MULTILINE)
    assert match, f"{name} not pinned in requirements-lock.txt"
    return Version(match.group(1))


def test_requirements_floor_excludes_mlflow_below_minimum():
    spec = _requirement("mlflow").specifier
    below_min = Version(f"{MLFLOW_MIN.major}.{MLFLOW_MIN.minor - 1}.99")
    assert not spec.contains(below_min), f"mlflow floor too low: {spec}"
    assert spec.contains(MLFLOW_MIN)


def test_lock_pins_mlflow_family_at_or_above_minimum():
    for name in MLFLOW_PACKAGES:
        assert _locked_version(name) >= MLFLOW_MIN, name
