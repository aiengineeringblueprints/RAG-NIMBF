"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path so `from config import ...` works
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def stable_prompt_template_env(monkeypatch):
    """Keep local .env prompt matrix from changing config unit-test counts."""
    monkeypatch.setenv("PROMPT_TEMPLATES", "concise")
