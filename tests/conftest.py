"""Shared test configuration and fixtures."""

import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

# Ensure project root is on sys.path so `from config import ...` works
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


@pytest.fixture(autouse=True)
def stable_prompt_template_env(monkeypatch):
    """Keep local .env prompt matrix from changing config unit-test counts."""
    monkeypatch.setenv("PROMPT_TEMPLATES", "concise")


class StubEmbedder(Embeddings):
    """Deterministic bag-of-characters embeddings; no network, no disk."""

    ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 "

    def _vec(self, text: str) -> list[float]:
        counts = [float(text.lower().count(c)) for c in self.ALPHABET]
        norm = sum(v * v for v in counts) ** 0.5 or 1.0
        return [v / norm for v in counts]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)
