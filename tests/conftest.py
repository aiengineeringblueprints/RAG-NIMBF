"""Shared test configuration and fixtures."""

import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

# Ensure project root is on sys.path so `from config import ...` works
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


_ENV_KEY_PATTERN = re.compile(r"^\s*#?\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=", re.M)
_GETENV_PATTERN = re.compile(r"""os\.(?:getenv|environ\.get)\(\s*["']([A-Z][A-Z0-9_]*)["']""")


def _benchmark_env_keys() -> frozenset[str]:
    """Env var names the framework reads: .env.example, local .env, config.py."""
    keys: set[str] = set()
    for name in (".env.example", ".env"):
        path = project_root / name
        if path.is_file():
            keys.update(_ENV_KEY_PATTERN.findall(path.read_text(errors="ignore")))
    keys.update(_GETENV_PATTERN.findall((project_root / "config.py").read_text()))
    return frozenset(keys)


BENCHMARK_ENV_KEYS = _benchmark_env_keys()


@pytest.fixture(autouse=True)
def isolated_benchmark_env(monkeypatch):
    """Make every test independent of the machine-local .env and shell (#60).

    Disables dotenv loading in ``config`` and clears every benchmark-related
    variable, so config code under test only sees its own defaults plus what
    the test sets explicitly via ``monkeypatch.setenv``.
    """
    import config

    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: False)
    for key in BENCHMARK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
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
