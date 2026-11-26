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
# os.getenv("X"), os.environ.get("X"), os.environ["X"], and env helpers such as
# config._env_bool("X") / tracking._env_flag("X").
_ENV_READ_PATTERN = re.compile(
    r"""(?:\bgetenv\(|\benviron\.get\(|\benviron\[|\b_env_\w+\()\s*["']([A-Z][A-Z0-9_]*)["']"""
)


def _benchmark_env_keys() -> frozenset[str]:
    """Env var names the framework reads.

    Collected from .env.example, the local .env, and every env read in
    config.py and the benchmark/ package.
    """
    keys: set[str] = set()
    for name in (".env.example", ".env"):
        path = project_root / name
        if path.is_file():
            keys.update(_ENV_KEY_PATTERN.findall(path.read_text(errors="ignore")))
    sources = [project_root / "config.py", *(project_root / "benchmark").rglob("*.py")]
    for path in sources:
        keys.update(_ENV_READ_PATTERN.findall(path.read_text(errors="ignore")))
    return frozenset(keys)


BENCHMARK_ENV_KEYS = _benchmark_env_keys()


@pytest.fixture(autouse=True)
def isolated_benchmark_env(monkeypatch, tmp_path_factory):
    """Make every test independent of the machine-local .env and shell (#60).

    Disables dotenv loading in ``config`` and clears every benchmark-related
    variable, so code under test only sees its defaults plus what the test
    sets explicitly via ``monkeypatch.setenv``. ``PROMPT_TEMPLATES`` is pinned
    to ``concise`` so config unit tests get a single prompt per matrix cell.

    MLflow tracking is pinned to a throwaway SQLite database so a test never
    touches the machine-local ``mlflow.db`` (#60): without a reachable MLflow
    server, ``benchmark/tracking.py`` and ``benchmark/tracing.py`` fall back
    to ``sqlite:///mlflow.db`` in the repo root, whose schema state depends
    on the host.
    """
    import config

    monkeypatch.setattr(config, "load_dotenv", lambda *args, **kwargs: False)
    for key in BENCHMARK_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PROMPT_TEMPLATES", "concise")
    mlflow_db = tmp_path_factory.mktemp("mlflow") / "mlflow-test.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{mlflow_db}")


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
