"""Index the test corpus into the Blueprint's local Chroma vector store.

Run from the Framework repo root, after installing Blueprint deps
(``Enterprise_RAG_Blueprint/{chain,loader}/requirements.txt``) and starting
Ollama with the ``nomic-embed-text`` model available at OLLAMA_BASE_URL.

Usage::

    cd Enterprise_RAG_Blueprint
    PYTHONPATH=. python ../scripts/index_blueprint_corpus.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Load Blueprint .env so chain.retriever picks up INDEX_NAME / VECTORDB_DIR / etc.
from dotenv import load_dotenv

BLUEPRINT_DIR = Path(__file__).resolve().parent.parent / "Enterprise_RAG_Blueprint"
load_dotenv(BLUEPRINT_DIR / ".env")

sys.path.insert(0, str(BLUEPRINT_DIR))
sys.path.insert(0, str(BLUEPRINT_DIR / "loader"))
# Blueprint uses package-relative imports like `from tag_management import ...`,
# so loader/ must be on sys.path directly.
os.chdir(BLUEPRINT_DIR / "loader")

from loader_functions import upload_all_docs_to_vector_store  # noqa: E402

CORPUS_DIR = BLUEPRINT_DIR / "test_documents"

if __name__ == "__main__":
    if not CORPUS_DIR.exists():
        raise SystemExit(f"corpus dir missing: {CORPUS_DIR}")
    print(f"[index] corpus dir: {CORPUS_DIR}")
    print(f"[index] index name: {os.environ.get('INDEX_NAME')}")
    print(f"[index] vectordb dir: {os.environ.get('VECTORDB_DIR')}")
    upload_all_docs_to_vector_store(
        root_path=str(CORPUS_DIR),
        document_tag="General",
        chunk_size=int(os.environ.get("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.environ.get("CHUNK_OVERLAP", "200")),
    )
    print("[index] done")
