# E. VectorStore Adapter Layer

## 1. Goal

Decouple the benchmark from any single vector database so the same corpus
and embedding model can be re-run against multiple backends. The interface
mirrors RAGPerf's `DBInstance` (arXiv:2603.10765v1, Figure 4 / Sec 3.3.2)
so adapter implementations stay drop-in compatible.

## 2. Architecture

```
                            +--------------------------+
   caller (YAML / Python)   | benchmark.vectorstore    |
   -----------------------> | get_vector_store(config) |
                            +-----------+--------------+
                                        |
                   +--------------------+--------------------+
                   |                    |                    |
            +------v------+       +-----v-----+        +-----v-----+
            | ChromaVSS   |       | QdrantVSS |  ...   | LanceDBVSS|
            +-------------+       +-----------+        +-----------+
                   |
        shares _chroma_lock + _chroma_client
        with benchmark.retrieval
        (so cleanup_collection / clear_cache
         still affect every adapter)
```

Files:

- `benchmark/vectorstore/__init__.py` — public re-exports.
- `benchmark/vectorstore/base.py` — `VectorStoreAdapter` ABC, `VectorStoreConfig`, `SearchHit`, helper functions.
- `benchmark/vectorstore/factory.py` — `get_vector_store(config)`, `register_adapter(name, factory)`, `SUPPORTED_BACKENDS()`.
- `benchmark/vectorstore/chroma_store.py` — Chroma adapter (default).
- `benchmark/vectorstore/qdrant_store.py` — Qdrant adapter.
- `benchmark/vectorstore/milvus_store.py` — Milvus adapter.
- `benchmark/vectorstore/lancedb_store.py` — LanceDB adapter.

The legacy `benchmark/retrieval.py` API (`build_vector_store`, `retrieve`,
`cleanup_collection`, `clear_cache`, `ChromaVectorStoreBackend`,
`LanceDBVectorStoreBackend`, `register_vector_store_backend`,
`available_vector_store_backends`) is unchanged for backward compatibility.
A thin re-export `benchmark.retrieval.get_vector_store` is provided so
callers can resolve adapters through either module.

## 3. Adapter Status

| Backend | Adapter                     | Status      | Dep                  | Tested?                              |
| ------- | --------------------------- | ----------- | -------------------- | ------------------------------------ |
| Chroma  | `ChromaVectorStore`         | stable      | `chromadb` (default) | yes — `tests/test_vectorstore_chroma.py` + conformance |
| Qdrant  | `QdrantVectorStore`         | implemented | `qdrant-client` (opt)| yes when dep installed; auto-skip    |
| Milvus  | `MilvusVectorStore`         | implemented | `pymilvus` (opt)     | yes when server reachable; auto-skip |
| LanceDB | `LanceDBVectorStore`        | implemented | `lancedb` (default)  | yes — `tests/test_vectorstore_lancedb.py` + conformance |

Each non-Chroma adapter does a lazy import inside its module so importing
`benchmark.vectorstore` is always cheap and never requires optional deps.

## 4. Per-backend dependencies

Add only what you use:

```bash
pip install chromadb                # default; already required by the framework
pip install lancedb                 # already required
pip install qdrant-client           # optional
pip install pymilvus                # optional; also requires a running Milvus server
```

Optional deps are listed as comments at the top of `requirements.txt`.

## 5. YAML config examples

Add a `vector_store:` block to any experiment YAML. Unknown keys are
ignored; missing keys default to Chroma-equivalent behaviour.

### Chroma (default — backward compatible)

```yaml
vector_store:
  backend: chroma
  path: .chroma
  collection: default
  index_type: hnsw          # only 'hnsw' supported by Chroma
  metric: cosine            # cosine | l2 | ip
```

### Qdrant (embedded local mode)

```yaml
vector_store:
  backend: qdrant
  path: ./qdrant_data
  collection: default
  index_type: hnsw
  metric: cosine
  dimension: 768            # required for build_index
```

### Qdrant (server mode)

```yaml
vector_store:
  backend: qdrant
  host: qdrant.local
  port: 6333
  api_key: ${QDRANT_API_KEY}  # resolved by your env loader
  collection: default
  metric: cosine
  dimension: 768
```

### Milvus (requires running server)

```yaml
vector_store:
  backend: milvus
  host: localhost
  port: 19530
  collection: default
  index_type: hnsw          # hnsw | ivf | flat
  metric: cosine            # cosine | l2 | ip
  dimension: 768            # required
```

### LanceDB

```yaml
vector_store:
  backend: lancedb
  path: .lancedb
  collection: default
  index_type: hnsw          # hnsw | ivf | flat
  metric: cosine
```

When `backend` is omitted, the factory returns a `ChromaVectorStore`
pointing at `.chroma`, reproducing the pre-refactor default exactly.

## 6. Programmatic usage

```python
from benchmark.vectorstore import VectorStoreConfig, get_vector_store

cfg = VectorStoreConfig(backend="qdrant", path=".qdrant", dimension=768)
store = get_vector_store(cfg)

store.build_index("hnsw", "cosine", "docs")        # idempotent
n = store.insert(vectors, chunks, "docs")          # returns count inserted
hits = store.search(query_vectors, "docs", top_k=5)
assert hits[0][0].text                              # SearchHit with .id/.score/.text/.metadata

store.delete("file_42")                             # by metadata.file_id
store.drop("docs")                                  # drop one collection
store.drop()                                        # drop all
```

Helper wrappers exist for callers that prefer to pass text rather than
pre-computed vectors:

```python
store.insert_texts(["hello", "world"], embedder, "docs")
hits = store.search_queries(["greeting"], embedder, "docs", top_k=3)
```

## 7. Backward-compatibility verification

The legacy entry points used by `main.py` (`build_vector_store`,
`retrieve`, `cleanup_collection`, `clear_cache`) are unchanged. The
existing test suite (`tests/test_retrieval.py`, `tests/test_config.py`)
continues to pass without modification:

- `tests/test_retrieval.py` — 33 tests covering cache key, build_vector_store
  dispatch, retrieve + MMR, HyDE, cleanup_collection, clear_cache.
- `tests/test_config.py` — 58 tests including `vector_db_backend` /
  `lancedb_path` loading and the custom-backend registration hook.

The new Chroma adapter reuses `benchmark.retrieval._get_client()` and
`_chroma_lock`, so collections it creates are visible to
`cleanup_collection` and `clear_cache`, and vice-versa. This was verified
end-to-end in `tests/test_vectorstore_chroma.py::test_get_vector_store_factory_returns_chroma`.

## 8. Known limitations

- **Chroma** is not thread-safe. The shared `_chroma_lock` in
  `benchmark.retrieval` serialises every adapter call; this preserves
  correctness but limits throughput under high concurrency.
- **Chroma** ignores `index_type` (always HNSW) — passing anything else
  raises `NotImplementedError` with a clear message.
- **Qdrant** local-mode requires the bundled RocksDB wheel; on platforms
  where it is unavailable, switch to server mode (`host: ...`).
- **Milvus** requires a running server. The adapter raises
  `ConnectionError` with the host/port if the server cannot be reached.
- **LanceDB** builds the actual index lazily on the table, so `build_index`
  only records the chosen metric for subsequent searches; this is logged
  at DEBUG level.
- **Score semantics differ across backends**: Chroma returns cosine
  *distance* (we convert to `1 - distance`), Qdrant and Milvus return
  similarity, LanceDB returns distance. Callers comparing scores across
  backends should normalise before drawing conclusions.
- **`update(file_id, new_text)`** is not implemented in the base class.
  Override per-adapter if needed; the default raises
  `NotImplementedError` so silent no-ops are impossible.
- **`delete(file_id)`** filters on the metadata key `file_id`. Set
  `metadata={"file_id": ...}` at insert time to make the helper work.

## 9. Registering a custom adapter

```python
from benchmark.vectorstore import register_adapter, VectorStoreAdapter, VectorStoreConfig

class MyBackend(VectorStoreAdapter):
    db_type = "mybackend"
    # ... implement all abstract methods ...

register_adapter("mybackend", lambda cfg: MyBackend(...))
store = get_vector_store(VectorStoreConfig(backend="mybackend"))
```

The registry is case-insensitive and lives in
`benchmark/vectorstore/factory.py`.
