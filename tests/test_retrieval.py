"""Tests for benchmark.retrieval — vector store, cache, MMR, and HyDE logic."""

import hashlib
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from benchmark.retrieval import (
    _cache_key,
    build_vector_store,
    cleanup_collection,
    clear_cache,
    retrieve,
    expand_query_with_hyde,
)


class TestCacheKey:
    def test_deterministic(self):
        k1 = _cache_key("nomic-embed-text:latest", 1000, 200, "recursive")
        k2 = _cache_key("nomic-embed-text:latest", 1000, 200, "recursive")
        assert k1 == k2

    def test_different_params_different_keys(self):
        k1 = _cache_key("model-a", 1000, 200, "recursive")
        k2 = _cache_key("model-b", 1000, 200, "recursive")
        assert k1 != k2

    def test_is_sha256_hex(self):
        k = _cache_key("model", 1000, 200, "recursive")
        assert len(k) == 64  # SHA-256 hex digest length
        assert all(c in "0123456789abcdef" for c in k)

    def test_provider_changes_key(self):
        k1 = _cache_key("model", 1000, 200, "recursive", embedding_provider="ollama")
        k2 = _cache_key("model", 1000, 200, "recursive", embedding_provider="huggingface")
        assert k1 != k2

    def test_dataset_subset_and_fingerprint_change_key(self):
        k1 = _cache_key(
            "model", 1000, 200, "recursive",
            dataset_name="ragbench",
            dataset_subset="FinQA",
            dataset_sample_size=50,
            corpus_fingerprint="abc",
        )
        k2 = _cache_key(
            "model", 1000, 200, "recursive",
            dataset_name="ragbench",
            dataset_subset="HotpotQA",
            dataset_sample_size=50,
            corpus_fingerprint="abc",
        )
        k3 = _cache_key(
            "model", 1000, 200, "recursive",
            dataset_name="ragbench",
            dataset_subset="FinQA",
            dataset_sample_size=50,
            corpus_fingerprint="def",
        )
        assert k1 != k2
        assert k1 != k3

    def test_vector_db_backend_changes_key(self):
        k1 = _cache_key("model", 1000, 200, "recursive", vector_db_backend="chroma")
        k2 = _cache_key("model", 1000, 200, "recursive", vector_db_backend="lancedb")
        assert k1 != k2

    def test_paragraph_strategy_ignores_chunk_size_and_overlap(self):
        base = _cache_key("model", None, None, "paragraph")
        assert base == _cache_key("model", 500, 200, "paragraph")
        assert base == _cache_key("model", 1000, 0, "paragraph")
        # other strategies must still honor chunk parameters
        assert _cache_key("model", 500, 200, "recursive") != _cache_key(
            "model", 1000, 200, "recursive"
        )


class TestVectorStoreBackends:
    @patch("benchmark.retrieval.get_embedding_model")
    def test_dispatches_to_registered_backend(self, mock_get_embedding_model):
        import benchmark.retrieval as ret_mod

        chunks = [Document(page_content="a"), Document(page_content="b")]
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding
        mock_store = MagicMock()
        backend = MagicMock()
        backend.build.return_value = mock_store

        with patch.dict(ret_mod._VECTOR_STORE_BACKENDS, {"custom": backend}):
            result = build_vector_store(
                chunks,
                "embed",
                "test_collection",
                embedding_provider="huggingface",
                vector_db_backend="custom",
                lancedb_path="custom-path",
                create_if_missing=False,
            )

        assert result is mock_store
        backend.build.assert_called_once()
        context = backend.build.call_args.args[0]
        assert context.chunks == chunks
        assert context.embedding_model is mock_embedding
        assert context.collection_name == "test_collection"
        assert context.expected_count == len(chunks)
        assert context.create_if_missing is False
        assert context.lancedb_path == "custom-path"
        mock_get_embedding_model.assert_called_once_with(
            "embed",
            "http://localhost:11434",
            None,
            provider="huggingface",
        )

    @patch("benchmark.retrieval.LanceDBVectorStore")
    @patch("benchmark.retrieval.get_embedding_model")
    def test_lancedb_backend_builds_adapter(
        self, mock_get_embedding_model, mock_lancedb_vector_store
    ):
        chunks = [Document(page_content="a")]
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding
        mock_store = MagicMock()
        mock_lancedb_vector_store.return_value = mock_store

        result = build_vector_store(
            chunks,
            "embed",
            "test_collection",
            vector_db_backend="lancedb",
            lancedb_path=".custom-lancedb",
            create_if_missing=False,
        )

        assert result is mock_store
        mock_lancedb_vector_store.assert_called_once_with(
            ".custom-lancedb",
            "test_collection",
            mock_embedding,
            create_if_missing=False,
            chunks=chunks,
        )

    @patch("benchmark.retrieval.get_embedding_model")
    def test_unsupported_backend_raises(self, mock_get_embedding_model):
        mock_get_embedding_model.return_value = MagicMock()

        try:
            build_vector_store([], "embed", "test_collection", vector_db_backend="unknown")
        except ValueError as exc:
            assert "Unsupported vector DB backend: unknown" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestBuildVectorStore:
    @patch("benchmark.retrieval.Chroma")
    @patch("benchmark.retrieval.get_embedding_model")
    @patch("benchmark.retrieval._get_client")
    def test_reuses_existing_collection_when_count_matches(
        self, mock_get_client, mock_get_embedding_model, mock_chroma
    ):
        chunks = [Document(page_content="a"), Document(page_content="b")]
        mock_client = MagicMock()
        mock_existing = MagicMock(name="collection_ref")
        mock_existing.name = "test_collection"
        mock_client.list_collections.return_value = [mock_existing]
        mock_collection = MagicMock()
        mock_collection.count.return_value = len(chunks)
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        result = build_vector_store(chunks, "embed", "test_collection")

        assert result is mock_store
        mock_client.delete_collection.assert_not_called()
        mock_store.add_documents.assert_not_called()

    @patch("benchmark.retrieval.Chroma")
    @patch("benchmark.retrieval.get_embedding_model")
    @patch("benchmark.retrieval._get_client")
    def test_rebuilds_existing_collection_when_count_differs(
        self, mock_get_client, mock_get_embedding_model, mock_chroma
    ):
        chunks = [Document(page_content="a"), Document(page_content="b")]
        mock_client = MagicMock()
        mock_existing = MagicMock(name="collection_ref")
        mock_existing.name = "test_collection"
        mock_client.list_collections.return_value = [mock_existing]
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_embedding = MagicMock()
        mock_get_embedding_model.return_value = mock_embedding
        mock_store = MagicMock()
        mock_chroma.return_value = mock_store

        result = build_vector_store(chunks, "embed", "test_collection")

        assert result is mock_store
        mock_client.delete_collection.assert_called_once_with("test_collection")
        mock_store.add_documents.assert_called_once_with(chunks)

    @patch("benchmark.retrieval.get_embedding_model")
    @patch("benchmark.retrieval._get_client")
    def test_query_mode_raises_when_existing_count_differs(
        self, mock_get_client, mock_get_embedding_model
    ):
        chunks = [Document(page_content="a"), Document(page_content="b")]
        mock_client = MagicMock()
        mock_existing = MagicMock(name="collection_ref")
        mock_existing.name = "test_collection"
        mock_client.list_collections.return_value = [mock_existing]
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        mock_client.get_collection.return_value = mock_collection
        mock_get_client.return_value = mock_client
        mock_get_embedding_model.return_value = MagicMock()

        try:
            build_vector_store(
                chunks, "embed", "test_collection", create_if_missing=False
            )
        except RuntimeError as exc:
            assert "has 0 documents, expected 2" in str(exc)
        else:
            raise AssertionError("expected RuntimeError")
        mock_client.delete_collection.assert_not_called()


class TestRetrieve:
    def _mock_store(self):
        store = MagicMock()
        store.similarity_search.return_value = [
            Document(page_content="doc1"),
            Document(page_content="doc2"),
        ]
        store.max_marginal_relevance_search.return_value = [
            Document(page_content="mmr1"),
            Document(page_content="mmr2"),
        ]
        return store

    def test_similarity_default(self):
        store = self._mock_store()
        results = retrieve(store, "query", 3)
        store.similarity_search.assert_called_once_with("query", k=3)
        assert len(results) == 2

    def test_similarity_explicit(self):
        store = self._mock_store()
        results = retrieve(store, "query", 5, retrieval_strategy="similarity")
        store.similarity_search.assert_called_once_with("query", k=5)

    def test_mmr_with_fetch_k(self):
        store = self._mock_store()
        results = retrieve(
            store, "query", 3,
            retrieval_strategy="mmr", fetch_k=20, mmr_lambda=0.7,
        )
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=3, fetch_k=20, lambda_mult=0.7,
        )
        assert results[0].page_content == "mmr1"

    def test_mmr_fetch_k_defaults_to_top_k_times_4(self):
        store = self._mock_store()
        retrieve(store, "query", 5, retrieval_strategy="mmr")
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=5, fetch_k=20, lambda_mult=0.5,
        )

    def test_mmr_respects_custom_fetch_k(self):
        store = self._mock_store()
        retrieve(store, "query", 5, retrieval_strategy="mmr", fetch_k=50)
        store.max_marginal_relevance_search.assert_called_once_with(
            "query", k=5, fetch_k=50, lambda_mult=0.5,
        )


class TestHyDE:
    def test_returns_hypothetical_answer(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = "The answer is 42."
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "The answer is 42."
        llm.invoke.assert_called_once()

    def test_falls_back_to_question_on_empty_response(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = ""
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"

    def test_falls_back_to_question_on_none_content(self):
        llm = MagicMock()
        response = MagicMock()
        response.content = None
        llm.invoke.return_value = response

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"

    def test_falls_back_to_question_on_exception(self):
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")

        result = expand_query_with_hyde(llm, "What is the answer?")
        assert result == "What is the answer?"


class TestMultiHopRetrieval:
    def _store_returning_per_query(self, per_query):
        store = MagicMock()
        calls = {"n": 0}

        def side_effect(query, k=3, **kwargs):
            docs = per_query[min(calls["n"], len(per_query) - 1)]
            calls["n"] += 1
            return docs

        store.similarity_search.side_effect = side_effect
        return store

    def _llm_reply(self, text):
        llm = MagicMock()
        response = MagicMock()
        response.content = text
        llm.invoke.return_value = response
        return llm

    def test_merges_docs_from_followup_round(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [Document(page_content="hop1", metadata={"doc_id": "a"})]
        round2 = [Document(page_content="hop2", metadata={"doc_id": "b"})]
        store = self._store_returning_per_query([round1, round2])
        llm = self._llm_reply("Who directed the film?")

        docs = retrieve_multihop(
            store, llm, "question", top_k=4, rounds=2,
        )

        doc_ids = [d.metadata.get("doc_id") for d in docs]
        assert doc_ids == ["a", "b"]
        # Second round used the LLM-generated follow-up query
        second_query = store.similarity_search.call_args_list[1][0][0]
        assert second_query == "Who directed the film?"

    def test_deduplicates_docs_across_rounds(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [Document(page_content="same", metadata={"doc_id": "a"})]
        round2 = [
            Document(page_content="same", metadata={"doc_id": "a"}),
            Document(page_content="new", metadata={"doc_id": "b"}),
        ]
        store = self._store_returning_per_query([round1, round2])
        llm = self._llm_reply("follow-up")

        docs = retrieve_multihop(store, llm, "question", top_k=4, rounds=2)
        assert [d.metadata["doc_id"] for d in docs] == ["a", "b"]

    def test_respects_top_k_limit(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [
            Document(page_content="r1a", metadata={"doc_id": "a"}),
            Document(page_content="r1b", metadata={"doc_id": "b"}),
        ]
        round2 = [
            Document(page_content="r2a", metadata={"doc_id": "c"}),
            Document(page_content="r2b", metadata={"doc_id": "d"}),
        ]
        store = self._store_returning_per_query([round1, round2])

        docs = retrieve_multihop(
            store, self._llm_reply("q2"), "question", top_k=3, rounds=2,
        )
        assert len(docs) == 3

    def test_llm_failure_falls_back_to_first_round(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [Document(page_content="hop1", metadata={"doc_id": "a"})]
        store = self._store_returning_per_query([round1])
        llm = MagicMock()
        llm.invoke.side_effect = RuntimeError("LLM unavailable")

        docs = retrieve_multihop(store, llm, "question", top_k=3, rounds=2)
        assert [d.metadata["doc_id"] for d in docs] == ["a"]

    def test_single_round_delegates_to_plain_retrieval(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [Document(page_content="doc1", metadata={"doc_id": "a"})]
        store = self._store_returning_per_query([round1])
        llm = MagicMock()

        docs = retrieve_multihop(store, llm, "question", top_k=3, rounds=1)
        llm.invoke.assert_not_called()
        assert [d.metadata["doc_id"] for d in docs] == ["a"]

    def test_followup_docs_survive_when_round1_fills_top_k(self):
        """Regression: round-1 already returns top_k unique docs, so the
        merged list was full and every round-2 document was truncated away.
        Multi-hop retrieval must reserve budget for follow-up rounds."""
        from benchmark.retrieval import retrieve_multihop

        round1_pool = [
            Document(page_content=f"d{i}", metadata={"doc_id": str(i)})
            for i in range(4)
        ]
        followup_pool = [
            Document(page_content="d5", metadata={"doc_id": "5"}),   # new
            Document(page_content="d6", metadata={"doc_id": "6"}),   # new
            Document(page_content="d2", metadata={"doc_id": "2"}),   # dup
        ]
        store = MagicMock()
        responses = [round1_pool, followup_pool]
        calls = {"n": 0}

        def side_effect(query, k=3, **kwargs):
            docs = responses[min(calls["n"], 1)]
            calls["n"] += 1
            return docs[:k]  # real stores respect k

        store.similarity_search.side_effect = side_effect

        docs = retrieve_multihop(
            store, self._llm_reply("follow-up"), "question",
            top_k=4, rounds=2,
        )

        retrieved = [d.metadata["doc_id"] for d in docs]
        assert "5" in retrieved and "6" in retrieved, (
            f"follow-up docs dropped: {retrieved}"
        )

    def test_empty_content_from_llm_ends_iteration(self):
        from benchmark.retrieval import retrieve_multihop

        round1 = [Document(page_content="hop1", metadata={"doc_id": "a"})]
        store = self._store_returning_per_query([round1])

        docs = retrieve_multihop(
            store, self._llm_reply(""), "question", top_k=3, rounds=3,
        )
        # only one similarity_search call (initial round)
        assert store.similarity_search.call_count == 1
        assert len(docs) == 1


class TestHybridRetrieval:
    """Hybrid retrieval: dense vector search fused with BM25 via weighted RRF."""

    def _store_returning(self, dense_docs):
        store = MagicMock()
        store.similarity_search.return_value = dense_docs
        return store

    def _chunks(self):
        return [
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"doc_id": "fox"}),
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
            Document(page_content="a completely different story about pirates", metadata={"doc_id": "pirates"}),
        ]

    def test_alpha_zero_prefers_bm25_ranking(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"doc_id": "fox"}),
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "flux capacitor", 2,
            chunks=self._chunks(), alpha=0.0,
        )

        assert docs[0].metadata["doc_id"] == "flux"

    def test_alpha_zero_returns_keyword_side_only(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="unrelated dense hit one", metadata={"doc_id": "d1"}),
            Document(page_content="unrelated dense hit two", metadata={"doc_id": "d2"}),
            Document(page_content="unrelated dense hit three", metadata={"doc_id": "d3"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "flux capacitor", 3,
            chunks=self._chunks(), alpha=0.0,
        )

        doc_ids = [d.metadata["doc_id"] for d in docs]
        assert "d1" not in doc_ids and "d2" not in doc_ids, (
            f"alpha=0 must not leak dense-only docs: {doc_ids}"
        )

    def test_alpha_one_returns_dense_side_only(self):
        from benchmark.retrieval import retrieve_hybrid

        store = self._store_returning([
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"doc_id": "fox"}),
        ])

        docs = retrieve_hybrid(
            store, "quick brown", 3,
            chunks=self._chunks(), alpha=1.0,
        )

        assert [d.metadata["doc_id"] for d in docs] == ["fox"]

    def test_bm25_ranking_matches_known_okapi_order(self):
        """Worked example: standard Okapi BM25 with the log-IDF must rank the
        document containing the rare query terms first, and rank by term
        frequency when every doc contains the term."""
        from benchmark.retrieval import BM25Index

        chunks = [
            Document(page_content="cache cache cache eviction policy"),
            Document(page_content="cache brief note"),
        ]
        index = BM25Index(chunks)
        assert index.search("cache eviction", 2)[0].page_content.startswith("cache cache")

    def test_fusion_order_deterministic_on_ties(self):
        """Two docs tied at the same fused score (same ranks on both sides)
        must always come out in dense-side order, in every process."""
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="alpha term", metadata={"doc_id": "first"}),
            Document(page_content="alpha term", metadata={"doc_id": "second"}),
        ]
        store = self._store_returning(dense)

        for _ in range(3):
            docs = retrieve_hybrid(
                store, "alpha term", 2,
                chunks=dense, alpha=0.5,
            )
            assert [d.metadata["doc_id"] for d in docs] == ["first", "second"]

    def test_alpha_one_prefers_dense_ranking(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"doc_id": "fox"}),
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "flux capacitor", 2,
            chunks=self._chunks(), alpha=1.0,
        )

        assert docs[0].metadata["doc_id"] == "fox"

    def test_fusion_merges_keyword_only_docs(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "pirates story", 3,
            chunks=self._chunks(), alpha=0.5,
        )

        doc_ids = [d.metadata["doc_id"] for d in docs]
        assert "pirates" in doc_ids
        assert len(docs) <= 3

    def test_deduplicates_docs_found_by_both_retrievers(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "flux capacitor", 3,
            chunks=self._chunks(), alpha=0.5,
        )

        doc_ids = [d.metadata["doc_id"] for d in docs]
        assert len(doc_ids) == len(set(doc_ids))

    def test_respects_top_k(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [
            Document(page_content="the quick brown fox jumps over the lazy dog", metadata={"doc_id": "fox"}),
            Document(page_content="quantum flux capacitor calibration", metadata={"doc_id": "flux"}),
            Document(page_content="a completely different story about pirates", metadata={"doc_id": "pirates"}),
        ]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(
            store, "flux capacitor pirates", 2,
            chunks=self._chunks(), alpha=0.5,
        )

        assert len(docs) == 2

    def test_empty_corpus_falls_back_to_dense(self):
        from benchmark.retrieval import retrieve_hybrid

        dense = [Document(page_content="doc1", metadata={"doc_id": "a"})]
        store = self._store_returning(dense)

        docs = retrieve_hybrid(store, "query", 3, chunks=[], alpha=0.5)

        assert [d.metadata["doc_id"] for d in docs] == ["a"]


class TestCleanupCollection:
    @patch("benchmark.retrieval._get_client")
    def test_deletes_existing_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_collection.name = "test_collection"
        mock_client.list_collections.return_value = [mock_collection]
        mock_get_client.return_value = mock_client

        cleanup_collection("test_collection")

        mock_client.delete_collection.assert_called_once_with("test_collection")

    @patch("benchmark.retrieval._get_client")
    def test_noop_for_nonexistent_collection(self, mock_get_client):
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client

        cleanup_collection("nonexistent")

        mock_client.delete_collection.assert_not_called()


class TestClearCache:
    @patch("benchmark.retrieval._get_client")
    def test_clears_cache_and_collections(self, mock_get_client):
        import benchmark.retrieval as ret_mod

        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        mock_get_client.return_value = mock_client

        # Manually populate the cache
        ret_mod._vector_store_cache["test_key"] = MagicMock()

        clear_cache()

        assert len(ret_mod._vector_store_cache) == 0
