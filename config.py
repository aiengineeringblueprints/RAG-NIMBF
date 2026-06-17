import os
import importlib
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

from benchmark.generation import AnswerStripMode
from benchmark.providers import parse_model_id


def _parse_list(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_list(value: str, env_var_name: str) -> list[int]:
    result: list[int] = []
    for v in value.split(","):
        v = v.strip()
        if not v:
            continue
        try:
            result.append(int(v))
        except ValueError:
            raise ValueError(
                f"Invalid integer value '{v}' in {env_var_name}. "
                f"Expected a comma-separated list of integers."
            ) from None
    return result


@dataclass(frozen=True)
class BenchmarkConfig:
    llm_model: str
    llm_provider: str
    embedding_model: str
    chunk_size: int | None
    chunk_overlap: int | None
    chunking_strategy: str
    retrieval_top_k: int
    max_new_tokens: int
    # Shared defaults (used when per-role vars are not set)
    ollama_base_url: str
    ollama_api_key: str | None
    openai_compat_base_url: str | None
    openai_compat_api_key: str | None
    # Per-role URLs (fallback to shared defaults)
    llm_ollama_base_url: str | None
    llm_ollama_api_key: str | None
    llm_openai_compat_base_url: str | None
    llm_openai_compat_api_key: str | None
    eval_critic_ollama_base_url: str | None
    eval_critic_ollama_api_key: str | None
    eval_critic_openai_compat_base_url: str | None
    eval_critic_openai_compat_api_key: str | None
    embedding_ollama_base_url: str | None
    embedding_ollama_api_key: str | None
    eval_critic_max_tokens: int
    # Dataset
    dataset_name: str
    dataset_subset: str
    dataset_sample_size: int
    eval_critic_llm: str
    eval_critic_embedding: str
    custom_metrics_bert_model: str | None = None
    dataset_path: str | None = None
    dataset_question_field: str = "question"
    dataset_ground_truth_field: str = "ground_truth"
    dataset_context_field: str = "context"
    dataset_metadata_field: str = "metadata"
    ragas_enabled: bool = True
    custom_metrics_enabled: bool = True
    # Prompt template
    prompt_template: str = "concise"
    # Reranker
    reranker_model: str | None = None
    reranker_top_k: int = 3
    # Generator output post-processing (thinking tags, reasoning heuristics)
    llm_answer_strip_mode: AnswerStripMode = "tags_only"
    llm_answer_value_fallback: bool = True
    # Retrieval strategy
    retrieval_strategy: str = "similarity"     # "similarity" | "mmr"
    retrieval_fetch_k: int | None = None       # MMR oversampling
    retrieval_mmr_lambda: float = 0.5          # 0 = max diversity, 1 = max relevance
    retrieval_use_hyde: bool = False           # HyDE query expansion
    # Retrieval mode
    retrieval_mode: str = "retrieval"  # "retrieval" | "direct"
    custom_retrieval_metrics_mode: str = "heuristic"  # heuristic | gold_doc
    # Semantic chunking
    semantic_breakpoint_type: str = "percentile"    # percentile | standard_deviation | interquartile
    semantic_breakpoint_amount: int = 95
    vector_db_backend: str = "chroma"  # chroma | lancedb
    lancedb_path: str = ".lancedb"
    benchmark_stage: str = "all"  # all | index | query
    # RAG system adapter
    rag_system_adapter: str = "internal"  # internal | http
    rag_adapter_accepts: str = ""  # comma-separated: chunker,embedder,retriever,reranker,llm,prompt
    rag_http_endpoint_url: str | None = None
    rag_http_timeout_seconds: float = 60.0
    rag_http_answer_field: str = "answer"
    rag_http_contexts_field: str = "contexts"
    rag_http_metadata_field: str = "metadata"
    rag_http_timings_field: str = "timings"
    rag_http_headers: str | None = None
    rag_http_auth_header: str | None = None
    rag_http_auth_value: str | None = None

    @property
    def name(self) -> str:
        if self.chunking_strategy == "semantic":
            parts = (
                f"semantic_bp{self.semantic_breakpoint_type}"
                f"{self.semantic_breakpoint_amount}"
                f"_{self.embedding_model}_{self.llm_model}"
                f"_{self.prompt_template}"
            )
        else:
            parts = (
                f"{self.chunking_strategy}_cs{self.chunk_size}_co{self.chunk_overlap}"
                f"_{self.embedding_model}_{self.llm_model}"
                f"_{self.prompt_template}"
            )
        if self.reranker_model:
            parts += f"_rerank-{self.reranker_model}"
        if self.retrieval_top_k != 5:
            parts += f"_k{self.retrieval_top_k}"
        if self.retrieval_strategy != "similarity":
            parts += f"_mmr-l{self.retrieval_mmr_lambda}"
        if self.retrieval_use_hyde:
            parts += "_hyde"
        if self.retrieval_mode == "direct":
            parts += "_direct"
        if self.vector_db_backend != "chroma":
            parts += f"_{self.vector_db_backend}"
        if self.rag_system_adapter != "internal":
            parts += f"_{self.rag_system_adapter}"
        return parts

    @property
    def embedding_provider(self) -> str:
        """Provider for the embedding model (parsed from prefix)."""
        return parse_model_id(self.embedding_model)[0]

    def llm_base_url(self) -> str:
        """Return the base URL for the generator LLM's provider."""
        if self.llm_provider == "openai":
            return self.llm_openai_compat_base_url or self.openai_compat_base_url or ""
        return self.llm_ollama_base_url or self.ollama_base_url

    def llm_api_key(self) -> str | None:
        """Return the API key for the generator LLM's provider."""
        if self.llm_provider == "openai":
            return self.llm_openai_compat_api_key or self.openai_compat_api_key
        return self.llm_ollama_api_key or self.ollama_api_key

    def eval_critic_base_url(self, provider: str) -> str:
        """Return the base URL for the critic LLM's provider."""
        if provider == "openai":
            return (
                self.eval_critic_openai_compat_base_url
                or self.openai_compat_base_url
                or ""
            )
        return self.eval_critic_ollama_base_url or self.ollama_base_url

    def eval_critic_api_key(self, provider: str) -> str | None:
        """Return the API key for the critic LLM's provider."""
        if provider == "openai":
            return (
                self.eval_critic_openai_compat_api_key
                or self.openai_compat_api_key
            )
        return self.eval_critic_ollama_api_key or self.ollama_api_key

    def embedding_base_url(self) -> str:
        """Return the base URL for embedding models."""
        return self.embedding_ollama_base_url or self.ollama_base_url

    def embedding_api_key(self) -> str | None:
        """Return the API key for embedding models."""
        return self.embedding_ollama_api_key or self.ollama_api_key



def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive non-zero integer, got {value}")


def _chunk_parameter_pairs_for_strategy(
    strategy: str,
    chunk_sizes: list[int],
    chunk_overlaps: list[int],
) -> list[tuple[int | None, int | None]]:
    """Return chunk parameter pairs that actually affect a strategy.

    LangChain's SemanticChunker is controlled by breakpoint parameters and
    embeddings, not fixed chunk sizes or overlaps. Represent those ignored
    values as None so reports and cache keys do not imply a size/overlap sweep.
    """
    if strategy == "semantic":
        return [(None, None)]
    return list(product(chunk_sizes, chunk_overlaps))


def get_all_combinations(config_path: str | Path | None = None) -> list[BenchmarkConfig]:
    """Return benchmark configs from .env or an optional JSON/YAML manifest.

    The manifest path can be passed explicitly or through BENCHMARK_CONFIG_FILE.
    Environment variables still provide provider URLs, API keys, and defaults for
    fields omitted by the manifest.
    """
    load_dotenv()

    manifest_env = os.getenv("BENCHMARK_CONFIG_FILE") or os.getenv(
        "EXPERIMENT_MANIFEST"
    )
    manifest_path = Path(config_path) if config_path is not None else (
        Path(manifest_env) if manifest_env else None
    )
    if manifest_path is not None:
        from benchmark.orchestration.matrix import (
            build_configs_from_spec,
            load_experiment_spec,
        )

        return build_configs_from_spec(
            load_experiment_spec(manifest_path),
            base_configs=get_env_combinations(load_env=False),
        )

    return get_env_combinations(load_env=False)


def get_env_combinations(load_env: bool = True) -> list[BenchmarkConfig]:
    """Return concrete configs from the legacy .env matrix."""
    if load_env:
        load_dotenv()

    llm_models = _parse_list(os.getenv("LLM_MODELS", "gemma3:4b"))
    embedding_models = _parse_list(os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest"))
    chunk_sizes = _parse_int_list(os.getenv("CHUNK_SIZES", "1000"), "CHUNK_SIZES")
    chunk_overlaps = _parse_int_list(os.getenv("CHUNK_OVERLAPS", "200"), "CHUNK_OVERLAPS")
    chunking_strategies = _parse_list(os.getenv("CHUNKING_STRATEGIES", "recursive"))
    retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", "256"))
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_api_key = os.getenv("OLLAMA_API_KEY") or None
    openai_compat_base_url = os.getenv("OPENAI_COMPAT_BASE_URL") or None
    openai_compat_api_key = os.getenv("OPENAI_COMPAT_API_KEY") or None
    dataset_name = os.getenv("DATASET_NAME", "t2-ragbench")
    dataset_subset = os.getenv("DATASET_SUBSET", "FinQA")
    dataset_sample_size = int(os.getenv("DATASET_SAMPLE_SIZE", "50"))
    dataset_path = os.getenv("DATASET_PATH") or None
    dataset_question_field = os.getenv("DATASET_QUESTION_FIELD", "question").strip() or "question"
    dataset_ground_truth_field = os.getenv("DATASET_GROUND_TRUTH_FIELD", "ground_truth").strip() or "ground_truth"
    dataset_context_field = os.getenv("DATASET_CONTEXT_FIELD", "context").strip() or "context"
    dataset_metadata_field = os.getenv("DATASET_METADATA_FIELD", "metadata").strip() or "metadata"

    # Validate dataset name against registry
    from benchmark.dataset_adapters import REGISTRY
    if dataset_name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {', '.join(sorted(REGISTRY))}"
        )
    eval_critic_llm = os.getenv("EVAL_CRITIC_LLM", "gemma3:12b")
    eval_critic_embedding = os.getenv(
        "EVAL_CRITIC_EMBEDDING",
        os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest").split(",")[0].strip(),
    )
    custom_metrics_bert_model = os.getenv("CUSTOM_METRICS_BERT_MODEL", "roberta-large").strip() or None
    _bert_enabled = os.getenv("CUSTOM_METRICS_BERTSCORE_ENABLED", "true").strip().lower()
    if _bert_enabled in ("0", "false", "no", "off"):
        custom_metrics_bert_model = None
    ragas_enabled = _env_bool("RAGAS_ENABLED", True)
    custom_metrics_enabled = _env_bool("CUSTOM_METRICS_ENABLED", True)

    # Per-role URLs (fall back to shared defaults when not set)
    llm_ollama_base_url = os.getenv("LLM_OLLAMA_BASE_URL") or None
    llm_ollama_api_key = os.getenv("LLM_OLLAMA_API_KEY") or None
    llm_openai_compat_base_url = os.getenv("LLM_OPENAI_COMPAT_BASE_URL") or None
    llm_openai_compat_api_key = os.getenv("LLM_OPENAI_COMPAT_API_KEY") or None
    eval_critic_ollama_base_url = os.getenv("EVAL_CRITIC_OLLAMA_BASE_URL") or None
    eval_critic_ollama_api_key = os.getenv("EVAL_CRITIC_OLLAMA_API_KEY") or None
    eval_critic_openai_compat_base_url = os.getenv("EVAL_CRITIC_OPENAI_COMPAT_BASE_URL") or None
    eval_critic_openai_compat_api_key = os.getenv("EVAL_CRITIC_OPENAI_COMPAT_API_KEY") or None
    embedding_ollama_base_url = os.getenv("EMBEDDING_OLLAMA_BASE_URL") or None
    embedding_ollama_api_key = os.getenv("EMBEDDING_OLLAMA_API_KEY") or None
    eval_critic_max_tokens = int(os.getenv("EVAL_CRITIC_MAX_TOKENS", "4096"))

    # Prompt templates
    prompt_templates = _parse_list(os.getenv("PROMPT_TEMPLATES", "concise"))

    # Validate template names early
    from benchmark.prompt_templates import BUILTIN_TEMPLATES
    for pt in prompt_templates:
        if pt not in BUILTIN_TEMPLATES:
            raise ValueError(
                f"Unknown prompt template '{pt}'. "
                f"Available: {', '.join(sorted(BUILTIN_TEMPLATES))}"
            )

    # Reranker config
    reranker_models_raw = _parse_list(os.getenv("RERANKER_MODELS", ""))
    reranker_models: list[str | None] = reranker_models_raw if reranker_models_raw else [None]
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "3"))

    llm_answer_strip_mode = os.getenv("LLM_ANSWER_STRIP_MODE", "tags_only").strip().lower()
    if llm_answer_strip_mode not in ("full", "tags_only", "off"):
        raise ValueError(
            f"Invalid LLM_ANSWER_STRIP_MODE={llm_answer_strip_mode!r}. "
            "Use: full, tags_only, off"
        )
    _vf = os.getenv("LLM_ANSWER_VALUE_FALLBACK", "true").strip().lower()
    llm_answer_value_fallback = _vf in ("1", "true", "yes", "on")

    # Retrieval strategy
    retrieval_strategy = os.getenv("RETRIEVAL_STRATEGY", "similarity").strip().lower()
    if retrieval_strategy not in ("similarity", "mmr"):
        raise ValueError(
            f"Invalid RETRIEVAL_STRATEGY={retrieval_strategy!r}. Use: similarity, mmr"
        )
    _fk_raw = os.getenv("RETRIEVAL_FETCH_K", "").strip()
    retrieval_fetch_k = int(_fk_raw) if _fk_raw else None
    retrieval_mmr_lambda = float(os.getenv("RETRIEVAL_MMR_LAMBDA", "0.5"))
    if not (0.0 <= retrieval_mmr_lambda <= 1.0):
        raise ValueError(
            f"RETRIEVAL_MMR_LAMBDA must be between 0.0 and 1.0, got {retrieval_mmr_lambda}"
        )

    # HyDE query expansion
    _hyde_raw = os.getenv("RETRIEVAL_USE_HYDE", "false").strip().lower()
    retrieval_use_hyde = _hyde_raw in ("1", "true", "yes", "on")

    # Retrieval mode
    retrieval_mode = os.getenv("RETRIEVAL_MODE", "retrieval").strip().lower()
    if retrieval_mode not in ("retrieval", "direct"):
        raise ValueError(
            f"Invalid RETRIEVAL_MODE={retrieval_mode!r}. Use: retrieval, direct"
        )

    custom_retrieval_metrics_mode = os.getenv(
        "CUSTOM_RETRIEVAL_METRICS_MODE", "heuristic"
    ).strip().lower()
    if custom_retrieval_metrics_mode not in ("heuristic", "gold_doc"):
        raise ValueError(
            "Invalid CUSTOM_RETRIEVAL_METRICS_MODE="
            f"{custom_retrieval_metrics_mode!r}. Use: heuristic, gold_doc"
        )

    # Semantic chunking
    semantic_breakpoint_type = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile").strip().lower()
    if semantic_breakpoint_type not in ("percentile", "standard_deviation", "interquartile"):
        raise ValueError(
            f"Invalid SEMANTIC_BREAKPOINT_TYPE={semantic_breakpoint_type!r}. "
            "Use: percentile, standard_deviation, interquartile"
        )
    semantic_breakpoint_amount = int(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
    _validate_positive_int(semantic_breakpoint_amount, "SEMANTIC_BREAKPOINT_AMOUNT")

    from benchmark.retrieval import available_vector_store_backends

    vector_db_backend = os.getenv("VECTOR_DB_BACKEND", "chroma").strip().lower()
    valid_vector_backends = available_vector_store_backends()
    if vector_db_backend not in valid_vector_backends:
        raise ValueError(
            f"Invalid VECTOR_DB_BACKEND={vector_db_backend!r}. "
            f"Use: {', '.join(valid_vector_backends)}"
        )
    lancedb_path = os.getenv("LANCEDB_PATH", ".lancedb").strip() or ".lancedb"

    benchmark_stage = os.getenv("BENCHMARK_STAGE", "all").strip().lower()
    if benchmark_stage not in ("all", "index", "query"):
        raise ValueError(
            f"Invalid BENCHMARK_STAGE={benchmark_stage!r}. Use: all, index, query"
        )
    if benchmark_stage == "index" and retrieval_mode == "direct":
        raise ValueError("BENCHMARK_STAGE=index requires RETRIEVAL_MODE=retrieval")

    adapter_modules = _parse_list(os.getenv("RAG_ADAPTER_MODULES", ""))
    for module_name in adapter_modules:
        importlib.import_module(module_name)

    from benchmark.adapters import RAG_ADAPTER_REGISTRY

    rag_system_adapter = os.getenv("RAG_SYSTEM_ADAPTER", "internal").strip().lower()
    rag_adapter_accepts = os.getenv("RAG_ADAPTER_ACCEPTS", "")
    valid_rag_adapters = tuple(sorted(RAG_ADAPTER_REGISTRY))
    if rag_system_adapter not in valid_rag_adapters:
        raise ValueError(
            f"Invalid RAG_SYSTEM_ADAPTER={rag_system_adapter!r}. "
            f"Use: {', '.join(valid_rag_adapters)}"
        )
    if rag_system_adapter != "internal" and benchmark_stage == "index":
        raise ValueError("BENCHMARK_STAGE=index is only supported for RAG_SYSTEM_ADAPTER=internal")
    rag_http_endpoint_url = os.getenv("RAG_HTTP_ENDPOINT_URL") or None
    rag_http_timeout_seconds = float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "60"))
    if rag_http_timeout_seconds <= 0:
        raise ValueError("RAG_HTTP_TIMEOUT_SECONDS must be positive")
    rag_http_answer_field = os.getenv("RAG_HTTP_ANSWER_FIELD", "answer").strip() or "answer"
    rag_http_contexts_field = os.getenv("RAG_HTTP_CONTEXTS_FIELD", "contexts").strip() or "contexts"
    rag_http_metadata_field = os.getenv("RAG_HTTP_METADATA_FIELD", "metadata").strip() or "metadata"
    rag_http_timings_field = os.getenv("RAG_HTTP_TIMINGS_FIELD", "timings").strip() or "timings"
    rag_http_headers = os.getenv("RAG_HTTP_HEADERS") or None
    rag_http_auth_header = os.getenv("RAG_HTTP_AUTH_HEADER") or None
    rag_http_auth_value = os.getenv("RAG_HTTP_AUTH_VALUE") or None
    if rag_system_adapter == "http" and not rag_http_endpoint_url:
        raise ValueError("RAG_HTTP_ENDPOINT_URL is required when RAG_SYSTEM_ADAPTER=http")

    # Validate integer values
    for cs in chunk_sizes:
        _validate_positive_int(cs, "CHUNK_SIZES value")
    for co in chunk_overlaps:
        _validate_positive_int(co, "CHUNK_OVERLAPS value")
    _validate_positive_int(retrieval_top_k, "RETRIEVAL_TOP_K")
    _validate_positive_int(max_new_tokens, "MAX_NEW_TOKENS")
    _validate_positive_int(dataset_sample_size, "DATASET_SAMPLE_SIZE")

    non_semantic_strategies = [
        strategy for strategy in chunking_strategies if strategy != "semantic"
    ]
    if non_semantic_strategies:
        # Validate chunk_overlap < chunk_size for combinations that use those
        # values. Semantic chunking ignores both fields.
        for cs in chunk_sizes:
            for co in chunk_overlaps:
                if co >= cs:
                    raise ValueError(
                        f"chunk_overlap ({co}) must be less than chunk_size ({cs})"
                    )

    # Parse provider prefixes for LLM models
    llm_parsed = [parse_model_id(m) for m in llm_models]

    configs: list[BenchmarkConfig] = []
    for (
        (provider, model_name),
        emb,
        strat,
        reranker,
        tmpl,
    ) in product(
        llm_parsed, embedding_models, chunking_strategies,
        reranker_models, prompt_templates,
    ):
        for cs, co in _chunk_parameter_pairs_for_strategy(
            strat, chunk_sizes, chunk_overlaps
        ):
            configs.append(
                BenchmarkConfig(
                    llm_model=model_name,
                    llm_provider=provider,
                    embedding_model=emb,
                    chunk_size=cs,
                    chunk_overlap=co,
                    chunking_strategy=strat,
                    retrieval_top_k=retrieval_top_k,
                    retrieval_strategy=retrieval_strategy,
                    retrieval_fetch_k=retrieval_fetch_k,
                    retrieval_mmr_lambda=retrieval_mmr_lambda,
                    retrieval_use_hyde=retrieval_use_hyde,
                    max_new_tokens=max_new_tokens,
                    ollama_base_url=ollama_base_url,
                    ollama_api_key=ollama_api_key,
                    openai_compat_base_url=openai_compat_base_url,
                    openai_compat_api_key=openai_compat_api_key,
                    llm_ollama_base_url=llm_ollama_base_url,
                    llm_ollama_api_key=llm_ollama_api_key,
                    llm_openai_compat_base_url=llm_openai_compat_base_url,
                    llm_openai_compat_api_key=llm_openai_compat_api_key,
                    eval_critic_ollama_base_url=eval_critic_ollama_base_url,
                    eval_critic_ollama_api_key=eval_critic_ollama_api_key,
                    eval_critic_openai_compat_base_url=eval_critic_openai_compat_base_url,
                    eval_critic_openai_compat_api_key=eval_critic_openai_compat_api_key,
                    embedding_ollama_base_url=embedding_ollama_base_url,
                    embedding_ollama_api_key=embedding_ollama_api_key,
                    eval_critic_max_tokens=eval_critic_max_tokens,
                    dataset_name=dataset_name,
                    dataset_subset=dataset_subset,
                    dataset_sample_size=dataset_sample_size,
                    dataset_path=dataset_path,
                    dataset_question_field=dataset_question_field,
                    dataset_ground_truth_field=dataset_ground_truth_field,
                    dataset_context_field=dataset_context_field,
                    dataset_metadata_field=dataset_metadata_field,
                    eval_critic_llm=eval_critic_llm,
                    eval_critic_embedding=eval_critic_embedding,
                    custom_metrics_bert_model=custom_metrics_bert_model,
                    ragas_enabled=ragas_enabled,
                    custom_metrics_enabled=custom_metrics_enabled,
                    reranker_model=reranker,
                    reranker_top_k=reranker_top_k,
                    prompt_template=tmpl,
                    llm_answer_strip_mode=cast(AnswerStripMode, llm_answer_strip_mode),
                    llm_answer_value_fallback=llm_answer_value_fallback,
                    semantic_breakpoint_type=semantic_breakpoint_type,
                    semantic_breakpoint_amount=semantic_breakpoint_amount,
                    retrieval_mode=retrieval_mode,
                    custom_retrieval_metrics_mode=custom_retrieval_metrics_mode,
                    vector_db_backend=vector_db_backend,
                    lancedb_path=lancedb_path,
                    benchmark_stage=benchmark_stage,
                    rag_system_adapter=rag_system_adapter,
                    rag_adapter_accepts=rag_adapter_accepts,
                    rag_http_endpoint_url=rag_http_endpoint_url,
                    rag_http_timeout_seconds=rag_http_timeout_seconds,
                    rag_http_answer_field=rag_http_answer_field,
                    rag_http_contexts_field=rag_http_contexts_field,
                    rag_http_metadata_field=rag_http_metadata_field,
                    rag_http_timings_field=rag_http_timings_field,
                    rag_http_headers=rag_http_headers,
                    rag_http_auth_header=rag_http_auth_header,
                    rag_http_auth_value=rag_http_auth_value,
                )
            )
    return configs
