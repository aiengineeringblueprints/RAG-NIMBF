import os
import importlib
import json
import hashlib
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
    dataset_corpus_path: str | None = None
    dataset_question_field: str = "question"
    dataset_ground_truth_field: str = "ground_truth"
    dataset_context_field: str = "context"
    dataset_metadata_field: str = "metadata"
    # License metadata for document ground-truth datasets (OCR-05). Derived
    # from the dataset adapter registry so it lands in the reproducibility
    # manifest (run metadata) automatically.
    dataset_license: str | None = None
    dataset_research_only: bool = False
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
    retrieval_strategy: str = "similarity"  # "similarity" | "mmr"
    retrieval_fetch_k: int | None = None  # MMR oversampling
    retrieval_mmr_lambda: float = 0.5  # 0 = max diversity, 1 = max relevance
    retrieval_use_hyde: bool = False  # HyDE query expansion
    retrieval_multihop: bool = False  # iterative multi-hop retrieval
    retrieval_multihop_rounds: int = 1  # total retrieval rounds when enabled
    # Retrieval mode
    retrieval_mode: str = "retrieval"  # "retrieval" | "direct"
    custom_retrieval_metrics_mode: str = "heuristic"  # heuristic | gold_doc
    # Semantic chunking
    semantic_breakpoint_type: str = (
        "percentile"  # percentile | standard_deviation | interquartile
    )
    semantic_breakpoint_amount: int = 95
    vector_db_backend: str = "chroma"  # chroma | lancedb
    lancedb_path: str = ".lancedb"
    benchmark_stage: str = "all"  # all | index | query | retrieve | parsing
    # RAG system adapter
    rag_system_adapter: str = "internal"  # internal | http | mcp
    rag_adapter_accepts: str = (
        ""  # comma-separated: chunker,embedder,retriever,reranker,llm,prompt
    )
    rag_http_endpoint_url: str | None = None
    rag_http_timeout_seconds: float = 60.0
    rag_http_answer_field: str = "answer"
    rag_http_contexts_field: str = "contexts"
    rag_http_metadata_field: str = "metadata"
    rag_http_timings_field: str = "timings"
    rag_http_headers: str | None = None
    rag_http_auth_header: str | None = None
    rag_http_auth_value: str | None = None
    # Document parser adapter (OCR benchmarking, ADR-005). Empty means no
    # parser is used; otherwise a registered name or an explicit plugin
    # module/attribute pair.
    parser_adapter: str = ""
    parser_version: str | None = None
    parser_adapter_modules: tuple[str, ...] = ()
    parser_http_endpoint_url: str | None = None
    parser_http_model: str = ""
    parser_http_prompt: str = ""
    parser_http_timeout_seconds: float = 60.0
    parser_http_headers: str | None = None
    parser_plugin_module: str | None = None
    parser_plugin_attribute: str | None = None
    # Corpus ingestion knob (OCR-07). When set, RAG corpus construction
    # routes through the named parser adapter: documents under
    # dataset_corpus_path are parsed to Markdown before chunking. Requires
    # a shared-corpus dataset (jsonl-shared).
    corpus_parser: str = ""
    # MCP tool backend. In context mode the tool supplies evidence to the
    # configured generator; in answer mode the tool is the complete QA system.
    mcp_transport: str = "stdio"  # stdio | streamable_http
    mcp_server_url: str | None = None
    mcp_command: str | None = None
    mcp_args_json: str | None = None
    mcp_env_vars: str = ""
    mcp_http_headers_json: str | None = None
    mcp_tool_name: str = "search"
    mcp_question_argument: str = "query"
    mcp_top_k_argument: str | None = "top_k"
    mcp_tool_arguments_json: str | None = None
    mcp_result_mode: str = "context"  # context | answer
    mcp_result_field: str | None = None
    mcp_timeout_seconds: float = 60.0
    mcp_execution_mode: str = "fixed"  # fixed | agentic
    mcp_allowed_tools_json: str | None = None
    mcp_max_agent_rounds: int = 4
    mcp_max_retries: int = 1
    mcp_retry_backoff_seconds: float = 0.25
    mcp_continue_on_error: bool = True
    mcp_corpus_path: str | None = None
    mcp_enforce_fairness: bool = True
    # Managed external RAG lifecycle. These fields are provider-neutral; a
    # concrete adapter maps them to its API/SDK names and must reject settings
    # it cannot honor instead of silently ignoring benchmark dimensions.
    rag_managed_base_url: str | None = None
    rag_managed_api_key_env: str = "RAG_MANAGED_API_KEY"
    rag_managed_request_timeout_seconds: float = 60.0
    rag_managed_poll_interval_seconds: float = 2.0
    rag_managed_poll_timeout_seconds: float = 1800.0
    rag_managed_reuse_resources: bool = True
    rag_managed_cleanup: bool = False
    rag_managed_dataset_prefix: str = "benchmark"
    rag_managed_options_json: str | None = None
    ingestion_method: str | None = None
    ingestion_options_json: str | None = None
    retrieval_candidate_k: int = 1024
    retrieval_similarity_threshold: float = 0.2
    retrieval_vector_similarity_weight: float = 0.3
    retrieval_keyword_enabled: bool = False
    generation_context_k: int = 6
    generation_temperature: float = 0.1
    generation_top_p: float = 0.3
    generation_presence_penalty: float = 0.4
    generation_frequency_penalty: float = 0.7
    # Optional load test of the selected generator LLM
    llm_performance_enabled: bool = False
    llm_performance_call_counts: tuple[int, ...] = (1, 3, 6, 10)
    llm_performance_warmup: bool = True
    llm_performance_timeout_seconds: float = 60.0
    llm_performance_source: str = "generation"  # generation | load_test

    @property
    def name(self) -> str:
        if self.chunking_strategy == "semantic":
            parts = (
                f"semantic_bp{self.semantic_breakpoint_type}"
                f"{self.semantic_breakpoint_amount}"
                f"_{self.embedding_model}_{self.llm_model}"
                f"_{self.prompt_template}"
            )
        elif self.chunking_strategy == "provider":
            parts = (
                f"provider-{self.ingestion_method or 'default'}"
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
        if self.retrieval_multihop:
            parts += f"_mh{self.retrieval_multihop_rounds}"
        if self.retrieval_candidate_k != 1024:
            parts += f"_candidate{self.retrieval_candidate_k}"
        if self.retrieval_similarity_threshold != 0.2:
            parts += f"_threshold{self.retrieval_similarity_threshold:g}"
        if self.retrieval_vector_similarity_weight != 0.3:
            parts += f"_vector{self.retrieval_vector_similarity_weight:g}"
        if self.retrieval_keyword_enabled:
            parts += "_keyword"
        if self.generation_context_k != 6:
            parts += f"_context{self.generation_context_k}"
        if self.generation_temperature != 0.1:
            parts += f"_temp{self.generation_temperature:g}"
        if self.retrieval_mode == "direct":
            parts += "_direct"
        if self.parser_adapter:
            parts += f"_parser-{self.parser_adapter}"
        if self.vector_db_backend != "chroma":
            parts += f"_{self.vector_db_backend}"
        if self.rag_system_adapter != "internal":
            parts += f"_{self.rag_system_adapter}"
        if self.rag_system_adapter not in {"internal", "http"}:
            # Managed sweeps often vary provider-specific JSON and sampling
            # fields. Include the complete effective surface in the run key so
            # resume files and reports cannot silently collide.
            managed_dimensions = {
                "benchmark_stage": self.benchmark_stage,
                "ingestion_method": self.ingestion_method,
                "ingestion_options_json": self.ingestion_options_json,
                "managed_options_json": self.rag_managed_options_json,
                "retrieval_candidate_k": self.retrieval_candidate_k,
                "retrieval_result_k": self.retrieval_top_k,
                "retrieval_similarity_threshold": self.retrieval_similarity_threshold,
                "retrieval_vector_similarity_weight": self.retrieval_vector_similarity_weight,
                "retrieval_keyword_enabled": self.retrieval_keyword_enabled,
                "reranker_model": self.reranker_model,
                "generation_context_k": self.generation_context_k,
                "generation_temperature": self.generation_temperature,
                "generation_top_p": self.generation_top_p,
                "generation_presence_penalty": self.generation_presence_penalty,
                "generation_frequency_penalty": self.generation_frequency_penalty,
            }
            if self.rag_system_adapter == "mcp":
                managed_dimensions = {
                    "mcp_transport": self.mcp_transport,
                    "mcp_server_url": self.mcp_server_url,
                    "mcp_command": self.mcp_command,
                    "mcp_args_json": self.mcp_args_json,
                    "mcp_tool_name": self.mcp_tool_name,
                    "mcp_question_argument": self.mcp_question_argument,
                    "mcp_top_k_argument": self.mcp_top_k_argument,
                    "mcp_tool_arguments_json": self.mcp_tool_arguments_json,
                    "mcp_result_mode": self.mcp_result_mode,
                    "mcp_result_field": self.mcp_result_field,
                    "mcp_execution_mode": self.mcp_execution_mode,
                    "mcp_allowed_tools_json": self.mcp_allowed_tools_json,
                    "mcp_max_agent_rounds": self.mcp_max_agent_rounds,
                    "mcp_max_retries": self.mcp_max_retries,
                    "mcp_corpus_path": self.mcp_corpus_path,
                }
            canonical = json.dumps(
                managed_dimensions,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            parts += f"_managed-{hashlib.sha256(canonical.encode()).hexdigest()[:10]}"
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
            return self.eval_critic_openai_compat_api_key or self.openai_compat_api_key
        return self.eval_critic_ollama_api_key or self.ollama_api_key

    def embedding_base_url(self) -> str:
        """Return the base URL for embedding models."""
        return self.embedding_ollama_base_url or self.ollama_base_url

    def embedding_api_key(self) -> str | None:
        """Return the API key for embedding models."""
        return self.embedding_ollama_api_key or self.ollama_api_key


def validate_benchmark_config(config: BenchmarkConfig) -> BenchmarkConfig:
    """Validate a concrete config after env, YAML, or tracker overrides."""
    if config.benchmark_stage not in {"all", "index", "query", "retrieve", "parsing"}:
        raise ValueError("benchmark_stage must be one of: all, index, query, retrieve, parsing")
    if config.benchmark_stage == "parsing" and not config.parser_adapter and not config.parser_plugin_module:
        raise ValueError(
            "benchmark_stage=parsing requires a parser adapter "
            "(PARSER_ADAPTER or PARSER_PLUGIN_MODULE)"
        )
    if config.retrieval_mode not in {"retrieval", "direct"}:
        raise ValueError("retrieval_mode must be 'retrieval' or 'direct'")
    if config.benchmark_stage == "index" and config.retrieval_mode == "direct":
        raise ValueError("benchmark_stage=index requires retrieval_mode=retrieval")
    if config.rag_system_adapter == "http" and config.benchmark_stage in {
        "index",
        "retrieve",
    }:
        raise ValueError(
            f"benchmark_stage={config.benchmark_stage} is unavailable for the "
            "answer-only HTTP adapter"
        )
    if config.rag_system_adapter == "mcp":
        if config.benchmark_stage in {"index", "retrieve"}:
            raise ValueError(
                f"benchmark_stage={config.benchmark_stage} is unavailable for the "
                "MCP answer adapter"
            )
        if config.mcp_transport not in {"stdio", "streamable_http"}:
            raise ValueError("mcp_transport must be 'stdio' or 'streamable_http'")
        if config.mcp_result_mode not in {"context", "answer"}:
            raise ValueError("mcp_result_mode must be 'context' or 'answer'")
        if config.mcp_execution_mode not in {"fixed", "agentic"}:
            raise ValueError("mcp_execution_mode must be 'fixed' or 'agentic'")
        if not config.mcp_tool_name.strip():
            raise ValueError("mcp_tool_name must not be empty")
        if not config.mcp_question_argument.strip():
            raise ValueError("mcp_question_argument must not be empty")
        if config.mcp_transport == "stdio" and not config.mcp_command:
            raise ValueError("mcp_command is required for stdio transport")
        if config.mcp_transport == "streamable_http" and not config.mcp_server_url:
            raise ValueError(
                "mcp_server_url is required for streamable_http transport"
            )
        if config.mcp_timeout_seconds <= 0:
            raise ValueError("mcp_timeout_seconds must be positive")
        if config.mcp_max_agent_rounds <= 0:
            raise ValueError("mcp_max_agent_rounds must be positive")
        if config.mcp_max_retries < 0:
            raise ValueError("mcp_max_retries must be non-negative")
        if config.mcp_retry_backoff_seconds < 0:
            raise ValueError("mcp_retry_backoff_seconds must be non-negative")
    for value, name in (
        (config.retrieval_top_k, "retrieval_top_k"),
        (config.retrieval_candidate_k, "retrieval_candidate_k"),
        (config.generation_context_k, "generation_context_k"),
    ):
        if value is None or int(value) <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (
            config.rag_managed_request_timeout_seconds,
            "rag_managed_request_timeout_seconds",
        ),
        (config.rag_managed_poll_interval_seconds, "rag_managed_poll_interval_seconds"),
        (config.rag_managed_poll_timeout_seconds, "rag_managed_poll_timeout_seconds"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    for value, name in (
        (config.retrieval_similarity_threshold, "retrieval_similarity_threshold"),
        (
            config.retrieval_vector_similarity_weight,
            "retrieval_vector_similarity_weight",
        ),
        (config.generation_top_p, "generation_top_p"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if not 0.0 <= config.generation_temperature <= 2.0:
        raise ValueError("generation_temperature must be between 0 and 2")
    if config.chunking_strategy not in {"semantic", "provider"}:
        if config.chunk_size is None or config.chunk_overlap is None:
            raise ValueError("Non-semantic chunking requires size and overlap")
        if config.chunk_overlap >= config.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
    if config.corpus_parser:
        if config.dataset_name != "jsonl-shared":
            raise ValueError(
                "corpus_parser requires a shared-corpus dataset "
                f"(dataset_name='jsonl-shared'), got {config.dataset_name!r}"
            )
        if not config.dataset_corpus_path:
            raise ValueError(
                "corpus_parser requires dataset_corpus_path pointing at the "
                "raw documents to parse"
            )
    _validate_json_object(config.rag_managed_options_json, "rag_managed_options_json")
    _validate_json_object(config.ingestion_options_json, "ingestion_options_json")
    _validate_json_object(config.mcp_http_headers_json, "mcp_http_headers_json")
    _validate_json_object(config.mcp_tool_arguments_json, "mcp_tool_arguments_json")
    if config.mcp_allowed_tools_json is not None:
        _validate_json_string_list(
            config.mcp_allowed_tools_json, "mcp_allowed_tools_json"
        )
    if config.mcp_args_json is not None:
        try:
            mcp_args = json.loads(config.mcp_args_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"mcp_args_json must contain valid JSON: {exc}") from exc
        if not isinstance(mcp_args, list) or not all(
            isinstance(item, str) for item in mcp_args
        ):
            raise ValueError("mcp_args_json must contain a JSON array of strings")
    return config


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive non-zero integer, got {value}")


def _validate_json_object(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must contain a JSON object")


def _validate_json_string_list(value: str | None, name: str) -> None:
    if value is None:
        return
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must contain valid JSON: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError(f"{name} must contain a JSON array of strings")


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
    if strategy in {"semantic", "provider"}:
        return [(None, None)]
    return list(product(chunk_sizes, chunk_overlaps))


def get_all_combinations(
    config_path: str | Path | None = None,
) -> list[BenchmarkConfig]:
    """Return benchmark configs from .env or an optional JSON/YAML manifest.

    The manifest path can be passed explicitly or through BENCHMARK_CONFIG_FILE.
    Environment variables still provide provider URLs, API keys, and defaults for
    fields omitted by the manifest.
    """
    load_dotenv()

    manifest_env = os.getenv("BENCHMARK_CONFIG_FILE") or os.getenv(
        "EXPERIMENT_MANIFEST"
    )
    manifest_path = (
        Path(config_path)
        if config_path is not None
        else (Path(manifest_env) if manifest_env else None)
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
    embedding_models = _parse_list(
        os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest")
    )
    chunk_sizes = _parse_int_list(os.getenv("CHUNK_SIZES", "1000"), "CHUNK_SIZES")
    chunk_overlaps = _parse_int_list(
        os.getenv("CHUNK_OVERLAPS", "200"), "CHUNK_OVERLAPS"
    )
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
    dataset_corpus_path = os.getenv("DATASET_CORPUS_PATH") or None
    dataset_question_field = (
        os.getenv("DATASET_QUESTION_FIELD", "question").strip() or "question"
    )
    dataset_ground_truth_field = (
        os.getenv("DATASET_GROUND_TRUTH_FIELD", "ground_truth").strip()
        or "ground_truth"
    )
    dataset_context_field = (
        os.getenv("DATASET_CONTEXT_FIELD", "context").strip() or "context"
    )
    dataset_metadata_field = (
        os.getenv("DATASET_METADATA_FIELD", "metadata").strip() or "metadata"
    )

    # Validate dataset name against registry
    from benchmark.dataset_adapters import REGISTRY

    if dataset_name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {', '.join(sorted(REGISTRY))}"
        )
    dataset_adapter = REGISTRY[dataset_name]
    dataset_license = dataset_adapter.license
    dataset_research_only = dataset_adapter.license_research_only
    eval_critic_llm = os.getenv("EVAL_CRITIC_LLM", "gemma3:12b")
    eval_critic_embedding = os.getenv(
        "EVAL_CRITIC_EMBEDDING",
        os.getenv("EMBEDDING_MODELS", "nomic-embed-text:latest").split(",")[0].strip(),
    )
    custom_metrics_bert_model = (
        os.getenv("CUSTOM_METRICS_BERT_MODEL", "roberta-large").strip() or None
    )
    _bert_enabled = (
        os.getenv("CUSTOM_METRICS_BERTSCORE_ENABLED", "true").strip().lower()
    )
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
    eval_critic_openai_compat_base_url = (
        os.getenv("EVAL_CRITIC_OPENAI_COMPAT_BASE_URL") or None
    )
    eval_critic_openai_compat_api_key = (
        os.getenv("EVAL_CRITIC_OPENAI_COMPAT_API_KEY") or None
    )
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
    reranker_models: list[str | None] = (
        reranker_models_raw if reranker_models_raw else [None]
    )
    reranker_top_k = int(os.getenv("RERANKER_TOP_K", "3"))

    llm_answer_strip_mode = (
        os.getenv("LLM_ANSWER_STRIP_MODE", "tags_only").strip().lower()
    )
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

    # Iterative multi-hop retrieval
    _mh_raw = os.getenv("RETRIEVAL_MULTIHOP", "false").strip().lower()
    retrieval_multihop = _mh_raw in ("1", "true", "yes", "on")
    retrieval_multihop_rounds = int(os.getenv("RETRIEVAL_MULTIHOP_ROUNDS", "1"))
    _validate_positive_int(retrieval_multihop_rounds, "RETRIEVAL_MULTIHOP_ROUNDS")

    # Retrieval mode
    retrieval_mode = os.getenv("RETRIEVAL_MODE", "retrieval").strip().lower()
    if retrieval_mode not in ("retrieval", "direct"):
        raise ValueError(
            f"Invalid RETRIEVAL_MODE={retrieval_mode!r}. Use: retrieval, direct"
        )

    custom_retrieval_metrics_mode = (
        os.getenv("CUSTOM_RETRIEVAL_METRICS_MODE", "heuristic").strip().lower()
    )
    if custom_retrieval_metrics_mode not in ("heuristic", "gold_doc"):
        raise ValueError(
            "Invalid CUSTOM_RETRIEVAL_METRICS_MODE="
            f"{custom_retrieval_metrics_mode!r}. Use: heuristic, gold_doc"
        )

    # Semantic chunking
    semantic_breakpoint_type = (
        os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile").strip().lower()
    )
    if semantic_breakpoint_type not in (
        "percentile",
        "standard_deviation",
        "interquartile",
    ):
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
    if benchmark_stage not in ("all", "index", "query", "retrieve", "parsing"):
        raise ValueError(
            f"Invalid BENCHMARK_STAGE={benchmark_stage!r}. "
            "Use: all, index, query, retrieve, parsing"
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
    if rag_system_adapter == "http" and benchmark_stage == "index":
        raise ValueError(
            "BENCHMARK_STAGE=index is unavailable for the answer-only HTTP adapter; "
            "use a managed adapter with ingestion support"
        )
    rag_http_endpoint_url = os.getenv("RAG_HTTP_ENDPOINT_URL") or None
    rag_http_timeout_seconds = float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "60"))
    if rag_http_timeout_seconds <= 0:
        raise ValueError("RAG_HTTP_TIMEOUT_SECONDS must be positive")
    rag_http_answer_field = (
        os.getenv("RAG_HTTP_ANSWER_FIELD", "answer").strip() or "answer"
    )
    rag_http_contexts_field = (
        os.getenv("RAG_HTTP_CONTEXTS_FIELD", "contexts").strip() or "contexts"
    )
    rag_http_metadata_field = (
        os.getenv("RAG_HTTP_METADATA_FIELD", "metadata").strip() or "metadata"
    )
    rag_http_timings_field = (
        os.getenv("RAG_HTTP_TIMINGS_FIELD", "timings").strip() or "timings"
    )
    rag_http_headers = os.getenv("RAG_HTTP_HEADERS") or None
    rag_http_auth_header = os.getenv("RAG_HTTP_AUTH_HEADER") or None
    rag_http_auth_value = os.getenv("RAG_HTTP_AUTH_VALUE") or None
    if rag_system_adapter == "http" and not rag_http_endpoint_url:
        raise ValueError(
            "RAG_HTTP_ENDPOINT_URL is required when RAG_SYSTEM_ADAPTER=http"
        )

    parser_adapter_modules = tuple(
        _parse_list(os.getenv("PARSER_ADAPTER_MODULES", ""))
    )
    for module_name in parser_adapter_modules:
        importlib.import_module(module_name)

    from benchmark.parsing import PARSER_ADAPTER_REGISTRY

    parser_adapter = os.getenv("PARSER_ADAPTER", "").strip().lower()
    if parser_adapter and parser_adapter not in PARSER_ADAPTER_REGISTRY:
        raise ValueError(
            f"Invalid PARSER_ADAPTER={parser_adapter!r}. "
            f"Use: {', '.join(sorted(PARSER_ADAPTER_REGISTRY))}"
        )
    parser_version = os.getenv("PARSER_VERSION") or None
    parser_http_endpoint_url = os.getenv("PARSER_HTTP_ENDPOINT_URL") or None
    if parser_adapter == "http" and not parser_http_endpoint_url:
        raise ValueError(
            "PARSER_HTTP_ENDPOINT_URL is required when PARSER_ADAPTER=http"
        )
    parser_http_model = os.getenv("PARSER_HTTP_MODEL", "").strip()
    parser_http_prompt = os.getenv("PARSER_HTTP_PROMPT", "").strip()
    parser_http_timeout_seconds = float(
        os.getenv("PARSER_HTTP_TIMEOUT_SECONDS", "60")
    )
    if parser_http_timeout_seconds <= 0:
        raise ValueError("PARSER_HTTP_TIMEOUT_SECONDS must be positive")
    parser_http_headers = os.getenv("PARSER_HTTP_HEADERS") or None
    parser_plugin_module = os.getenv("PARSER_PLUGIN_MODULE") or None
    parser_plugin_attribute = os.getenv("PARSER_PLUGIN_ATTRIBUTE") or None
    if parser_plugin_module and not parser_plugin_attribute:
        raise ValueError(
            "PARSER_PLUGIN_ATTRIBUTE is required when PARSER_PLUGIN_MODULE is set"
        )
    if parser_plugin_module and parser_adapter:
        raise ValueError(
            "Set either PARSER_PLUGIN_MODULE or PARSER_ADAPTER, not both"
        )

    corpus_parser = os.getenv("CORPUS_PARSER", "").strip().lower()
    if corpus_parser and corpus_parser not in PARSER_ADAPTER_REGISTRY:
        raise ValueError(
            f"Invalid CORPUS_PARSER={corpus_parser!r}. "
            f"Use: {', '.join(sorted(PARSER_ADAPTER_REGISTRY))}"
        )
    dataset_license = os.getenv("DATASET_LICENSE") or dataset_license
    if corpus_parser:
        if dataset_name != "jsonl-shared":
            raise ValueError(
                "CORPUS_PARSER requires a shared-corpus dataset "
                f"(DATASET_NAME=jsonl-shared), got {dataset_name!r}"
            )
        if not dataset_corpus_path:
            raise ValueError(
                "CORPUS_PARSER requires DATASET_CORPUS_PATH pointing at the "
                "raw documents to parse"
            )

    mcp_transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    mcp_server_url = os.getenv("MCP_SERVER_URL") or None
    mcp_command = os.getenv("MCP_COMMAND") or None
    mcp_args_json = os.getenv("MCP_ARGS_JSON") or None
    mcp_env_vars = os.getenv("MCP_ENV_VARS", "")
    mcp_http_headers_json = os.getenv("MCP_HTTP_HEADERS_JSON") or None
    mcp_tool_name = os.getenv("MCP_TOOL_NAME", "search").strip() or "search"
    mcp_question_argument = (
        os.getenv("MCP_QUESTION_ARGUMENT", "query").strip() or "query"
    )
    mcp_top_k_argument = os.getenv("MCP_TOP_K_ARGUMENT", "top_k").strip() or None
    mcp_tool_arguments_json = os.getenv("MCP_TOOL_ARGUMENTS_JSON") or None
    mcp_result_mode = os.getenv("MCP_RESULT_MODE", "context").strip().lower()
    mcp_result_field = os.getenv("MCP_RESULT_FIELD") or None
    mcp_timeout_seconds = float(os.getenv("MCP_TIMEOUT_SECONDS", "60"))
    mcp_execution_mode = os.getenv("MCP_EXECUTION_MODE", "fixed").strip().lower()
    mcp_allowed_tools_json = os.getenv("MCP_ALLOWED_TOOLS_JSON") or None
    mcp_max_agent_rounds = int(os.getenv("MCP_MAX_AGENT_ROUNDS", "4"))
    mcp_max_retries = int(os.getenv("MCP_MAX_RETRIES", "1"))
    mcp_retry_backoff_seconds = float(
        os.getenv("MCP_RETRY_BACKOFF_SECONDS", "0.25")
    )
    mcp_continue_on_error = _env_bool("MCP_CONTINUE_ON_ERROR", True)
    mcp_corpus_path = os.getenv("MCP_CORPUS_PATH") or None
    mcp_enforce_fairness = _env_bool("MCP_ENFORCE_FAIRNESS", True)
    if rag_system_adapter == "mcp":
        if mcp_transport not in {"stdio", "streamable_http"}:
            raise ValueError("MCP_TRANSPORT must be 'stdio' or 'streamable_http'")
        if mcp_result_mode not in {"context", "answer"}:
            raise ValueError("MCP_RESULT_MODE must be 'context' or 'answer'")
        if mcp_execution_mode not in {"fixed", "agentic"}:
            raise ValueError("MCP_EXECUTION_MODE must be 'fixed' or 'agentic'")
        if mcp_transport == "stdio" and not mcp_command:
            raise ValueError("MCP_COMMAND is required for stdio transport")
        if mcp_transport == "streamable_http" and not mcp_server_url:
            raise ValueError(
                "MCP_SERVER_URL is required for streamable_http transport"
            )
        if mcp_timeout_seconds <= 0:
            raise ValueError("MCP_TIMEOUT_SECONDS must be positive")
        if mcp_max_agent_rounds <= 0:
            raise ValueError("MCP_MAX_AGENT_ROUNDS must be positive")
        if mcp_max_retries < 0:
            raise ValueError("MCP_MAX_RETRIES must be non-negative")
        if mcp_retry_backoff_seconds < 0:
            raise ValueError("MCP_RETRY_BACKOFF_SECONDS must be non-negative")
    _validate_json_object(mcp_http_headers_json, "MCP_HTTP_HEADERS_JSON")
    _validate_json_object(mcp_tool_arguments_json, "MCP_TOOL_ARGUMENTS_JSON")
    _validate_json_string_list(mcp_allowed_tools_json, "MCP_ALLOWED_TOOLS_JSON")
    if mcp_args_json is not None:
        try:
            parsed_mcp_args = json.loads(mcp_args_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"MCP_ARGS_JSON must contain valid JSON: {exc}") from exc
        if not isinstance(parsed_mcp_args, list) or not all(
            isinstance(item, str) for item in parsed_mcp_args
        ):
            raise ValueError("MCP_ARGS_JSON must contain a JSON array of strings")

    rag_managed_base_url = os.getenv("RAG_MANAGED_BASE_URL") or None
    rag_managed_api_key_env = (
        os.getenv("RAG_MANAGED_API_KEY_ENV", "RAG_MANAGED_API_KEY").strip()
        or "RAG_MANAGED_API_KEY"
    )
    rag_managed_request_timeout_seconds = float(
        os.getenv("RAG_MANAGED_REQUEST_TIMEOUT_SECONDS", "60")
    )
    rag_managed_poll_interval_seconds = float(
        os.getenv("RAG_MANAGED_POLL_INTERVAL_SECONDS", "2")
    )
    rag_managed_poll_timeout_seconds = float(
        os.getenv("RAG_MANAGED_POLL_TIMEOUT_SECONDS", "1800")
    )
    for value, name in (
        (rag_managed_request_timeout_seconds, "RAG_MANAGED_REQUEST_TIMEOUT_SECONDS"),
        (rag_managed_poll_interval_seconds, "RAG_MANAGED_POLL_INTERVAL_SECONDS"),
        (rag_managed_poll_timeout_seconds, "RAG_MANAGED_POLL_TIMEOUT_SECONDS"),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    rag_managed_reuse_resources = _env_bool("RAG_MANAGED_REUSE_RESOURCES", True)
    rag_managed_cleanup = _env_bool("RAG_MANAGED_CLEANUP", False)
    rag_managed_dataset_prefix = (
        os.getenv("RAG_MANAGED_DATASET_PREFIX", "benchmark").strip() or "benchmark"
    )
    rag_managed_options_json = os.getenv("RAG_MANAGED_OPTIONS_JSON") or None
    ingestion_method = os.getenv("INGESTION_METHOD") or None
    ingestion_options_json = os.getenv("INGESTION_OPTIONS_JSON") or None
    _validate_json_object(rag_managed_options_json, "RAG_MANAGED_OPTIONS_JSON")
    _validate_json_object(ingestion_options_json, "INGESTION_OPTIONS_JSON")
    retrieval_candidate_k = int(os.getenv("RETRIEVAL_CANDIDATE_K", "1024"))
    retrieval_similarity_threshold = float(
        os.getenv("RETRIEVAL_SIMILARITY_THRESHOLD", "0.2")
    )
    retrieval_vector_similarity_weight = float(
        os.getenv("RETRIEVAL_VECTOR_SIMILARITY_WEIGHT", "0.3")
    )
    retrieval_keyword_enabled = _env_bool("RETRIEVAL_KEYWORD_ENABLED", False)
    generation_context_k = int(os.getenv("GENERATION_CONTEXT_K", "6"))
    generation_temperature = float(os.getenv("GENERATION_TEMPERATURE", "0.1"))
    generation_top_p = float(os.getenv("GENERATION_TOP_P", "0.3"))
    generation_presence_penalty = float(os.getenv("GENERATION_PRESENCE_PENALTY", "0.4"))
    generation_frequency_penalty = float(
        os.getenv("GENERATION_FREQUENCY_PENALTY", "0.7")
    )
    _validate_positive_int(retrieval_candidate_k, "RETRIEVAL_CANDIDATE_K")
    _validate_positive_int(generation_context_k, "GENERATION_CONTEXT_K")
    if not 0.0 <= retrieval_similarity_threshold <= 1.0:
        raise ValueError("RETRIEVAL_SIMILARITY_THRESHOLD must be between 0 and 1")
    if not 0.0 <= retrieval_vector_similarity_weight <= 1.0:
        raise ValueError("RETRIEVAL_VECTOR_SIMILARITY_WEIGHT must be between 0 and 1")
    if not 0.0 <= generation_temperature <= 2.0:
        raise ValueError("GENERATION_TEMPERATURE must be between 0 and 2")
    if not 0.0 <= generation_top_p <= 1.0:
        raise ValueError("GENERATION_TOP_P must be between 0 and 1")

    # Integrated LLM load/performance benchmark
    llm_performance_enabled = _env_bool("LLM_PERFORMANCE_ENABLED", False)
    llm_performance_call_counts = tuple(
        _parse_int_list(
            os.getenv("LLM_PERFORMANCE_CALL_COUNTS", "1,3,6,10"),
            "LLM_PERFORMANCE_CALL_COUNTS",
        )
    )
    llm_performance_warmup = _env_bool("LLM_PERFORMANCE_WARMUP", True)
    llm_performance_timeout_seconds = float(
        os.getenv("LLM_PERFORMANCE_TIMEOUT_SECONDS", "60")
    )
    llm_performance_source = (
        os.getenv("LLM_PERFORMANCE_SOURCE", "generation").strip().lower()
    )
    if llm_performance_source not in ("generation", "load_test"):
        raise ValueError("LLM_PERFORMANCE_SOURCE must be 'generation' or 'load_test'")
    for count in llm_performance_call_counts:
        _validate_positive_int(count, "LLM_PERFORMANCE_CALL_COUNTS value")
    if llm_performance_timeout_seconds <= 0:
        raise ValueError("LLM_PERFORMANCE_TIMEOUT_SECONDS must be positive")

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
        llm_parsed,
        embedding_models,
        chunking_strategies,
        reranker_models,
        prompt_templates,
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
                    retrieval_multihop=retrieval_multihop,
                    retrieval_multihop_rounds=retrieval_multihop_rounds,
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
                    dataset_corpus_path=dataset_corpus_path,
                    dataset_question_field=dataset_question_field,
                    dataset_ground_truth_field=dataset_ground_truth_field,
                    dataset_context_field=dataset_context_field,
                    dataset_metadata_field=dataset_metadata_field,
                    dataset_license=dataset_license,
                    dataset_research_only=dataset_research_only,
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
                    parser_adapter=parser_adapter,
                    parser_version=parser_version,
                    parser_adapter_modules=parser_adapter_modules,
                    parser_http_endpoint_url=parser_http_endpoint_url,
                    parser_http_model=parser_http_model,
                    parser_http_prompt=parser_http_prompt,
                    parser_http_timeout_seconds=parser_http_timeout_seconds,
                    parser_http_headers=parser_http_headers,
                    parser_plugin_module=parser_plugin_module,
                    parser_plugin_attribute=parser_plugin_attribute,
                    corpus_parser=corpus_parser,
                    mcp_transport=mcp_transport,
                    mcp_server_url=mcp_server_url,
                    mcp_command=mcp_command,
                    mcp_args_json=mcp_args_json,
                    mcp_env_vars=mcp_env_vars,
                    mcp_http_headers_json=mcp_http_headers_json,
                    mcp_tool_name=mcp_tool_name,
                    mcp_question_argument=mcp_question_argument,
                    mcp_top_k_argument=mcp_top_k_argument,
                    mcp_tool_arguments_json=mcp_tool_arguments_json,
                    mcp_result_mode=mcp_result_mode,
                    mcp_result_field=mcp_result_field,
                    mcp_timeout_seconds=mcp_timeout_seconds,
                    mcp_execution_mode=mcp_execution_mode,
                    mcp_allowed_tools_json=mcp_allowed_tools_json,
                    mcp_max_agent_rounds=mcp_max_agent_rounds,
                    mcp_max_retries=mcp_max_retries,
                    mcp_retry_backoff_seconds=mcp_retry_backoff_seconds,
                    mcp_continue_on_error=mcp_continue_on_error,
                    mcp_corpus_path=mcp_corpus_path,
                    mcp_enforce_fairness=mcp_enforce_fairness,
                    rag_managed_base_url=rag_managed_base_url,
                    rag_managed_api_key_env=rag_managed_api_key_env,
                    rag_managed_request_timeout_seconds=rag_managed_request_timeout_seconds,
                    rag_managed_poll_interval_seconds=rag_managed_poll_interval_seconds,
                    rag_managed_poll_timeout_seconds=rag_managed_poll_timeout_seconds,
                    rag_managed_reuse_resources=rag_managed_reuse_resources,
                    rag_managed_cleanup=rag_managed_cleanup,
                    rag_managed_dataset_prefix=rag_managed_dataset_prefix,
                    rag_managed_options_json=rag_managed_options_json,
                    ingestion_method=ingestion_method,
                    ingestion_options_json=ingestion_options_json,
                    retrieval_candidate_k=retrieval_candidate_k,
                    retrieval_similarity_threshold=retrieval_similarity_threshold,
                    retrieval_vector_similarity_weight=retrieval_vector_similarity_weight,
                    retrieval_keyword_enabled=retrieval_keyword_enabled,
                    generation_context_k=generation_context_k,
                    generation_temperature=generation_temperature,
                    generation_top_p=generation_top_p,
                    generation_presence_penalty=generation_presence_penalty,
                    generation_frequency_penalty=generation_frequency_penalty,
                    llm_performance_enabled=llm_performance_enabled,
                    llm_performance_call_counts=llm_performance_call_counts,
                    llm_performance_warmup=llm_performance_warmup,
                    llm_performance_timeout_seconds=llm_performance_timeout_seconds,
                    llm_performance_source=llm_performance_source,
                )
            )
    return configs
