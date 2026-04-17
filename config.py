import os
from dataclasses import dataclass
from itertools import product
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
    chunk_size: int
    chunk_overlap: int
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
    # Semantic chunking
    semantic_breakpoint_type: str = "percentile"    # percentile | standard_deviation | interquartile
    semantic_breakpoint_amount: int = 95

    @property
    def name(self) -> str:
        parts = (
            f"{self.chunking_strategy}_cs{self.chunk_size}_co{self.chunk_overlap}"
            f"_{self.embedding_model}_{self.llm_model}"
            f"_{self.prompt_template}"
        )
        if self.reranker_model:
            parts += f"_rerank-{self.reranker_model}"
        if self.retrieval_strategy != "similarity":
            parts += f"_mmr-l{self.retrieval_mmr_lambda}"
        if self.retrieval_use_hyde:
            parts += "_hyde"
        if self.retrieval_mode == "direct":
            parts += "_direct"
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


def _validate_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be a positive non-zero integer, got {value}")


def get_all_combinations() -> list[BenchmarkConfig]:
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

    # Semantic chunking
    semantic_breakpoint_type = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile").strip().lower()
    if semantic_breakpoint_type not in ("percentile", "standard_deviation", "interquartile"):
        raise ValueError(
            f"Invalid SEMANTIC_BREAKPOINT_TYPE={semantic_breakpoint_type!r}. "
            "Use: percentile, standard_deviation, interquartile"
        )
    semantic_breakpoint_amount = int(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
    _validate_positive_int(semantic_breakpoint_amount, "SEMANTIC_BREAKPOINT_AMOUNT")

    # Validate integer values
    for cs in chunk_sizes:
        _validate_positive_int(cs, "CHUNK_SIZES value")
    for co in chunk_overlaps:
        _validate_positive_int(co, "CHUNK_OVERLAPS value")
    _validate_positive_int(retrieval_top_k, "RETRIEVAL_TOP_K")
    _validate_positive_int(max_new_tokens, "MAX_NEW_TOKENS")
    _validate_positive_int(dataset_sample_size, "DATASET_SAMPLE_SIZE")

    # Validate chunk_overlap < chunk_size for every combination
    for cs in chunk_sizes:
        for co in chunk_overlaps:
            if co >= cs:
                raise ValueError(
                    f"chunk_overlap ({co}) must be less than chunk_size ({cs})"
                )

    # Parse provider prefixes for LLM models
    llm_parsed = [parse_model_id(m) for m in llm_models]

    combos = list(product(
        llm_parsed, embedding_models, chunk_sizes, chunk_overlaps, chunking_strategies,
        reranker_models, prompt_templates,
    ))

    return [
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
            eval_critic_llm=eval_critic_llm,
            eval_critic_embedding=eval_critic_embedding,
            reranker_model=reranker,
            reranker_top_k=reranker_top_k,
            prompt_template=tmpl,
            llm_answer_strip_mode=cast(AnswerStripMode, llm_answer_strip_mode),
            llm_answer_value_fallback=llm_answer_value_fallback,
            semantic_breakpoint_type=semantic_breakpoint_type,
            semantic_breakpoint_amount=semantic_breakpoint_amount,
            retrieval_mode=retrieval_mode,
        )
        for (provider, model_name), emb, cs, co, strat, reranker, tmpl in combos
    ]
