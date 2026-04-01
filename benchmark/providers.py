"""Provider factory — routes model creation to Ollama or OpenAI-compatible backends."""

from langchain_core.language_models.chat_models import BaseChatModel


def parse_model_id(model_string: str) -> tuple[str, str]:
    """Parse a ``provider:model_name`` string.

    Returns ``(provider, model_name)``.  Unprefixed strings default to
    ``"ollama"`` for backwards compatibility.

    Examples::

        >>> parse_model_id("ollama:gemma3:4b")
        ('ollama', 'gemma3:4b')
        >>> parse_model_id("openai:Qwen/Qwen3-32B-AWQ")
        ('openai', 'Qwen/Qwen3-32B-AWQ')
        >>> parse_model_id("nomic-embed-text:latest")
        ('ollama', 'nomic-embed-text:latest')
    """
    if ":" not in model_string:
        return ("ollama", model_string)

    prefix, _, rest = model_string.partition(":")
    if prefix in ("ollama", "openai"):
        return (prefix, rest)

    # Colons in the model name itself (e.g. "gemma3:4b") → treat as ollama
    return ("ollama", model_string)


def get_chat_model(
    *,
    provider: str,
    model_name: str,
    base_url: str,
    api_key: str | None = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> BaseChatModel:
    """Create a chat model for the given provider.

    Parameters
    ----------
    provider:
        ``"ollama"`` or ``"openai"``.
    model_name:
        Model identifier understood by the backend.
    base_url:
        Base URL of the provider API.
    api_key:
        Bearer token / API key (used by OpenAI-compatible; optional for Ollama).
    max_tokens:
        Maximum tokens to generate.
    temperature:
        Sampling temperature.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance.
    """
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model_name,
            base_url=base_url,
            streaming=False,
            num_predict=max_tokens,
            temperature=temperature,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_tokens=max_tokens,
            temperature=temperature,
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: 'ollama', 'openai'."
    )
