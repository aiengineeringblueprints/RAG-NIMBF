from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings


def get_embedding_model(
    model_name: str,
    base_url: str,
    api_key: str | None = None,
) -> Embeddings:
    """Create an Ollama embedding model.

    Parameters
    ----------
    model_name:
        Ollama embedding model identifier.
    base_url:
        Ollama server base URL.
    api_key:
        Optional Bearer token for authenticated Ollama endpoints.
    """
    kwargs: dict = {"model": model_name, "base_url": base_url}
    if api_key:
        kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
    return OllamaEmbeddings(**kwargs)
