"""Provider factory — routes model creation to Ollama or OpenAI-compatible backends."""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

logger = logging.getLogger(__name__)

# Recognised provider prefixes. ``vllm`` routes through the OpenAI-compat
# client pointing at vLLM's ``/v1`` endpoint and additionally enables the
# prometheus scraper in ``benchmark.vllm_metrics``.
_KNOWN_PROVIDERS = ("ollama", "openai", "huggingface", "vllm")

# ``vllm:<model>@<url>`` — the URL is the vLLM server root (no /v1 suffix).
# We accept optional whitespace and trailing slashes for ergonomics.
_VLLM_URL_RE = re.compile(r"^(?P<model>[^@]+)@(?P<url>https?://.+)$", re.IGNORECASE)


def parse_model_id(model_string: str) -> tuple[str, str]:
    """Parse a ``provider:model_name`` string.

    Returns ``(provider, model_name)``.  Unprefixed strings default to
    ``"ollama"`` for backwards compatibility.

    For ``vllm:`` the model id may carry a deployment URL via the
    ``vllm:<model>@<url>`` form. In that case the URL is **not** returned
    here (use :func:`extract_vllm_endpoint`); callers receive just the model
    name so the rest of the pipeline keeps working with plain model strings.

    Examples::

        >>> parse_model_id("ollama:gemma3:4b")
        ('ollama', 'gemma3:4b')
        >>> parse_model_id("openai:Qwen/Qwen3-32B-AWQ")
        ('openai', 'Qwen/Qwen3-32B-AWQ')
        >>> parse_model_id("huggingface:BAAI/bge-small-en-v1.5")
        ('huggingface', 'BAAI/bge-small-en-v1.5')
        >>> parse_model_id("vllm:Qwen2.5-7B-Instruct@http://localhost:8000")
        ('vllm', 'Qwen2.5-7B-Instruct@http://localhost:8000')
        >>> parse_model_id("vllm:Qwen2.5-7B-Instruct")
        ('vllm', 'Qwen2.5-7B-Instruct')
        >>> parse_model_id("nomic-embed-text:latest")
        ('ollama', 'nomic-embed-text:latest')
    """
    if ":" not in model_string:
        return ("ollama", model_string)

    prefix, _, rest = model_string.partition(":")
    if prefix in _KNOWN_PROVIDERS:
        return (prefix, rest)

    # Colons in the model name itself (e.g. "gemma3:4b") → treat as ollama
    return ("ollama", model_string)


def extract_vllm_endpoint(
    model_id: str,
    *,
    fallback_url: str | None = None,
) -> str | None:
    """Return the vLLM server URL embedded in a model id.

    Accepts both the prefixed form (``vllm:<model>@<url>``) and the bare
    model-name form (``<model>@<url>``) because :class:`config.BenchmarkConfig`
    stores the model name with the provider prefix already stripped.

    Returns ``None`` if no URL is embedded and no ``fallback_url`` is given.
    Both ``http://host:8000`` and ``http://host:8000/v1`` are normalised to
    a base URL with a ``/v1`` suffix for direct consumption by the
    OpenAI-compat client.
    """
    body = model_id
    if body.startswith("vllm:"):
        body = body[len("vllm:"):]
    match = _VLLM_URL_RE.match(body.strip())
    if match:
        return _normalize_vllm_base_url(match.group("url").rstrip("/"))
    if fallback_url:
        return _normalize_vllm_base_url(fallback_url.rstrip("/"))
    return None


def extract_vllm_model_name(model_id: str) -> str:
    """Return just the model portion of a vLLM id (strip ``@<url>``)."""
    body = model_id
    if body.startswith("vllm:"):
        body = body[len("vllm:"):]
    match = _VLLM_URL_RE.match(body.strip())
    return match.group("model").strip() if match else body.strip()


def _normalize_vllm_base_url(url: str) -> str:
    """Ensure the vLLM URL ends in ``/v1`` (no double-suffix)."""
    if url.endswith("/v1"):
        return url
    return url + "/v1"


def _strip_v1_suffix(url: str) -> str:
    if url.endswith("/v1"):
        return url[: -len("/v1")]
    return url


def _strip_vllm_url(model_name: str) -> str:
    """Return just the model portion of a ``model@url`` vLLM id."""
    if "@" not in model_name:
        return model_name
    return model_name.split("@", 1)[0].strip()


def detect_vllm_deployment(base_url: str, *, timeout_s: float = 1.5) -> bool:
    """Probe ``/v1/models`` to confirm the endpoint is a live vLLM server.

    Returns ``True`` if the probe returns any HTTP 2xx response. Any network
    error, timeout, or non-2xx is treated as "not vLLM". The probe is best
    effort — a ``False`` does not prevent the scraper from being used.
    """
    probe_url = base_url.rstrip("/")
    if not probe_url.endswith("/v1/models"):
        if probe_url.endswith("/v1"):
            probe_url = probe_url + "/models"
        elif probe_url.endswith("/models"):
            pass
        else:
            probe_url = probe_url + "/v1/models"
    try:
        req = urllib.request.Request(probe_url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
        return False
    except Exception:  # pragma: no cover - defensive
        return False


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

        kwargs: dict[str, Any] = dict(
            model=model_name,
            base_url=base_url,
            streaming=False,
            num_predict=max_tokens,
            temperature=temperature,
        )
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}

        llm = ChatOllama(**kwargs)

        # Disable thinking mode for reasoning models (Qwen3/3.5, DeepSeek-R1, etc.)
        # so the actual answer lands in message.content instead of being consumed
        # by chain-of-thought reasoning that exhausts the token budget.
        try:
            return llm.bind(think=False)
        except Exception:
            return llm

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_tokens=max_tokens,
            temperature=temperature,
            stream_usage=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    if provider == "vllm":
        # vLLM exposes an OpenAI-compatible API at ``<server>/v1``. We reuse
        # the OpenAI client pointed at the vLLM base URL. ``enable_thinking``
        # is forwarded for reasoning models (Qwen3, DeepSeek-R1, etc.) so the
        # answer lands in ``content`` instead of being consumed by CoT.
        from langchain_openai import ChatOpenAI

        # The model id may carry the deployment URL (``model@http://...``);
        # the API call needs just the model name.
        clean_model_name = _strip_vllm_url(model_name)
        return ChatOpenAI(
            model=clean_model_name,
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_tokens=max_tokens,
            temperature=temperature,
            stream_usage=True,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: {', '.join(repr(p) for p in _KNOWN_PROVIDERS)}."
    )


class _ContentAsStringChatModel(BaseChatModel):
    """Wrapper that ensures ``message.content`` is always a string.

    Some OpenAI-compatible servers (vLLM serving Qwen3) return JSON content
    that ``langchain-openai`` auto-parses into a Python dict.  RAGAS expects
    raw strings and crashes with ``OutputParserException`` / ``StringIO``
    validation errors.  This wrapper coerces any non-string content back to
    a JSON string before passing it to RAGAS.
    """

    _wrapped: BaseChatModel

    def __init__(self, wrapped: BaseChatModel) -> None:
        super().__init__()
        self._wrapped = wrapped

    @property
    def _llm_type(self) -> str:
        return getattr(self._wrapped, "_llm_type", "content-as-string-wrapper")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self._wrapped._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._coerce_result(result)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = await self._wrapped._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        return self._coerce_result(result)

    @staticmethod
    def _coerce_result(result: ChatResult) -> ChatResult:
        new_gens: list[ChatGeneration] = []
        for gen in result.generations:
            msg = gen.message
            if not isinstance(msg.content, str):
                # Serialize dicts/lists back to a JSON string
                text = json.dumps(msg.content) if isinstance(msg.content, (dict, list)) else str(msg.content)
                msg = AIMessage(
                    content=text,
                    additional_kwargs=msg.additional_kwargs,
                    response_metadata=msg.response_metadata,
                    id=msg.id,
                )
                new_gens.append(ChatGeneration(message=msg, generation_info=gen.generation_info))
            else:
                new_gens.append(gen)
        result.generations = new_gens
        return result


def wrap_for_ragas(llm: BaseChatModel) -> BaseChatModel:
    """Wrap an LLM so that ``message.content`` is always a string.

    Apply this to critic / evaluator LLMs before passing them to RAGAS.
    Generator LLMs (question answering) do not need it.
    """
    return _ContentAsStringChatModel(llm)
