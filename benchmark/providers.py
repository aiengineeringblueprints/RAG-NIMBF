"""Provider factory — routes model creation to Ollama or OpenAI-compatible backends."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def parse_model_id(model_string: str) -> tuple[str, str]:
    """Parse a ``provider:model_name`` string.

    Returns ``(provider, model_name)``.  Unprefixed strings default to
    ``"ollama"`` for backwards compatibility.

    Examples::

        >>> parse_model_id("ollama:gemma3:4b")
        ('ollama', 'gemma3:4b')
        >>> parse_model_id("openai:Qwen/Qwen3-32B-AWQ")
        ('openai', 'Qwen/Qwen3-32B-AWQ')
        >>> parse_model_id("huggingface:BAAI/bge-small-en-v1.5")
        ('huggingface', 'BAAI/bge-small-en-v1.5')
        >>> parse_model_id("nomic-embed-text:latest")
        ('ollama', 'nomic-embed-text:latest')
    """
    if ":" not in model_string:
        return ("ollama", model_string)

    prefix, _, rest = model_string.partition(":")
    if prefix in ("ollama", "openai", "huggingface"):
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

        kwargs: dict[str, Any] = dict(
            model=model_name,
            base_url=base_url,
            streaming=False,
            num_predict=max_tokens,
            temperature=temperature,
        )
        if api_key:
            kwargs["client_kwargs"] = {"headers": {"Authorization": f"Bearer {api_key}"}}
        return ChatOllama(**kwargs)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key or "not-needed",
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )

    raise ValueError(
        f"Unknown provider '{provider}'. Supported: 'ollama', 'openai'."
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
