# Providers and Models

Sources:

- [benchmark/providers.py](../benchmark/providers.py)
- [benchmark/embedding.py](../benchmark/embedding.py)
- [benchmark/generation.py](../benchmark/generation.py)

Provider parsing:

- `parse_model_id()` separates provider prefix from model name.
- Ollama is the default local provider.
- OpenAI-compatible chat endpoints are supported through configured base URL and API key.

Model usage roles:

- Generator LLM: answer generation in [[Generation Layer]].
- Critic LLM: RAGAS evaluation in [[Evaluation and Metrics]].
- Embedding model: vector store, semantic chunking, and custom metric relevance.
- Agent model: autonomous proposer/analyzer brain in [[Agentic Runner]].

Important wrappers:

- `get_chat_model()` creates chat model instances.
- `wrap_for_ragas()` adapts chat output so RAGAS sees content as strings.
- `_ContentAsStringChatModel` normalizes content for downstream compatibility.

Configuration:

- See [[Configuration Reference]] for shared and per-role base URL/API key fallback order.

