from benchmark.adapters.components import ComponentBundle


def test_bundle_defaults_all_none():
    b = ComponentBundle()
    assert b.chunker is None
    assert b.embedder is None
    assert b.retriever_factory is None
    assert b.reranker is None
    assert b.llm is None
    assert b.prompt_template is None


def test_bundle_accepts_fields():
    b = ComponentBundle(prompt_template="Answer: {context}")
    assert b.prompt_template == "Answer: {context}"
    assert b.llm is None
