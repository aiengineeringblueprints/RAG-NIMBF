# Testing and Coverage

Test folder: [tests/](../tests)

Current inspection found 12 test files and about 201 test functions under `tests/*.py`.

Coverage map:

- `test_config.py`: env parsing, validation, combinations.
- `test_dataset.py`: dataset loading/adapters.
- `test_dataset_adapters.py` may not exist; adapter coverage appears folded into dataset/config tests.
- `test_chunking.py`: splitter creation and document chunking.
- `test_retrieval.py`: vector-store/retrieval behavior.
- `test_generation.py`: answer cleanup, validation, generation helpers.
- `test_embedding.py`: embedding model factory.
- `test_providers.py`: provider parsing/chat model wrappers.
- `test_reranker.py`: reranker behavior.
- `test_evaluation.py`: RAGAS evaluation wrapper behavior.
- `test_metrics.py`: custom metrics and GPU metrics.
- `test_models.py`: reporting/result models.
- `test_prompt_templates.py`: built-in prompt template registry.

Standalone:

- `test_ragas_wikiqa.py`: separate script for RAGAS WikiQA experimentation.

Strengths:

- Config/env parsing and validation are heavily covered.
- Generation cleanup and fallback behavior have detailed tests.
- Dataset adapter transforms are tested with mocked loading.
- Retrieval tests cover cache keys, provider/dataset fingerprinting, routing, HyDE fallback behavior, and Chroma cleanup.

Thin or absent coverage:

- `benchmark/tracking.py` MLflow behavior.
- Custom metric helpers such as hit@k, nDCG, recall@k, ROUGE-L, BLEU, METEOR, and BERTScore.
- Reporting analysis, exports, terminal output, and plot generation.
- End-to-end `run_single_benchmark()` orchestration.
- Real data/vector store/model/artifact round trips; most tests are mock-heavy.

Recommended verification commands:

```bash
python -m unittest
python -m py_compile main.py config.py benchmark/*.py agentic/*.py
```

Risk-based testing guidance:

- Config/env changes: run `tests/test_config.py`.
- Dataset changes: run dataset tests and a tiny smoke benchmark.
- Retrieval/chunking changes: run chunking/retrieval tests and inspect `.chroma/` cache behavior.
- Generation cleanup changes: run generation tests with thinking/refusal/numeric cases.
- Reporting/tracking changes: run reporting/model tests and a tiny benchmark to ensure outputs still write.
- Agentic changes: run unit checks plus `python -m agentic.cli --max-iterations 1 --sample-size 1` when models are available.

Related notes:

- [[Known Risks and Gaps]]
- [[Operations Runbook]]
