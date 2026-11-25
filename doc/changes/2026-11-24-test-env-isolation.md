# Test suite isolated from the local .env

Date: 2026-11-24

## Summary
`config.get_all_combinations()` / `get_env_combinations()` call `load_dotenv()`,
so a machine-local `.env` (and shell-exported benchmark variables) leaked into
the test suite. On the DGX Spark five tests failed only because its `.env` set
e.g. `RETRIEVAL_STRATEGY=mmr` (issue #60). The suite now runs hermetically on
any machine.

## Changes
- **Autouse fixture `isolated_benchmark_env`** (`tests/conftest.py`), replacing
  `stable_prompt_template_env`:
  - patches `config.load_dotenv` to a no-op, so config loading inside a test
    never writes `.env` values into `os.environ` (which also stopped leaking
    across tests within one session);
  - deletes every benchmark-related variable for the duration of each test.
    The key set (`BENCHMARK_ENV_KEYS`, ~180 names) is collected once at import
    from `.env.example` (including commented-out keys), the local `.env` if
    present, and every env read in `config.py` and `benchmark/**/*.py`
    (`os.getenv("…")`, `os.environ.get("…")`, `os.environ["…"]`, and helpers
    named `_env_*("…")` such as `config._env_bool`, `tracking._env_flag`);
  - keeps `PROMPT_TEMPLATES=concise` as before.
- **Regression tests** `tests/test_env_isolation.py`: `config.load_dotenv` on a
  temp `.env` does not leak; the env matrix yields code defaults
  (`retrieval_strategy == "similarity"`); a pytest subprocess started with
  `RETRIEVAL_STRATEGY` / `LLM_MODELS` exported still sees them cleared inside
  the test. Covered vars include ones read via `_env_bool`
  (`ENTERPRISE_RAG_MCP_ENABLED`) and outside `config.py`
  (`ELECTRICITY_PRICE_EUR_PER_KWH`, `benchmark/costing.py`).
- Removed `tests/test_dataset.py::_clean_local_dataset_field_env`; the global
  fixture now covers the `DATASET_*_FIELD` vars it cleared.

## Behaviour changes and migration
- Tests can no longer read real `.env` values or shell-exported benchmark
  variables. A test that needs a variable must set it explicitly with
  `monkeypatch.setenv`. No production code changed.
- Only the `config` module's `load_dotenv` is patched; other explicit
  `dotenv` calls (e.g. the Enterprise RAG plugin loading the Blueprint `.env`)
  are unaffected.

## Verification
- New tests red before the fix on the Spark (4 failed, including an
  order-dependent leak from one test into the next), green after.
- Full suite on the DGX Spark with its real `.env` in place:
  `1030 passed` (previously 5 failures, listed in #60).
- Code review (standards + spec) flagged that the first key harvest missed
  helper reads and `benchmark/` reads; extended test went red, then green.
  Final full suite on the Spark: one run with 1 intermittent failure (see
  follow-ups), then `1030 passed` in two consecutive runs.

## Follow-ups
- New env reads that use neither `os.getenv`/`os.environ` nor an `_env_*`
  helper, or that live outside `config.py`/`benchmark/`, are not cleared
  automatically; add them to `.env.example` (which is harvested too).
- `tests/test_embedding.py::TestResilientOllamaEmbeddings::test_embed_documents_no_retry_on_client_errors`
  failed once in a full run (`mock_sleep.assert_not_called`) and passed in
  isolation and in every other full run. It patches the global `time.sleep`,
  so likely a background thread from an earlier test calls it. Unrelated to
  this change; worth patching `benchmark.embedding.time.sleep` instead.
