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
    The key set (`BENCHMARK_ENV_KEYS`, ~160 names) is collected once at import
    from `.env.example` (including commented-out keys), the local `.env` if
    present, and every `os.getenv("…")` / `os.environ.get("…")` in `config.py`;
  - keeps `PROMPT_TEMPLATES=concise` as before.
- **Regression tests** `tests/test_env_isolation.py`: `config.load_dotenv` on a
  temp `.env` does not leak; the env matrix yields code defaults
  (`retrieval_strategy == "similarity"`); a pytest subprocess started with
  `RETRIEVAL_STRATEGY` / `LLM_MODELS` exported still sees them cleared inside
  the test.

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

## Follow-ups
- `tests/test_dataset.py::_clean_local_dataset_field_env` is now redundant
  with the global fixture but left in place as harmless local documentation.
