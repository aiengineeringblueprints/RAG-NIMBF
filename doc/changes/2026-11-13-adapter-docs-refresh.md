# 2026-11-13 — Adapter documentation refresh

## What

Audit of all adapter/integration docs against the actual code (`benchmark/adapters/`, `config.py`, `benchmark/orchestration/runner.py`) and fixes for every stale or blocking finding:

- `doc/Benchmark_Your_RAG.md`: integration table now covers all five modes (HTTP, Python plugin, managed, MCP, internal) with links to the dedicated guides; HTTP request example now includes the `config.name` key that `benchmark/adapters/http.py` actually sends; added the "plugin modules load only via `RAG_ADAPTER_MODULES` env, no manifest field" warning.
- `doc/External_RAG_Python_Adapter.md`: corrected the component-injection contract — `supports_components()` is the required capability source; `RAG_ADAPTER_ACCEPTS` can only narrow, never enable injection (matches `benchmark/adapters/components.py:resolve_injection_slots`); replaced dead `main.py:199-300` reference with `benchmark/orchestration/runner.py`; added the `RAG_ADAPTER_MODULES` env-only warning.
- `doc/mcp-host-benchmarking.md`: added note that `enterprise_rag_mcp` is a plugin adapter loaded via `RAG_ADAPTER_MODULES` in `.env`.
- `doc/External_RAG_System_Usage.md`, `README.md`, `config.py` comment: `RAG_SYSTEM_ADAPTER` lists now include `mcp`, `ragflow`/`optimaiserag` and registered plugins.
- `doc/00-README.md`: index "Guides" table now lists all core integration guides (previously none of the six were listed).

## Why

The main integration guide under-represented two of five integration modes and the component-injection doc described a contract that no longer matches the code — a new external company following it would get silently empty component bundles or validation failures on plugin loading.

## Migration

Docs-only, plus one comment-only change in `config.py`. No behavior change; no migration needed.

## Verification

- Cross-checked every documented field/env var/signature against `benchmark/adapters/http.py`, `components.py`, `config.py`, and the registry in `benchmark/adapters/__init__.py`.
- Verified registered adapter names: `internal`, `http`, `mcp`, `ragflow`, `optimaiserag`, plus plugin-registered names from `RAG_ADAPTER_MODULES`.
- Verified injection policy in `components.py:resolve_injection_slots` (`set()` without `set_components`; capability ∩ user intent otherwise).
