# ADR-002: YAML-first configuration; manifest required at the CLI

**Status:** Accepted

## Context

Workflow settings (models, chunking, retrieval, datasets, evaluator) were
spread across `experiments/*.yaml` manifests and a legacy `.env`-driven
matrix expansion. Two sources of truth meant every new knob needed both a
YAML field and an env fallback, and experiments were not reproducible from
the repository alone (env state lived on the machine, not in git).

## Decision

YAML-first, YAML-only for workflow settings:

- Experiment manifests (`experiments/*.yaml`) are the required entry point.
  `main.py` is a thin CLI that refuses to run without a manifest.
- Secrets and machine-local values (API keys, host URLs) stay in `.env`
  only and are never committed or embedded in manifests.
- The legacy `.env` matrix expansion is removed.
- Every config knob is a field on `BenchmarkConfig` in `config.py` with a
  test in `tests/test_config.py`.

## Consequences

- One source of truth per experiment; results are reproducible from the
  manifest in git plus a machine-local `.env`.
- Env vars survive only for secrets/local URLs (and smoke-test toggles
  such as `RAGAS_ENABLED`), documented in the README env-var table.
- Future architecture reviews must not re-propose the env-driven matrix or
  optional-manifest operation.
