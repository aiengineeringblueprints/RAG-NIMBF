# ADR-006: Regression criteria are an analysis policy, not a run knob

**Status:** Accepted

## Context

ADR-002 requires every workflow setting to be a field on `BenchmarkConfig`
with a test in `tests/test_config.py`. The Blueprint regression benchmark
(issue #64, `benchmark/regression/`) adds settings that decide *how two
finished runs are judged*: the primary metric order, the minimum improvement,
the trade-off guardrails and the bootstrap settings (`n_boot`, `seed`).

These settings never change how a run executes. Putting them on
`BenchmarkConfig` would make them part of every run's config (and its config
hash), so changing a guardrail would look like a different experiment and
old runs could not be re-judged under a new policy.

## Decision

- The comparison policy lives in its own versioned YAML file,
  `experiments/blueprint-regression/criteria.yaml`, passed to
  `python -m benchmark.regression compare --criteria`.
- It is loaded into a typed, validated record
  (`benchmark.regression.criteria.RegressionCriteria` with `Guardrail`
  entries). Unknown keys, unknown guardrail limits and wrong types fail
  loudly; tests live in `tests/test_regression_benchmark.py`.
- The bootstrap settings (`n_boot`, `seed`) are part of that file, not CLI
  flags, so a verdict is reproducible from git alone.
- The comparison report records the loaded criteria next to the verdict.

## Consequences

- ADR-002 still holds for everything a run does; this is the one documented
  exception for offline analysis policy.
- Changing the policy is a reviewed change to `criteria.yaml`; existing runs
  can be re-judged without re-running them.
