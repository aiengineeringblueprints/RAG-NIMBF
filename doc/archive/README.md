# Archived Documents

Historical material kept for research provenance (issue #67). Nothing in this
directory is current: do not follow instructions here without checking the
live guides in `doc/` first.

| Document | What it was | Superseded by |
|----------|-------------|---------------|
| [prompts.md](prompts.md) | Early draft of the context-only answer prompt | `benchmark/prompt_templates`, prompt settings in `experiments/*.yaml` |
| [eval-matrix.md](eval-matrix.md) | Early experiment planning matrix (German) | Manifests in `experiments/`, `doc/env-vars.md` |
| [fixes.md](fixes.md) | Deep-dive optimization report against the removed single-file `main.py` loop | ADR-003 (worker is the only orchestration loop), `doc/changes/` |
| [info.md](info.md) | Scratch notes on an old `view_mlflow_runs.py` CLI | `scripts/view_mlflow_runs.py --help`, `doc/results-and-mlflow.md` |

`fixes.md` and `info.md` were moved here from `doc/` in the issue-#67
cleanup commit; all four documents remain tracked in this directory.
