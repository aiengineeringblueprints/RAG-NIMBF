# Repository Cleanup and Discoverability (issue #67)

Date: 2026-11-25

## Summary

Repository-structure cleanup per issue #67: consolidated the duplicated
`doc/` vs `docs/` directories, archived outdated/unreferenced documents into
`doc/archive/`, moved stray root-level scripts into `scripts/`/`tools/`,
removed tracked scratch files, extended `.gitignore`, and rewrote
`doc/00-README.md` as a status-marked documentation index plus a repository
layout / development section in the root README. No benchmark behaviour is
changed.

## Changes

- **`docs/` → `doc/agents/`**: the three agent guides moved into the single
  documentation tree. Note: the root `AGENTS.md` is gitignored and not
  tracked — its path references were updated in the local working copy only
  (including the `docs/adr/` → `doc/adr/` typo fix); other clones keep their
  local AGENTS.md. The tracked `doc/agents/domain.md` was corrected in this
  PR.
- **`doc/archive/` (new)**: superseded documents with ARCHIVED headers and a
  README table mapping each to its replacement:
  `prompts.md` (root prompt draft → `benchmark/prompt_templates`),
  `EVAL_MATRIX.md` (German planning matrix → `experiments/*.yaml`),
  `doc/fixes.md` (report against the removed single-file `main.py` loop →
  ADR-003), `doc/info.md` (old MLflow-viewer scratch notes →
  `scripts/view_mlflow_runs.py --help`).
- **Root-level strays**: `build_results_report.py` → `scripts/`;
  `helper_functions/view_ground_truths.py` → `tools/` (directory removed).
- **Removed tracked scratch**: `.scratch/*` (completed planning specs; ADRs
  are the durable record) and `tmp/replay.py`; both patterns gitignored.
- **`.gitignore`**: added `/.scratch/`, `/.vscode/`, `/tmp/` under a new
  "Scratch / private working files" section.
- **`doc/00-README.md`**: rewritten as the documentation index with a status
  legend (current / experimental / historical), grouped sections (start
  here, running benchmarks, integrating external systems, results,
  ADRs/agent guides), the previously unindexed guides added
  (`implementation-overview.md`, `experiment-todos.md`, `clearml_usage.md`,
  documents 09–12), and the May 2026 gap analyses explicitly marked
  historical.
- **`Optimizations/README.md` (new)**: marks the research notes as archived
  and indexes each note by topic.
- **Root `README.md`**: added "Repository Layout" and "Development and
  Testing" sections (links to the tracked `doc/agents/` guides; the
  gitignored local `AGENTS.md` is not linked from the README).

## Behaviour changes and migration

None for the framework. Paths that changed:

- `docs/agents/*.md` → `doc/agents/*.md`
- `helper_functions/view_ground_truths.py` → `tools/view_ground_truths.py`
- `build_results_report.py` → `scripts/build_results_report.py`

## Verification

- Full pytest suite green (see PR body for the recorded result).
- `python -m benchmark.worker plan experiments/full-grid-example.yaml` smoke
  pass.
- Reference check before every move/delete: greps over `benchmark/`, `tests/`,
  `scripts/`, `experiments/`, `doc/`, manifests and CI for the moved/removed
  names — the only hits were the AGENTS.md paths updated above.

## Follow-ups

Open cleanup candidates deliberately retained (status unclear or
research-relevant): `EVAL_MATRIX.md`-style parameter studies may still be
referenced by the thesis work; `doc/superpowers/` (plans/specs from the
agent workflow) left untouched; sandbox subprojects
(`NGEN-AI/`, `LLM_Performance_Tests-main/`, `hallucination_eval/`) are
untracked local clones and out of scope.

### Review follow-up (same PR)

- `scripts/backfill_mlflow.py`: `project_root` fixed to
  `Path(__file__).resolve().parent.parent`; matrix path now
  `doc/archive/eval-matrix.md` (the script was broken before — it looked for
  root `EVAL_MATRIX.md` relative to `scripts/`, so `results/` also resolved
  wrong).
- `doc/agents/domain.md`: all `docs/adr/` references corrected to
  `doc/adr/` (pure-rename fallout missed in the first pass).
- Stale references to moved files annotated at their historical locations:
  `doc/09-framework-status-assessment.md` (2× EVAL_MATRIX),
  `doc/03-new-functionalities.md` (prompts.md),
  `Optimizations/full-project-audit-report.md` (helper_functions path).
