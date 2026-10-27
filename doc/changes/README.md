# Change Log

One file per coding session that changes behaviour, configuration, metrics,
manifests, or docs. These records explain **why** something changed, which the
git diff alone cannot.

## File name

`YYYY-MM-DD-<short-kebab-topic>.md`, e.g. `2026-10-23-mcp-metrics-and-fairness.md`.
Several sessions on one day get separate files.

## Template

```markdown
# <Title>

Date: YYYY-MM-DD

## Summary
Two or three sentences: what changed and the motivation.

## Changes
Grouped by concern. For each: what changed, why, and the files touched.

## Behaviour changes and migration
Anything that can break existing manifests, results, or workflows, and how
to adapt. Write "None" if nothing.

## Verification
Tests added or changed, test-suite result, and any manual or smoke runs
with their key numbers.

## Follow-ups
Known gaps, open questions, and deliberately deferred work.
```

## Rules

- Reference durable guides (`doc/*.md`) instead of duplicating them; update
  those guides in the same session when user-facing behaviour changes, and add
  new guides to the table in `doc/00-README.md`.
- Name config fields, metrics, and files exactly as they appear in code.
- Record results that informed a decision (e.g. smoke-run numbers), not raw logs.
