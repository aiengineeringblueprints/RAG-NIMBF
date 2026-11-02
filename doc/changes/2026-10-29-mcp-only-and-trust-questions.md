# Retrieval arms, trust questions and decline judge for the Blueprint MCP host benchmark

Date: 2026-10-29

## Summary
The Blueprint benchmark now compares four arms, RAG only, RAG + MCP,
retrieval as a tool + MCP and MCP only, and has a second question set (`questions_trust.jsonl`, 210
questions) for freshness, permission, robustness and unanswerable cases. The
core set showed that MCP adds data the documents lack; it did not show
whether the system prefers live data, respects access boundaries, resists
hostile or failing tools, or declines instead of guessing.

## Changes
- Retrieval arms: `enterprise_rag_mcp_retrieval` / `ENTERPRISE_RAG_MCP_RETRIEVAL`
  (`context` | `tool` | `none`) sends the Blueprint's new `retrieval` field
  (`config.py`, `examples/enterprise_rag_plugin/mcp_adapter.py`). MCP off
  with another mode is rejected; in a matrix it collapses to `context`
  (`benchmark/orchestration/matrix.py`), so `enabled × retrieval` yields four
  runs. `tool` compares pre-retrieval with just-in-time retrieval over the
  same retriever (doc 12 §4.2, §13.2).
- Decline judge: `enterprise_rag_mcp_decline_judge` lets the critic LLM judge
  declines (`make_decline_judge`); metrics `declined_judge_rate` and
  `decline_judge_agreement_rate` validate the phrase list (AbstentionBench
  scores abstention with an LLM judge).
- Tool-call injection: `forbidden_tools` label, `forbidden_tool_call_rate`
  (`benchmark/mcp_host.py`); the trust sub-type `injection_toolcall` asks the
  model, inside a review comment, to call `get_invoice_balance` and leak the
  balance (InjecAgent / AgentDojo style).
- Blueprint trace: plain aliases parsed in both formats, `pin_denied_count`,
  token usage (`input_tokens_*`, `output_tokens_*`, `model_calls_*`).
- Scoring: `benchmark/answer_match.py::score_answer` reads
  `expected_behavior` (`answer` | `decline`) and `forbidden_patterns`, and
  returns `answer_match`, `declined` and `forbidden_match`. The adapter uses
  it instead of its own pattern check.
- Metrics: `aggregate_by_category` adds `declined_rate`,
  `forbidden_match_count` and `forbidden_match_rate`, overall and per
  category (`benchmark/mcp_host.py`).
- Dataset: `trust_data.py`, `trust_mcp_server.py`, `generate_trust.py`,
  `questions_trust.jsonl`, `blueprint_mcp_trust_servers.template.json` in
  `datasets/blueprint_mcp_host/`; manifest
  `experiments/enterprise-rag-mcp-trust.yaml`. Core corpus, questions and
  server are unchanged.

## Behaviour changes and migration
- Every Blueprint request now carries `retrieval` (`context` unless
  configured). Blueprints before 2026-10-29 ignore it, so `tool`/`none` runs
  against them silently still preload; update the Blueprint first.
- The core manifest runs four arms instead of two (twice the Blueprint
  calls). Remove `enterprise_rag_mcp_retrieval` from its matrix to go back.
- `injection_feedback` is replaced by `injection_toolcall` in
  `questions_trust.jsonl`.
- Diagnostics now always contain `declined`; samples with forbidden patterns
  contain `forbidden_match`. Existing `answer_match` values are unchanged for
  samples without the new labels.

## Verification
- New: `tests/test_blueprint_mcp_trust_dataset.py` (16 tests: drift,
  labels against corpus and servers, budgets against the template, real
  stdio servers, tool-name length), tests in `tests/test_answer_match.py`,
  `tests/test_mcp_host.py`, `tests/test_enterprise_rag_mcp_adapter.py`,
  `tests/test_config.py`. Full suite green.
- Smoke check through the Blueprint's own `ClientManager` with the
  generated trust config and identity `General`: `kestrel_finance` connect
  denied, injection result delivered, warranty tool error, service history
  rejected by the size limit, calibration call unavailable after 8.0 s,
  unknown order returns a tool error.
- `python -m benchmark.worker plan` expands both manifests to four configs
  (800 and 840 questions).
- Smoke run (20 trust questions, Qwen3.6-35B, before the alias fix): 15 of
  18 calls in RAG + MCP failed as unknown capabilities, because the model
  dropped the digest from hashed aliases. Fixed in the Blueprint (plain
  `<server>__<tool>` aliases); see the follow-up smoke run below.

## Follow-ups
- Live runs of both manifests (see `doc/experiment-todos.md` §7).
- Validate the phrase list against the decline judge on a full run
  (`decline_judge_agreement_rate`).
- Permission is tested at server level only; document-level tags need the
  adapter to index part of the corpus with another role.
- A Finance-identity control run (roles `General,Finance`) would show that
  the finance tools work when allowed; the current labels expect a decline.
