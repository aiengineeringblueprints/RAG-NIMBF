# Agentic MCP Capabilities: What Exists, What Is Missing

State as of this implementation round.

## Already implemented

**Execution (`benchmark/adapters/mcp.py`)**

- Tool discovery via `list_tools`; optional allowlist (`mcp_allowed_tools_json`)
  validated at connect time; runtime rejection of disallowed tool calls.
- Multi-round loop: model ↔ tools until the model answers or
  `mcp_max_agent_rounds` is hit; tool results fed back as `ToolMessage`s.
- Any discovered tool is callable (not just the fixed tool); static
  `mcp_tool_arguments_json` are merged under model arguments; `top_k`
  default injected for the primary tool.
- Transport-level resilience: retries with exponential backoff, timeouts,
  continue-on-error per sample, persistent session reused across samples.

**Per-sample accounting (diagnostics, one row per sample in exports)**

- `agent_rounds`, `agent_round_log` (which tools per round),
  `agent_tool_calls`, `agent_tools_used`, `agent_retrieved`,
  `agent_rounds_exhausted`, `agent_tokens_input/output/total`,
  `agent_model_seconds`, plus every raw tool call with redacted arguments,
  attempts, per-attempt timings and errors.

**Run-level metrics (MLflow `adapter_agent_*`, summary JSON/CSV)**

- `agent_rounds_mean/max`, `agent_tool_calls_mean/max`,
  `agent_rounds_exhausted_count`, `agent_no_retrieval_count`,
  `agent_error_count`, `agent_tokens_mean/total`,
  `agent_model_seconds` vs `agent_tool_seconds` split.

**Guardrails**

- Fairness validation for internal-vs-MCP comparisons (shared corpus,
  dataset, top-k, generator, prompt, token limit).
- Failed samples are recorded with provenance, never silently dropped or
  answered from memory by the framework.

## Missing / crucial next steps (ordered by value)

1. ~~**Agent system-prompt control.**~~ **Done.** `mcp_agent_system_prompt`
   (YAML) / `MCP_AGENT_SYSTEM_PROMPT` (.env) overrides the hardcoded
   tool-use instruction — prompting strategies are now benchmarkable.
2. ~~**No-retrieval handling policy.**~~ **Partially done.**
   `mcp_agent_require_retrieval: true` / `MCP_AGENT_REQUIRE_RETRIEVAL=true`
   fails samples where the agent answers without any tool call (recorded as
   `agent_skipped_retrieval`). A softer "inject one forced retrieval" mode
   remains future work.
3. ~~**Cost-adjusted quality metrics.**~~ **Done.** Per-sample
   `agent_tokens_total` and `agent_rounds` now flow into the comparison
   machinery (`comparisons.json/md` and `python -m benchmark.compare`), so
   token/round deltas between an agentic config and the baseline get the
   same bootstrap CI and real/noise verdict as quality metrics.
4. **Per-tool selection accuracy.** With multi-tool servers, compare
   `agent_tool_call_counts` against gold expectations (which store *should*
   answer) — needs a dataset annotation field.
5. **Streaming/parallel tool calls.** A round's multiple tool calls execute
   sequentially; parallel invocation would change latency results for
   multi-tool rounds.
6. **Conversation truncation.** Long agentic traces re-send the full message
   history each round; for expensive models a context-budget/truncation
   policy matters and is not implemented.
7. **Agentic fairness checks.** Fairness validation covers corpus/dataset/
   generator, but not `mcp_max_agent_rounds` parity when comparing agentic
   runs — document or enforce it.
