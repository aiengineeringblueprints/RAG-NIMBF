# 12 — MCP in RAG: State of the Art (2024–2026)

*Research note, 2026-10-01. Deep research on the Model Context Protocol (MCP) as an interface for Retrieval-Augmented Generation: protocol, ecosystem, academic literature, industry practice, security, and what it means for this framework's MCP adapter (`benchmark/adapters/mcp.py`). Complements [doc 10](10-rag-evaluation-best-practices.md), which covers tool-call metrics (RAGAS, TruLens MCP spans) and fairness controls.*

**Evidence markers** (kept throughout so claims can be re-checked before citing):

- **[V]** — checked on the primary source (arXiv abstract page, official spec/changelog, vendor engineering blog).
- **[S]** — confirmed only via secondary coverage or search snippets (HF Papers, OpenReview, news, mirrors).
- **[U]** — unverified, ambiguous, or the author's own inference/recommendation.

Benchmark numbers belong to the model generation that was tested, mostly mid-2025 to mid-2026 models. Do not compare scores across benchmarks.

---

## 0. Key takeaways

1. **MCP does not replace RAG; it is the transport/interface layer through which agents call retrieval.**
   - Practitioners and search vendors agree on the split:
     - RAG/indexing decides *what* is retrievable.
     - MCP decides *how* an agent reaches it.
     - The agent decides *when* to retrieve.
   - Federated "just call live APIs via MCP" brings back the classic federated-search problems: no global ranking, latency set by the slowest source, uneven relevance ([LlamaIndex][li-vs], [Elastic][es-obsolete]).
2. **The protocol has matured fast and is now vendor-neutral.**
   - Spec releases: 2024-11-05, 2025-03-26, 2025-06-18, 2025-11-25, 2026-07-28.
   - The latest is the largest so far: stateless core, extensions framework, Sampling and Roots deprecated.
   - MCP moved to the Linux-Foundation-hosted **Agentic AI Foundation** on 2025-12-09.
   - OpenAI, Google, Microsoft and AWS all ship MCP clients and/or managed servers.
3. **Tool overload is the central performance problem, and retrieval is the main fix.**
   - Tool definitions can consume 50k+ tokens before the agent even reads the request.
   - Accuracy drops sharply once more than about 30–100 tools are loaded.
   - "Tool RAG" approaches (RAG-MCP, MCP-Zero, ScaleMCP, Tool Search, semantic tool discovery) cut tool tokens by 50–99% and raise selection accuracy.
4. **Hard evidence comparing MCP-RAG with pipeline RAG is thin, and the gap is real.**
   - No peer-reviewed study compares the *same* RAG system exposed via MCP against an in-process pipeline with corpus, top-k and generator held fixed.
   - Closest data points:
     - MCPGAUGE: automatic MCP access costs about 9.5% accuracy and uses 3.25–236.5× more input tokens.
     - A practitioner benchmark: MCP wins 52% vs 38% of comparisons, but with +151% latency and roughly 36× the cost. The corpora differed, so the comparison is not fair.
     - Tool-schema compression: up to +20.5 pp exact match (EM) under tight context budgets.
   - **This framework is positioned to fill exactly this gap.**
5. **Current MCP benchmarks show agents are still weak.**
   - Top scores: MCP-Universe 43.7%, Toolathlon 38.6%, MCPMark 52.6% pass@1, MCP-Bench overall 0.749.
   - LiveMCPBench attributes **about half of all failures to retrieval errors**.
   - MCP-Atlas attributes 63% of failures to "cognitive" errors (stopping early, poor synthesis) rather than tool-call errors.
6. **Security is the biggest open risk for RAG over MCP.**
   - Retrieved chunks are untrusted input: a poisoned vector store can trigger *tool calls*, not just wrong answers (RADE).
   - Tool poisoning succeeds up to 72.8% of the time (MCPTox), and more capable models are *more* vulnerable.
   - Of about 21k internet-facing servers, 91.8% of those tested lack OAuth.
   - ACL-aware retrieval and least-privilege credentials are mandatory.

---

## 1. MCP primitives and how retrieval is exposed

| Primitive | Controlled by | RAG usage |
|---|---|---|
| **Tools** | Model | Default way to expose retrieval (`search(query, top_k, filters)`). The agent decides when to retrieve. |
| **Resources** | Application/host | Static or browsable corpora, schemas, documents the user picks. Operations: `resources/list`, `resources/read`, URI templates (RFC 6570). Annotations (`audience`, `priority` 0–1, `lastModified`) guide what goes into context [V][spec-res]. |
| **Prompts** | User | Reusable RAG prompt templates; rarely used for retrieval itself. |
| **Sampling** | Server borrows the client's LLM | Was usable for server-side query rewriting or reranking. **Deprecated since 2026-07-28**; new RAG servers should not depend on it [V][chg-2607]. |

**De facto retrieval patterns:**

- **`search` + `fetch` (OpenAI convention).** ChatGPT Deep Research and custom connectors require a server to expose exactly these two tools; `fetch` takes an id returned by `search` [S][oai-dr]. This is the closest thing to a cross-vendor "RAG server" interface.
- **Structured output and resource links** (since 2025-06-18) [V][chg-2506].
  - Structured output (`structuredContent` + `outputSchema`) carries scores, ids and metadata.
  - Resource links in tool results let the client fetch full documents lazily.
  - Patterns for large datasets built on ResourceLink: [arXiv:2510.05968][rl-patterns] [S].
- **Retrieval-only vs answer-generating tools.** Many vendor servers expose both. Examples: Vectara `search_vectara` vs `ask_vectara`, Glean `search` vs `chat`, Pinecone context vs chat. This distinction matters for benchmarking (§13).

---

## 2. Protocol evolution

| Version | Highlights relevant to RAG | Source |
|---|---|---|
| **2024-11-05** | First release. Primitives: tools, resources, prompts, sampling. Transports: stdio and HTTP+SSE. | [V][chg-2503] |
| **2025-03-26** | OAuth 2.1 authorization; **Streamable HTTP** replaces HTTP+SSE; tool annotations (read-only, destructive); JSON-RPC batching (removed again in the next release). | [V][chg-2503] |
| **2025-06-18** | **Structured tool output**; servers become OAuth Resource Servers (Protected Resource Metadata) and clients must send RFC 8707 Resource Indicators; **elicitation**; **resource links**; `MCP-Protocol-Version` header; batching removed. | [V][chg-2506] |
| **2025-11-25** | Experimental **Tasks** (long-running work with polling); **tool calling inside sampling**; URL-mode elicitation; Client ID Metadata Documents (CIMD); JSON Schema 2020-12 as default; input-validation failures returned as tool errors so the model can self-correct; SDK tiering. | [V][chg-2511], [S][workos] |
| **2026-07-28** (current; release candidate 2026-05-21) | **Stateless core**: no sessions, no `initialize` handshake, version and capabilities carried in `_meta` on every request. **Multi Round-Trip Requests** (`input_required`) replace server-initiated requests. `subscriptions/listen` replaces resource subscribe. **Deprecated:** Sampling, Roots, Logging, HTTP+SSE, Dynamic Client Registration (≥12-month window). Tasks moved into an extension. **Extensions framework** (Tasks, MCP Apps, Enterprise Managed Authorization). Cache hints (`ttlMs`, `cacheScope`). OpenTelemetry trace context. Deterministic `tools/list` order. | [V][chg-2607], [V][blog-2607] |

**Other developments:**

- **MCP Apps** (`ui://` resources): first official extension, 2026-01-26, built jointly by Anthropic and OpenAI [V][apps].
- **Server Cards** (`/.well-known/mcp.json`, SEP-1649): still experimental [S][card].
- **Roadmap (2026-08-22)** [V][roadmap]:
  - agentic messaging and server-initiated events;
  - an HTTP-native transport;
  - agent identity (DPoP, Workload Identity Federation);
  - **progressive tool discovery for large catalogs**;
  - SDK conformance.
- **Open discrepancy:** the 2026-07-28 changelog calls `server/discover` mandatory, but the GA blog calls it optional [U].

**What this means for benchmarking:**

- Statelessness changes the connection and session cost model. The adapter currently holds a persistent session.
- Cache hints and a deterministic tool order make runs more reproducible and friendlier to KV caching.
- Deprecating Sampling removes the "server borrows your LLM" pattern.

---

## 3. Governance and adoption

**Governance:**

- **Agentic AI Foundation (AAIF).** Anthropic donated MCP on 2025-12-09 [V][aaif].
  - AAIF is a directed fund under the Linux Foundation.
  - Co-founders: Anthropic, OpenAI (AGENTS.md) and Block (goose). Backers include Google, Microsoft, AWS, Cloudflare and Bloomberg.
  - At launch it reported 10,000+ active public MCP servers.
- **A2A** joined the AAIF in August 2026 [V][a2a-aaif].
- **Official MCP Registry**: preview since 2025-09-08 [V][registry]. Whether it has reached GA is unverified [U].

**Adoption by major vendors:**

| Vendor | MCP adoption |
|---|---|
| OpenAI | MCP in the Agents SDK (2025-03-26) [S][tc-oai]; Responses API built-in `mcp` tool and connectors [S][oai-mcp]; ChatGPT Developer Mode as a full MCP client. |
| Google | Managed remote MCP servers for Google Cloud (Dec 2025) [S][g-managed]; ADK `McpToolset`; Gemini Managed Agents `mcp_server` tool. |
| Microsoft | Native MCP in Windows 11 with an on-device registry [S][win]; MCP GA in Copilot Studio; Azure AI Foundry Agent Service MCP tools; Azure AI Search knowledge bases exposed as MCP endpoints. |
| AWS | Bedrock AgentCore GA 2025-10-13; its Gateway turns APIs and Lambdas into MCP tools [S][agentcore]; awslabs Bedrock KB Retrieval MCP server. |
| Anthropic | Claude connectors; Tool Search Tool / `defer_loading`; *Code execution with MCP*; Agent Skills made an open standard (2025-12-18) [S][skills]. |

---

## 4. Architecture patterns for RAG via MCP

### 4.1 MCP vs RAG: complementary layers

- **LlamaIndex, "MCP vs. Vector Search" (2025-06-18)** [V][li-vs]
  - MCP does not kill vector search. Federated MCP search has no global relevance ranking, waits on the slowest API, and is only as good as each provider's own search.
  - Most enterprise data is unstructured (PDFs, slides, scans) and must be parsed and indexed first. This connects to doc 11 (OCR/parsing).
  - Recommendation: combine direct-API tools with retrieval tools over pre-indexed data.
- **Elastic Search Labs (2026-03-05)** [V][es-obsolete]
  - Federated MCP retrieval brings back unpredictable latency, fragmented relevance and shallow context.
  - Indexes are what make MCP agents work at all.
- **Practitioner consensus** [S][stackone], [S][airbyte]: production agents use both, i.e. semantic search over an index plus MCP calls into live systems ("agentic RAG via MCP").

### 4.2 Pre-retrieval vs just-in-time retrieval (context engineering)

- **Anthropic, "Effective context engineering for AI agents" (2025-09-29)** [V][ctx-eng]
  - Contrasts embedding retrieval done *before* inference with "just-in-time" retrieval, where the agent holds lightweight identifiers and loads data through tools.
  - Recommends a **hybrid**: some data up front for speed, the rest explored autonomously.
  - Warns about **context rot** (recall degrades as context grows) and about tool sets with overlapping functions.
  - Recommends compaction and clearing old tool results.
- **LangChain** groups context strategies into four: write, select, compress, isolate [S][lc-ctx].
- **Manus** [S][manus]
  - KV-cache hit rate is the key production metric: cached tokens cost about a tenth as much, and input outweighs output roughly 100:1.
  - Tools are masked through logits rather than removed, which keeps the prompt prefix cache-stable. This argues against swapping tool lists in the middle of a session.
- **Survey:** Mei et al., *A Survey of Context Engineering for LLMs*, arXiv:2507.13334 (166 pages, 1,400+ references) [S][ctx-survey]. It treats RAG, memory, tool-integrated reasoning and protocols such as MCP as one system-design space.

### 4.3 Tool design for retrieval tools

Anthropic, "Writing effective tools for agents" (2025-09-11) [V][tools]:

- Use a few consolidated, namespaced tools.
- Return high-signal fields (names rather than UUIDs).
- Offer a `response_format` choice of concise or detailed (72 vs 206 tokens in the example).
- Paginate and truncate results. Claude Code caps tool responses at 25k tokens by default.
- Evaluate on held-out tasks, tracking runtime, number of tool calls, token consumption and tool errors.

### 4.4 Reducing tool-definition overhead

| Technique | Mechanism | Reported effect |
|---|---|---|
| **Tool Search Tool / `defer_loading`** (Anthropic, 2025-11-24) | Tools are discovered on demand instead of preloaded. | About 85% fewer tokens (≈72–77k → ≈8.7k for 50+ tools). MCP-eval accuracy: Opus 4 49%→74%, Opus 4.5 79.5%→88.1% [V][adv-tool]. |
| **Programmatic tool calling** | The model orchestrates tools in code. | −37% tokens (43,588 → 27,297) [V][adv-tool]. |
| **Tool use examples** | Few-shot examples in tool definitions. | 72%→90% on complex parameter handling [V][adv-tool]. |
| **Code execution with MCP** (Anthropic, 2025-11-04) | Servers presented as code modules on a filesystem; intermediate results stay in a sandbox; PII can be tokenized. | 150k → 2k tokens (−98.7%) in the example workflow [V][code-exec]. |
| **Cloudflare Code Mode** | MCP tools exposed as a TypeScript API that the model writes code against. | About 99.9% fewer tokens for thousands of endpoints [V/S][cf-code]. |
| **Tool-schema compression** | Compact schema serialization. | 44–50% fewer schema tokens; +20.5 pp EM at an 8k budget; +48 pp EM on HotpotQA when schemas overflow the context [V][schema-comp]. |
| **Tool RAG** (see §5) | Retrieve only the relevant tool descriptions per query. | 50–99.6% fewer tool tokens. |

**Measured overhead in the wild** [S][gh-tokens], [S][gh-55k]:

- The GitHub MCP server alone costs about 17.6k–55k tokens, depending on version and on who measured.
- One reported three-server setup used about 143k of a 200k-token window.
- Measurement tool: [`mcp-tokens`][mcp-tokens].

### 4.5 GraphRAG and knowledge graphs via MCP

- Neo4j `mcp-neo4j-cypher` exposes schema, read-Cypher and write-Cypher tools [S][neo4j].
- `mcp-neo4j-graphrag` combines vector search, fulltext search and graph traversal [S][neo4j-gr].
- No dedicated academic paper on MCP-based KG-RAG was found. KG-R1 (arXiv:2509.26383) is agentic KG-RAG without MCP [U].

---

## 5. Literature: retrieval *of* tools (Tool RAG for MCP)

| # | Paper | ID / date | Key result |
|---|---|---|---|
| 1 | **RAG-MCP: Mitigating Prompt Bloat in LLM Tool Selection via RAG** — Gan & Sun | arXiv:2505.03275, 2025-05 [S] | Indexes server descriptions and passes only the top-k servers. Prompt tokens cut by more than 50%; tool-selection accuracy **43.13% vs 13.62%** for the all-tools baseline. Success above 90% with fewer than 30 tools, collapsing beyond about 100. |
| 2 | **MCP-Zero: Active Tool Discovery for Autonomous LLM Agents** — Fei et al. | arXiv:2506.01056, 2025-06 [S] | The LLM writes structured tool requests; a hierarchical server→tool semantic router resolves them. Dataset: 308 servers, 2,797 tools. **98% fewer tokens on APIBank** with accuracy maintained. |
| 3 | **ScaleMCP: Dynamic and Auto-Synchronizing MCP Tools** — Lumer et al. (PwC) | arXiv:2505.06416, 2025-05 [S] | The agent calls a tool retriever; the tool store stays in sync with servers through CRUD. Introduces TDWA embeddings. Tested on 5,000 MCP servers, 10 LLMs and 5 embedders. |
| 4 | **Tool-to-Agent Retrieval** — Lumer et al. | arXiv:2511.01854, 2025-11 [S] | Tools and their parent servers/agents in one shared vector space. **+19.4% Recall@5, +17.7% nDCG@5** on LiveMCPBench. |
| 5 | **Dynamic ReAct: Scalable Tool Selection for Large-Scale MCP Environments** — Gaurav et al. | arXiv:2509.20386, 2025-09 [V] | Search-and-load meta-tool. Up to 50% less tool loading with no loss in task accuracy. |
| 6 | **Semantic Tool Discovery for LLMs: Vector-Based MCP Tool Selection** — Mudunuri et al. | arXiv:2603.20313, 2026-03 [V] | Dense retrieval of the top 3–5 tools out of 121. **99.6% fewer tool tokens, hit@3 97.1%, MRR 0.91**, under 100 ms. |
| 7 | **HumanMCP: Human-Like Query Dataset for MCP Tool Retrieval** — Laddha et al. | arXiv:2602.23367 [V] | Persona-based queries ranging from precise to vague. Shows that synthetic queries inflate measured retrieval quality. |
| 8 | **Retrieval Models Aren't Tool-Savvy (ToolRet)** — Shi et al. | ACL Findings 2025, arXiv:2503.01763 [S] | 7.6k tasks over a 43k-tool corpus. The best IR model reaches only **nDCG@10 33.83**. The standard tool-retrieval IR benchmark, though not MCP-specific. |
| 9 | **MCP Tool Descriptions Are Smelly!** — Hasan et al. | arXiv:2602.14878, 2026-02 [V] | 856 tools from 103 servers; **97.1%** have at least one "smell". Augmenting descriptions raises median task success by 5.85 pp but adds 67% more execution steps. |
| 10 | **From Docs to Descriptions: Smell-Aware Evaluation of MCP Server Descriptions** — Wang et al. | arXiv:2602.18914, 2026-02 [V] | 10,831 servers, 18 smell categories. Fixing smells improves selection by 11.6% and 8.8% (two smell categories). Among equivalent servers, a well-described one is selected 72% of the time vs a 20% baseline. |
| 11 | **Tool-Schema Compression Enables Agentic RAG Under Constrained Context Budgets** — Sakizli | arXiv:2605.26165, 2026-05 [V] | See §4.4. Uncompressed JSON schemas overflow at about 494 tools; compressed schemas still work beyond 800. |

**Takeaway.** Tool selection is itself a retrieval problem, so IR metrics (Recall@k, nDCG, MRR) apply directly. Tool *description quality* is a confound in every MCP-RAG comparison.

---

## 6. Literature: MCP agent benchmarks

| Benchmark | ID / venue | Scope | Headline result |
|---|---|---|---|
| **MCP-Bench** (Accenture) | arXiv:2508.20453 [V] | 28 live servers, 250 tools, deliberately fuzzy instructions | Schema compliance above 98%, yet overall score tops out at **0.749**. Gaps are in planning and dependencies. |
| **MCP-Universe** (Salesforce) | arXiv:2508.14704 [S] | 11 real servers, 6 domains, execution-based evaluators | **GPT-5 43.72%**, Grok-4 33.33%, Claude-4-Sonnet 29.44%. |
| **MCPToolBench++** | arXiv:2508.07575 [V] | 1.5k QA pairs from more than 4k servers | No model leads on both AST and Pass@1. |
| **LiveMCPBench** | arXiv:2508.01780 [V] | 95 tasks, 70 servers, 527 tools | Claude-Sonnet-4 **78.95%**, most other models 30–50%. **About half of failures are retrieval errors.** The LLM judge agrees with humans 81% of the time. |
| **MCP-RADAR** | arXiv:2505.16700 [V] | 507 tasks; 5 dimensions (accuracy, tool-selection efficiency, resources, parameters, speed) | Clear accuracy–efficiency trade-off. |
| **MCPEval** (Salesforce) | arXiv:2507.12806 [V] | Automatic task generation and trajectory-level evaluation | Open-source framework. |
| **MCP-AgentBench** | arXiv:2509.09734 [S] | 33 servers, 188 tools, 600 queries | Outcome-oriented MCP-Eval. |
| **MCPMark** | arXiv:2509.24002, ICLR 2026 [V/S] | 127 CRUD-heavy tasks (Notion, GitHub, Postgres, …) | gpt-5-medium **52.56% pass@1, 33.86% pass^4**; about 17 tool calls per task. |
| **MCP-Atlas** (Scale AI) | arXiv:2602.00933 [V] | 1,000 tasks, 36 servers, 220 tools; scored by claim coverage | Best 82.2%. **63.3% of failures are cognitive**, not tool-call errors. |
| **Toolathlon** | arXiv:2510.25726, ICLR 2026 [V] | 32 apps, 604 tools, about 20 turns per task | Claude-4.5-Sonnet **38.6%**. |
| **MCPVerse** | arXiv:2508.16260 [V] | 550+ tools, action space above 140k tokens | Most models degrade as the tool set grows. |
| **MCPEvol-Bench** | arXiv:2607.14642, 2026-07 [V] | Tool drift simulated with 11 mutation operators on 123 servers | GPT-5.4 −13.7%, Claude-Sonnet-4.6 −14.4% on the evolved servers. |
| **MCPWorld** | arXiv:2506.07672 [S] | API, GUI and hybrid computer-use tasks | Baseline 75.12%. |
| **TOUCAN** (training data) | arXiv:2510.01179 [V] | 1.5M trajectories from about 500 MCP servers | Fine-tuned models beat larger closed models on BFCL V3. |

**Observation.** None of these benchmarks targets *knowledge-grounded QA over a fixed corpus*, which is the classic RAG task. They measure tool orchestration. RAG quality metrics (faithfulness, context precision/recall, gold-retrieval recall) are almost absent from MCP benchmarking.

---

## 7. Evidence: MCP-based retrieval vs classic RAG

| Source | Setup | Finding | Caveat |
|---|---|---|---|
| **MCPGAUGE**: *Help or Hurdle? Rethinking MCP-Augmented LLMs* (arXiv:2508.12566) [S] | 6 commercial LLMs, 30 MCP tool suites | Automatic MCP access lowers accuracy by **9.5%** on average and multiplies input tokens by **3.25–236.5**. Shows a "warm-up" effect: little proactive tool use in the first turn. | Not RAG-specific; tool suites are heterogeneous. |
| **Infragistics MCP vs RAG benchmark** (2026-07-10) [V][infra] | 100 documentation queries; Bedrock Knowledge Base RAG vs a 4-tool live-docs MCP server; LLM judge | MCP wins **52% vs 38%** (10% ties). Latency 31.9 s vs 12.7 s (+151%). Cost per query **$0.169 vs $0.0047** (about 36×). Citations 4.5 vs 2.2. MCP is better at exact API lookups, RAG at stable how-to content. | The two systems used different corpora and freshness, so the comparison is not fair. |
| **Tool-Schema Compression** (arXiv:2605.26165) [V] | 14 models under constrained context budgets | Schema tokens crowd out retrieved context; compression gains +20.5 pp EM at 8k. | The effect largely disappears at 32k. |
| **AgenticRAG for enterprise knowledge bases** (arXiv:2605.05538) [V] | Agent tools search / find / open / summarize | BRIGHT recall@1 **49.6% (+21.8 pp)**; FinanceBench 92%; **5.9× better than single-shot retrieval**. | Tool-based, but the abstract does not mention MCP. |
| **MCP-enabled meta-optics design** (arXiv:2508.10277) [S] | Domain application | MCP succeeded in 100% of trials vs 1/50 for documentation-RAG. | Domain-specific; not verified. |

**Synthesis [U].**

- Agentic, tool-based retrieval can beat single-shot RAG on hard multi-hop and reasoning-intensive retrieval.
- It pays for this with much higher latency and token cost.
- It *loses* accuracy when tools are added indiscriminately or when schemas crowd out context.
- The net effect depends on tool-set size, schema compactness, tool-loading mode, number of agent rounds, and the model.

No published study controls all of these factors, so all of them should be treated as **experimental variables**.

---

## 8. Agentic RAG, search agents, deep research

- **Agentic RAG survey**, Singh et al., arXiv:2501.09136 [S][agrag]
  - Taxonomy organized by agentic pattern: reflection, planning, tool use, multi-agent.
- **SoK: Agentic RAG**, Mishra et al., arXiv:2603.07379 [V]
  - Models agentic RAG as a POMDP.
  - Risks: compounding hallucination, memory poisoning, cascading tool failures.
  - Argues that **static evaluation is inadequate**.
- **Search-R1**, Jin et al., arXiv:2503.09516 [S]
  - Multi-turn search trained with RL, masking retrieved tokens.
  - +26% for Qwen2.5-7B across 7 QA datasets.
- **ReSearch**, Chen et al., arXiv:2503.19470 [S]
  - Search as part of the reasoning chain, trained with RL; reflection emerges on its own.
  - Its successor ReCall generalizes the approach to arbitrary tools.
- **Deep Research Agents: A Systematic Examination and Roadmap**, Huang et al., arXiv:2506.18096 [V]
  - Explicitly analyzes MCP integration for extensibility.
- **Deep Research: A Systematic Survey**, Shi et al., arXiv:2512.02038 [V]
  - Four components: query planning, information acquisition, memory, answer generation.
- **A Survey of LLM-based Deep Search Agents**, Xi et al., arXiv:2508.05668 [V]
- **Interoperability protocols survey (MCP, ACP, A2A, ANP)**, Ehtesham et al., arXiv:2505.02279 [V]

**Trend.**

- RAG is shifting from a fixed retrieve-then-generate pipeline toward *retrieval as a tool inside a reasoning loop*.
- MCP is becoming the default packaging for those tools.
- Evaluation has to move from single-turn metrics to trajectory-level metrics.

---

## 9. Vendor landscape: retrieval MCP servers

Defaults matter for fair benchmarks. They differ between servers, so they must be pinned explicitly.

| Vendor / server | Exposed retrieval surface | Notable defaults / notes | Src |
|---|---|---|---|
| Qdrant `mcp-server-qdrant` | `qdrant-store`, `qdrant-find` | FastEmbed only (default `all-MiniLM-L6-v2`); `QDRANT_SEARCH_LIMIT`=10; read-only flag; stdio, SSE and streamable-http | [V][qdrant] |
| Pinecone Assistant | One remote MCP endpoint per assistant; context tool (`query`, `top_k`) | `top_k` default 15 | [S][pinecone] |
| Weaviate (built-in, preview) | `weaviate-query-hybrid`, config and tenant tools, upsert | Hybrid alpha 0.75; RBAC; disabled by default | [S][weaviate] |
| Milvus / Zilliz | `milvus_vector_search` with filter expressions | limit 5 | [S][milvus] |
| Chroma `chroma-mcp` | `chroma_query_documents` with `where` and `where_document` filters | `n_results` 5 | [S][chroma] |
| Elasticsearch | Agent Builder MCP endpoint (Elastic 9.2+): Query DSL, ES\|QL, mappings | The standalone server is deprecated | [S][elastic] |
| MongoDB MCP Server | Atlas operations, vector and lexical index creation, semantic queries | — | [S][mongo] |
| Vectara | `search_vectara` (retrieval) vs `ask_vectara` (full RAG); hallucination-correction tools | Can generate the answer server-side | [S][vectara] |
| LlamaCloud / LlamaIndex | One tool per managed index; `workflow_as_mcp` | — | [S][llamacloud] |
| Haystack Hayhooks | Each pipeline becomes an MCP tool | — | [S][haystack] |
| **RAGFlow** | `ragflow_retrieval` (question, dataset_ids, page_size, similarity_threshold, vector_similarity_weight) | page_size 10, threshold 0.2. This repo already has a RAGFlow adapter. | [S][ragflow] |
| Azure AI Search / Foundry IQ | One MCP endpoint per knowledge base; agentic retrieval runs inside Search | Managed-identity auth | [S][azure] |
| AWS Bedrock KB Retrieval (awslabs) | `QueryKnowledgeBases` with optional reranking | Reranking off by default | [S][bedrock] |
| Glean | Permission-aware `search`, `chat`, `read_document` | Managed remote MCP | [S][glean] |
| Context7 / Exa / Tavily | Docs and web search | Tavily's remote endpoint takes the API key in the URL query, so it ends up in logs | [S][exa], [S][tavily] |
| Google Vertex AI RAG Engine | No first-party MCP server found; integration goes through ADK | — | [U] |

**Web-search MCP evaluations (AIMultiple):**

- Web-access MCP benchmark [V][aim-mcp]: Bright Data 100% on search/extract; Firecrawl fastest at about 7 s; 76.8% success under a 250-concurrent-agent stress test.
- Agentic search APIs [S][aim-search]: Brave, Firecrawl, Exa and Parallel are statistically tied; latency ranges from 669 ms to 13.6 s.

---

## 10. Frameworks: MCP client integration

| Framework | Integration | Src |
|---|---|---|
| LangChain | `langchain-mcp-adapters` (`MultiServerMCPClient`, `load_mcp_tools`). Being replaced by `langchain.mcp` (built on FastMCP, beta, `langchain[mcp]>=1.4.0`). This repo's agentic mode converts tools to LangChain schemas, so the migration is relevant. | [S][lc-mcp], [V][lc-migrate] |
| OpenAI Agents SDK | `HostedMCPTool` (the Responses API calls the server itself), `MCPServerStreamableHttp`, `MCPServerStdio`; tool filters, `cache_tools_list`, tracing. Hosted and local modes put latency in different places. | [V][oai-sdk] |
| LlamaIndex | `BasicMCPClient`, `McpToolSpec`, `workflow_as_mcp` | [S][li-mcp] |
| Google ADK | `McpToolset` (discovery, schema adaptation, tool filtering) | [S][adk] |
| Microsoft Agent Framework | Native MCP tools and A2A; agents can be exposed as MCP servers | [S][maf] |
| Spring AI | MCP client boot starter (sync/async; STDIO, SSE, Streamable HTTP) | [S][spring] |
| Pydantic AI | `MCPServerStreamableHTTP` toolset | [S][pyd] |
| FastMCP | `FastMCP.from_openapi` turns a REST API into MCP tools/resources | [S][fastmcp] |

---

## 11. Security: implications for RAG over MCP

### 11.1 Attack classes and incidents

| Attack / incident | Mechanism | Src |
|---|---|---|
| **Tool poisoning, rug pull, shadowing** (Invariant Labs, 2025-04) | Hidden instructions in tool descriptions; descriptions changed after the user approved them; a malicious server changes how *trusted* tools are used. | [V][inv-tp] |
| **Line jumping** (Trail of Bits) | `tools/list` output enters the model context, so injection happens *before any tool is called*. | [S][tob] |
| **GitHub MCP "toxic agent flow"** | A malicious public issue makes the agent leak private-repo data into a public PR. No tool is compromised; untrusted content plus a broad token is enough. | [S][inv-gh] |
| **Lethal trifecta** (Willison) | An agent session that combines private data, untrusted content and an egress channel is exploitable. | [S][trifecta] |
| **Supabase MCP leak** | A support ticket injected instructions; the agent held a `service_role` key that bypasses row-level security (RLS) and dumped private tables into the ticket. | [S][supabase] |
| **Asana MCP** (2025-06) | A tenant-isolation bug; about 1,000 customers potentially exposed. | [S][asana] |
| **CVE-2025-6514** (mcp-remote, CVSS 9.6) | A crafted `authorization_endpoint` reaches the OS shell, giving remote code execution (RCE). | [S][jfrog] |
| **CVE-2025-49596** (MCP Inspector, CVSS 9.4) | An unauthenticated proxy combined with browser flaws gives RCE from simply visiting a website. | [S][oligo] |
| **postmark-mcp** (2025-09) | The first malicious MCP package found in the wild: after 15 clean versions, an update silently BCC'd every email to the attacker. | [S][postmark] |
| **Retrieval-Agent Deception (RADE)** | Poisoned public data lands in the victim's vector DB. Retrieved chunks carry MCP commands, demonstrated end-to-end as credential theft. **This is the core RAG-specific attack.** | [V][safety-audit] (verify the acronym expansion in the PDF) |

### 11.2 Academic security work

| Paper | ID | Key finding |
|---|---|---|
| MCP Safety Audit (Radosevich & Halloran) | arXiv:2504.03767 [V] | Introduces RADE; releases MCPSafetyScanner. |
| MCPTox | arXiv:2508.14925 [V] | 45 servers, 1,348 poisoning cases. Attack success rate up to **72.8%**; more capable models are more vulnerable. |
| MCPSecBench | arXiv:2508.13220 [S] | 17 attack types; over 85% compromise at least one platform. |
| Systematic Analysis of MCP Security (Guo et al.) | arXiv:2508.12538 [V] | 31 attack methods. Agents trust tool descriptions uncritically and cannot reliably separate data from commands. |
| MCP-Guard | arXiv:2508.10991 [S] | 3-stage detector with F1 95.4%; releases MCP-AttackBench (70k samples). |
| MCIP (EMNLP 2025) | arXiv:2505.14590 [S] | Contextual-integrity taxonomy plus training data for risk detection. |
| ETDI | arXiv:2506.01333 [S] | Signed, versioned tool manifests and policy-based access control. |
| MCP at First Glance | arXiv:2506.13538 [S] | Of 1,899 servers, 7.2% have general vulnerabilities and 5.5% show tool poisoning. |
| MCP-SafetyBench (ICLR 2026) | arXiv:2512.15163 [S] | Every model tested is vulnerable; clear safety–utility trade-off. |
| SoK: Security and Safety in the MCP Ecosystem | arXiv:2512.08290 [V] | Taxonomy across resources, prompts and tools. |
| Rethinking MCP Security (MCPZoo) | arXiv:2607.11086 [V] | 64,611 servers. Scanners flag 96.9% as risky, but **fewer than 50% of the alerts are true positives**. |
| Exposed by Design | arXiv:2608.00150 [V] | About 21k internet-facing servers; **91.8% of those tested lack OAuth**; 687 shell-exec tools with no access control. |
| CaMeL (Google DeepMind) | arXiv:2503.18813 [S] | Separates control flow from data flow so untrusted data cannot change program flow. Provable security at 77% AgentDojo utility (vs 84% undefended). |
| PoisonedRAG (USENIX Security 2025) | arXiv:2402.07867 [S] | Injecting 5 texts per target question gives about 90% attack success. |

### 11.3 Standards and defenses

- **OWASP MCP Top 10 (beta)** [S][owasp-mcp]. Examples: MCP01 token/secret exposure, MCP03 tool poisoning, MCP04 supply chain, MCP10 context injection and over-sharing.
- **OWASP LLM Top 10 2025, LLM08 "Vector and Embedding Weaknesses"** [S][owasp-llm08]. RAG-specific: tenant isolation, access control on retrieval, embedding poisoning.
- **OWASP Agentic Top 10 2026** [S]. ASI06 covers memory and context poisoning.
- **CoSAI MCP Security white paper** (2026-01) [S][cosai]. About 40 threats in 12 categories.
- **MCP spec "Security Best Practices"** [V][spec-sec]:
  - no token passthrough;
  - per-client consent, to prevent confused-deputy attacks;
  - SSRF protection during OAuth discovery;
  - sandboxing of local servers;
  - least-privilege scopes, granted progressively;
  - in the stateless spec, state handles must be bound to the verified user.
- **Gateways and portals:** Docker MCP Gateway (container isolation, call interceptors) [S][docker-gw]; Azure API Management as an AI gateway [S][ms-gw]; Cloudflare MCP Server Portals [S][cf-portal].
- **Scanners:** Invariant `mcp-scan` (tool pinning and hashing) [S][mcp-scan]. Treat scanner output as noisy (MCPZoo).

### 11.4 Concrete consequences for RAG [U — synthesized from the sources above]

1. **Retrieved chunks are untrusted input.**
   - Tag every chunk with its provenance.
   - Never let retrieved text decide which tools get called (the CaMeL pattern).
2. **Break the lethal trifecta.**
   - Give retrieval servers read-only, narrowly scoped credentials.
   - Do not put them in the same session as tools that can send data out, or route that egress through gateway interceptors.
3. **Enforce ACLs at query time, inside the server.**
   - Base them on an audience-validated end-user token.
   - Prefer one namespace per tenant over metadata filters alone.
   - Never pass the user's identity as a tool argument the model can change.
4. **Pin tool definitions** by hash or version, so rug pulls and shadowing of the retrieval tool are detected.
5. **Treat ingestion as supply chain.** Pin the versions of parsers and connectors that are invoked through MCP.

---

## 12. Enterprise / EU angle

Sources are thin and mostly vendor marketing. No MCP-specific guidance from BSI or the EU AI Office was found [U].

- **Data residency is not sovereignty.** A US provider hosting in Frankfurt is still subject to the CLOUD Act [S][sovereignty].
- **On-prem and EU-hosted offerings.** MCP gateways and servers are marketed with ISO 27001 / BSI C5 claims [S][securecloud], [S][frends].
- **EU AI Act.** Vendors argue that MCP gateways reaching tools in high-risk domains fall under conformity assessment, which would require logging, access control and human-oversight hooks. This is a vendor interpretation, not legal guidance [S][maxim].
- **Code execution with MCP** keeps intermediate data, including tokenized PII, out of the model context. That is a useful property for GDPR-sensitive RAG [V][code-exec].

---

## 13. Implications for this framework

This section maps the findings onto the current `McpRagAdapter` (`benchmark/adapters/mcp.py`). Everything here is a recommendation **[U]**, not established practice.

**Current state of the adapter:**

- Execution modes: `fixed` calls a single tool per question; `agentic` runs an LLM-controlled multi-tool loop limited by `allowed_tools` and `max_agent_rounds`.
- Result modes: `context` returns chunks and the harness generates the answer; otherwise the server answers.
- Retries and timeouts.
- Fairness validation is on by default (`MCP_ENFORCE_FAIRNESS`).

These modes map well onto the distinctions the literature draws: pipeline vs agentic retrieval, and retrieval-only vs answer-generating tools.

### 13.1 Gaps worth closing

| # | Recommendation | Motivation |
|---|---|---|
| 1 | **Record tool-definition token overhead per run**: tokens in the serialized `tools/list` and in tool results, logged to MLflow. | Overhead ranges from 50k+ tokens (Anthropic) to 236× (MCPGAUGE), and is currently invisible in reports. |
| 2 | **Log a tool-schema hash, the server version and the transport** in run provenance. | Tool drift costs 13–14% (MCPEvol-Bench); servers rename and deprecate tools (Elastic, Context7); the hash also enables rug-pull detection. |
| 3 | **Make the tool-loading mode an experimental variable**: all tools / allowlist / tool retrieval (top-k tools) / compressed schemas. | Tool Search raised Opus 4 from 49% to 74%; schema compression +20.5 pp EM; RAG-MCP collapses beyond about 100 tools. |
| 4 | **Distractor-tool sweeps**: add N irrelevant tools in agentic mode and measure the degradation. | Would reproduce the RAG-MCP, MCPVerse and LiveMCPBench findings on a RAG-QA task, which no existing benchmark covers. |
| 5 | **Trajectory metrics**: number of tool calls, rounds, redundant calls, tool errors, parameter validity, and tool-selection accuracy when the gold tools are known. Builds on doc 10, recommendation 9. | Agentic RAG needs trajectory-level evaluation (SoK Agentic RAG; MCP-RADAR; Anthropic's tool-eval metric set). |
| 6 | **Require stable chunk ids in `context` mode** (e.g. a `result_field` path to ids), so gold-retrieval metrics such as recall@k and nDCG work for MCP systems. | Many servers return free text without ids, which breaks retrieval evaluation. |
| 7 | **Pin server-side retrieval defaults in manifests** (top-k, similarity threshold, hybrid alpha) and include them in the fairness check. | Defaults differ: Qdrant 10, Pinecone 15, Chroma/Milvus 5, RAGFlow threshold 0.2. |
| 8 | **Repeat runs and separate cold from warm runs** (persistent session, tool-list cache, KV cache). | Agentic runs are stochastic (AIMultiple used 5 runs); caching dominates cost in production (Manus). |
| 9 | **Report cost per query and a latency breakdown** (connect, list_tools, each call, generation). | In Infragistics, MCP cost about 36× more and added 151% latency. |
| 10 | **Optional security suite**: success rate of poisoned-corpus injection (PoisonedRAG/RADE-style), malicious tool-description variants (MCPTox-style), cross-tenant ACL probes. | No benchmark combines RAG quality with MCP security, and the fairness validation currently has no security dimension. |
| 11 | **Track spec changes.** The 2026-07-28 stateless core removes sessions and the `initialize` handshake. Check the MCP Python SDK version and whether the persistent-session design and connection-time accounting still hold. | Protocol-level change; Sampling and SSE are deprecated. |

### 13.2 Research contribution opportunity

The literature consistently names one missing piece: **a controlled comparison of the same retriever and corpus exposed in four ways, with generator, top-k and prompt held fixed**:

- (a) in-process,
- (b) via HTTP,
- (c) via a fixed MCP tool,
- (d) via an agentic MCP loop.

It should measure RAGAS quality, gold-retrieval metrics, tokens, latency and cost.

The existing adapter set (`internal`, `http`, `mcp` fixed/agentic) plus fairness validation already supports this design. Suggested first experiment:

```yaml
# sketch — same corpus, same generator, four arms
matrix:
  rag_system: [internal, http, mcp]
  mcp_execution_mode: [fixed, agentic]      # only for rag_system=mcp
  mcp_distractor_tools: [0, 20, 100]        # new knob (gap #4)
```

The `mcp_distractor_tools` knob does not exist yet. Per `AGENTS.md`, adding it needs a `BenchmarkConfig` field, an env fallback and a test.

---

## 14. Open research gaps (summary)

1. No controlled comparison of MCP-based vs pipeline RAG (§13.2).
2. No MCP benchmark targets corpus-grounded QA with RAG quality metrics.
3. No academic work was found on federated RAG via MCP or on MCP-native KG-RAG.
4. Tool-description quality and tool drift are uncontrolled confounds in all MCP evaluations.
5. Security and utility are evaluated separately; there is no joint evaluation (utility under attack) for RAG.
6. No regulator guidance (EU/DE) specific to MCP.

---

## 15. Items to re-verify before citing

- Whether `server/discover` is mandatory or optional in 2026-07-28: the changelog and the blog disagree.
- Whether the MCP Registry has reached GA.
- The expansion of the RADE acronym (check the arXiv:2504.03767 PDF).
- The exact Asana exposure window, and the canonical OWASP LLM08 URL.
- Papers marked [U]: arXiv:2510.15994 (MSB), 2510.23673 (MCPGuard), 2510.19423 (ETOM), 2607.20531 (DynamicMCPBench), 2508.12752, 2608.23992 (Hybrid Semantic Tool Discovery for Enterprise MCP Gateway).
- Version-specific details taken from search snippets: the Weaviate MCP preview version, the Azure AI Search API version, and the status of `langchain.mcp`.

---

## References

### Academic papers (arXiv abstract pages)

- **Tool retrieval:**
  - [2505.03275](https://arxiv.org/abs/2505.03275) RAG-MCP
  - [2506.01056](https://arxiv.org/abs/2506.01056) MCP-Zero
  - [2505.06416](https://arxiv.org/abs/2505.06416) ScaleMCP
  - [2511.01854](https://arxiv.org/abs/2511.01854) Tool-to-Agent Retrieval
  - [2509.20386](https://arxiv.org/abs/2509.20386) Dynamic ReAct
  - [2603.20313](https://arxiv.org/abs/2603.20313) Semantic Tool Discovery
  - [2602.23367](https://arxiv.org/abs/2602.23367) HumanMCP
  - [2503.01763](https://arxiv.org/abs/2503.01763) ToolRet
  - [2602.14878](https://arxiv.org/abs/2602.14878) Smelly MCP descriptions
  - [2602.18914](https://arxiv.org/abs/2602.18914) Docs to Descriptions
  - [2605.26165](https://arxiv.org/abs/2605.26165) Tool-Schema Compression
- **Benchmarks:**
  - [2508.20453](https://arxiv.org/abs/2508.20453) MCP-Bench
  - [2508.14704](https://arxiv.org/abs/2508.14704) MCP-Universe
  - [2508.07575](https://arxiv.org/abs/2508.07575) MCPToolBench++
  - [2508.01780](https://arxiv.org/abs/2508.01780) LiveMCPBench
  - [2505.16700](https://arxiv.org/abs/2505.16700) MCP-RADAR
  - [2507.12806](https://arxiv.org/abs/2507.12806) MCPEval
  - [2506.07672](https://arxiv.org/abs/2506.07672) MCPWorld
  - [2509.09734](https://arxiv.org/abs/2509.09734) MCP-AgentBench
  - [2509.24002](https://arxiv.org/abs/2509.24002) MCPMark
  - [2602.00933](https://arxiv.org/abs/2602.00933) MCP-Atlas
  - [2510.25726](https://arxiv.org/abs/2510.25726) Toolathlon
  - [2508.16260](https://arxiv.org/abs/2508.16260) MCPVerse
  - [2607.14642](https://arxiv.org/abs/2607.14642) MCPEvol-Bench
  - [2510.01179](https://arxiv.org/abs/2510.01179) TOUCAN
- **Surveys and ecosystem:**
  - [2503.23278](https://arxiv.org/abs/2503.23278) MCP Landscape (Hou et al.)
  - [2505.02279](https://arxiv.org/abs/2505.02279) Agent interoperability protocols
  - [2506.13538](https://arxiv.org/abs/2506.13538) MCP at First Glance
  - [2509.25292](https://arxiv.org/abs/2509.25292) MCP ecosystem measurement
  - [2607.25635](https://arxiv.org/abs/2607.25635) MCP applications study
  - [2512.08290](https://arxiv.org/abs/2512.08290) SoK MCP security
  - [2508.12566](https://arxiv.org/abs/2508.12566) MCPGAUGE
  - [2507.13334](https://arxiv.org/abs/2507.13334) Context Engineering survey
  - Singh et al., *A Survey of the Model Context Protocol*, Preprints.org 2025 (not on arXiv)
- **Agentic RAG and deep research:**
  - [2501.09136](https://arxiv.org/abs/2501.09136) Agentic RAG survey
  - [2603.07379](https://arxiv.org/abs/2603.07379) SoK Agentic RAG
  - [2605.05538](https://arxiv.org/abs/2605.05538) AgenticRAG for enterprise
  - [2503.09516](https://arxiv.org/abs/2503.09516) Search-R1
  - [2503.19470](https://arxiv.org/abs/2503.19470) ReSearch
  - [2506.18096](https://arxiv.org/abs/2506.18096) Deep Research Agents
  - [2512.02038](https://arxiv.org/abs/2512.02038) Deep Research survey
  - [2508.05668](https://arxiv.org/abs/2508.05668) Deep Search Agents survey
  - [2510.05968](https://arxiv.org/pdf/2510.05968) ResourceLink patterns
  - [2508.10277](https://arxiv.org/abs/2508.10277) MCP for meta-optics design
- **Security:**
  - [2504.03767](https://arxiv.org/abs/2504.03767) MCP Safety Audit
  - [2508.14925](https://arxiv.org/abs/2508.14925) MCPTox
  - [2508.13220](https://arxiv.org/abs/2508.13220) MCPSecBench
  - [2508.12538](https://arxiv.org/abs/2508.12538) Systematic MCP security analysis
  - [2508.10991](https://arxiv.org/abs/2508.10991) MCP-Guard
  - [2505.14590](https://arxiv.org/abs/2505.14590) MCIP
  - [2506.01333](https://arxiv.org/abs/2506.01333) ETDI
  - [2512.15163](https://arxiv.org/abs/2512.15163) MCP-SafetyBench
  - [2607.11086](https://arxiv.org/abs/2607.11086) MCPZoo
  - [2608.00150](https://arxiv.org/abs/2608.00150) Exposed by Design
  - [2503.18813](https://arxiv.org/abs/2503.18813) CaMeL
  - [2402.07867](https://arxiv.org/abs/2402.07867) PoisonedRAG

### Web sources

The bracketed reference ids used in the text resolve to these URLs:

- **Spec and governance:**
  - [chg-2503](https://modelcontextprotocol.io/specification/2025-03-26/changelog)
  - [chg-2506](https://modelcontextprotocol.io/specification/2025-06-18/changelog)
  - [chg-2511](https://modelcontextprotocol.io/specification/2025-11-25/changelog)
  - [chg-2607](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx)
  - [blog-2607](https://blog.modelcontextprotocol.io/posts/2026-07-28/)
  - [spec-res](https://modelcontextprotocol.io/specification/2025-11-25/server/resources)
  - [spec-sec](https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices)
  - [workos](https://workos.com/blog/mcp-2025-11-25-spec-update)
  - [apps](https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/)
  - [card](https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1649)
  - [roadmap](https://blog.modelcontextprotocol.io/posts/mcp-roadmap/)
  - [aaif](https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/)
  - [a2a-aaif](https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/)
  - [registry](https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/)
- **Adoption:**
  - [tc-oai](https://techcrunch.com/2025/03/26/openai-adopts-rival-anthropics-standard-for-connecting-ai-models-to-data/)
  - [oai-mcp](https://developers.openai.com/api/docs/guides/tools-connectors-mcp)
  - [oai-dr](https://platform.openai.com/docs/guides/deep-research)
  - [g-managed](https://thenewstack.io/google-launches-managed-remote-mcp-servers-for-its-cloud-services/)
  - [win](https://blogs.windows.com/windowsexperience/2025/05/19/securing-the-model-context-protocol-building-a-safer-agentic-future-on-windows/)
  - [agentcore](https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available)
  - [skills](https://siliconangle.com/2025/12/18/anthropic-makes-agent-skills-open-standard/)
- **Engineering articles:**
  - [ctx-eng](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
  - [tools](https://www.anthropic.com/engineering/writing-tools-for-agents)
  - [code-exec](https://www.anthropic.com/engineering/code-execution-with-mcp)
  - [adv-tool](https://www.anthropic.com/engineering/advanced-tool-use)
  - [cf-code](https://blog.cloudflare.com/code-mode/)
  - [li-vs](https://www.llamaindex.ai/blog/does-mcp-kill-vector-search)
  - [es-obsolete](https://www.elastic.co/search-labs/blog/future-of-search-engines-indexed-search-mcp)
  - [stackone](https://www.stackone.com/blog/rag-vs-mcp/)
  - [airbyte](https://airbyte.com/agentic-data/mcp-vs-rag)
  - [lc-ctx](https://www.langchain.com/blog/context-engineering-for-agents)
  - [manus](https://medium.com/@peakji/context-engineering-for-ai-agents-lessons-from-building-manus-71883f0a67f2)
  - [infra](https://www.infragistics.com/blogs/mcp-vs-rag-benchmark)
  - [gh-tokens](https://getunblocked.com/blog/github-mcp-token-cost/)
  - [gh-55k](https://dev.to/rudratosh/your-github-mcp-server-costs-55000-tokens-before-your-agent-reads-a-single-word-4eah)
  - [mcp-tokens](https://github.com/sd2k/mcp-tokens)
  - [aim-mcp](https://aimultiple.com/browser-mcp)
  - [aim-search](https://aimultiple.com/agentic-search)
- **Retrieval MCP servers:**
  - [qdrant](https://github.com/qdrant/mcp-server-qdrant)
  - [pinecone](https://docs.pinecone.io/guides/assistant/mcp-server)
  - [weaviate](https://docs.weaviate.io/weaviate/configuration/mcp-server)
  - [milvus](https://github.com/zilliztech/zilliz-mcp-server)
  - [chroma](https://github.com/chroma-core/chroma-mcp)
  - [elastic](https://www.elastic.co/docs/explore-analyze/ai-features/agent-builder/mcp-server)
  - [mongo](https://www.mongodb.com/docs/mcp-server/tools/)
  - [vectara](https://github.com/vectara/vectara-mcp)
  - [llamacloud](https://github.com/run-llama/mcp-server-llamacloud)
  - [haystack](https://haystack.deepset.ai/blog/mcp-with-haystack)
  - [ragflow](https://ragflow.io/docs/mcp_tools)
  - [azure](https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-retrieve)
  - [bedrock](https://awslabs.github.io/mcp/servers/bedrock-kb-retrieval-mcp-server)
  - [glean](https://docs.glean.com/administration/platform/mcp/about)
  - [exa](https://docs.exa.ai/reference/exa-mcp)
  - [tavily](https://docs.tavily.com/documentation/mcp)
  - [neo4j](https://graphacademy.neo4j.com/courses/genai-mcp-neo4j-tools/2-using-neo4j-mcp-tools/1-mcp-neo4j-cypher/)
  - [neo4j-gr](https://github.com/neo4j-field/mcp-neo4j-graphrag)
- **Frameworks:**
  - [lc-mcp](https://github.com/langchain-ai/langchain-mcp-adapters)
  - [lc-migrate](https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters)
  - [oai-sdk](https://openai.github.io/openai-agents-python/mcp/)
  - [li-mcp](https://developers.llamaindex.ai/python/framework/module_guides/mcp/llamaindex_mcp/)
  - [adk](https://adk.dev/tools-custom/mcp-tools/)
  - [maf](https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools)
  - [spring](https://docs.spring.io/spring-ai/reference/api/mcp/mcp-client-boot-starter-docs.html)
  - [pyd](https://mcphero.app/docs/programmatic-usage/pydantic-ai)
  - [fastmcp](https://gofastmcp.com/integrations/openapi)
- **Security:**
  - [inv-tp](https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks)
  - [tob](https://blog.trailofbits.com/2025/04/21/jumping-the-line-how-mcp-servers-can-attack-you-before-you-ever-use-them/)
  - [inv-gh](https://invariantlabs.ai/blog/mcp-github-vulnerability)
  - [trifecta](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)
  - [supabase](https://generalanalysis.com/blog/supabase-mcp-blog)
  - [asana](https://www.bleepingcomputer.com/news/security/asana-warns-mcp-ai-feature-exposed-customer-data-to-other-orgs/)
  - [jfrog](https://jfrog.com/blog/2025-6514-critical-mcp-remote-rce-vulnerability/)
  - [oligo](https://www.oligo.security/blog/critical-rce-vulnerability-in-anthropic-mcp-inspector-cve-2025-49596)
  - [postmark](https://thehackernews.com/2025/09/first-malicious-mcp-server-found.html)
  - [safety-audit](https://arxiv.org/abs/2504.03767)
  - [owasp-mcp](https://owasp.org/www-project-mcp-top-10/)
  - [owasp-llm08](https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/)
  - [cosai](https://www.coalitionforsecureai.org/coalition-for-secure-ai-releases-extensive-taxonomy-for-model-context-protocol-security/)
  - [docker-gw](https://www.docker.com/blog/docker-mcp-gateway-secure-infrastructure-for-agentic-ai/)
  - [ms-gw](https://techcommunity.microsoft.com/blog/microsoft-security-blog/secure-model-context-protocol-mcp-implementation-with-azure-and-local-servers/4449660)
  - [cf-portal](https://blog.cloudflare.com/zero-trust-mcp-server-portals/)
  - [mcp-scan](https://invariantlabs.ai/blog/introducing-mcp-scan)
- **EU / enterprise:**
  - [sovereignty](https://neuraltrust.ai/blog/data-sovereignty-complete-guide)
  - [securecloud](https://www.securecloud.de/en/use-cases/mcp-server)
  - [frends](https://frends.com/insights/model-context-protocol-mcp-for-regulated-enterprises-eu-data-residency-gdpr-and-sovereign-ai-integration)
  - [maxim](https://www.getmaxim.ai/articles/top-5-mcp-gateways-for-regulated-industries-in-2026/)

[chg-2503]: https://modelcontextprotocol.io/specification/2025-03-26/changelog
[chg-2506]: https://modelcontextprotocol.io/specification/2025-06-18/changelog
[chg-2511]: https://modelcontextprotocol.io/specification/2025-11-25/changelog
[chg-2607]: https://github.com/modelcontextprotocol/modelcontextprotocol/blob/main/docs/specification/2026-07-28/changelog.mdx
[blog-2607]: https://blog.modelcontextprotocol.io/posts/2026-07-28/
[spec-res]: https://modelcontextprotocol.io/specification/2025-11-25/server/resources
[spec-sec]: https://modelcontextprotocol.io/docs/tutorials/security/security_best_practices
[workos]: https://workos.com/blog/mcp-2025-11-25-spec-update
[apps]: https://blog.modelcontextprotocol.io/posts/2026-01-26-mcp-apps/
[card]: https://github.com/modelcontextprotocol/modelcontextprotocol/issues/1649
[roadmap]: https://blog.modelcontextprotocol.io/posts/mcp-roadmap/
[aaif]: https://blog.modelcontextprotocol.io/posts/2025-12-09-mcp-joins-agentic-ai-foundation/
[a2a-aaif]: https://a2a-protocol.org/latest/blog/2026/08/27/a-new-chapter-for-a2a-joining-the-agentic-ai-foundation/
[registry]: https://blog.modelcontextprotocol.io/posts/2025-09-08-mcp-registry-preview/
[tc-oai]: https://techcrunch.com/2025/03/26/openai-adopts-rival-anthropics-standard-for-connecting-ai-models-to-data/
[oai-mcp]: https://developers.openai.com/api/docs/guides/tools-connectors-mcp
[oai-dr]: https://platform.openai.com/docs/guides/deep-research
[g-managed]: https://thenewstack.io/google-launches-managed-remote-mcp-servers-for-its-cloud-services/
[win]: https://blogs.windows.com/windowsexperience/2025/05/19/securing-the-model-context-protocol-building-a-safer-agentic-future-on-windows/
[agentcore]: https://aws.amazon.com/about-aws/whats-new/2025/10/amazon-bedrock-agentcore-available
[skills]: https://siliconangle.com/2025/12/18/anthropic-makes-agent-skills-open-standard/
[ctx-eng]: https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents
[tools]: https://www.anthropic.com/engineering/writing-tools-for-agents
[code-exec]: https://www.anthropic.com/engineering/code-execution-with-mcp
[adv-tool]: https://www.anthropic.com/engineering/advanced-tool-use
[cf-code]: https://blog.cloudflare.com/code-mode/
[li-vs]: https://www.llamaindex.ai/blog/does-mcp-kill-vector-search
[es-obsolete]: https://www.elastic.co/search-labs/blog/future-of-search-engines-indexed-search-mcp
[stackone]: https://www.stackone.com/blog/rag-vs-mcp/
[airbyte]: https://airbyte.com/agentic-data/mcp-vs-rag
[lc-ctx]: https://www.langchain.com/blog/context-engineering-for-agents
[manus]: https://medium.com/@peakji/context-engineering-for-ai-agents-lessons-from-building-manus-71883f0a67f2
[ctx-survey]: https://arxiv.org/abs/2507.13334
[agrag]: https://arxiv.org/abs/2501.09136
[rl-patterns]: https://arxiv.org/pdf/2510.05968
[schema-comp]: https://arxiv.org/abs/2605.26165
[infra]: https://www.infragistics.com/blogs/mcp-vs-rag-benchmark
[gh-tokens]: https://getunblocked.com/blog/github-mcp-token-cost/
[gh-55k]: https://dev.to/rudratosh/your-github-mcp-server-costs-55000-tokens-before-your-agent-reads-a-single-word-4eah
[mcp-tokens]: https://github.com/sd2k/mcp-tokens
[aim-mcp]: https://aimultiple.com/browser-mcp
[aim-search]: https://aimultiple.com/agentic-search
[qdrant]: https://github.com/qdrant/mcp-server-qdrant
[pinecone]: https://docs.pinecone.io/guides/assistant/mcp-server
[weaviate]: https://docs.weaviate.io/weaviate/configuration/mcp-server
[milvus]: https://github.com/zilliztech/zilliz-mcp-server
[chroma]: https://github.com/chroma-core/chroma-mcp
[elastic]: https://www.elastic.co/docs/explore-analyze/ai-features/agent-builder/mcp-server
[mongo]: https://www.mongodb.com/docs/mcp-server/tools/
[vectara]: https://github.com/vectara/vectara-mcp
[llamacloud]: https://github.com/run-llama/mcp-server-llamacloud
[haystack]: https://haystack.deepset.ai/blog/mcp-with-haystack
[ragflow]: https://ragflow.io/docs/mcp_tools
[azure]: https://learn.microsoft.com/en-us/azure/search/agentic-retrieval-how-to-retrieve
[bedrock]: https://awslabs.github.io/mcp/servers/bedrock-kb-retrieval-mcp-server
[glean]: https://docs.glean.com/administration/platform/mcp/about
[exa]: https://docs.exa.ai/reference/exa-mcp
[tavily]: https://docs.tavily.com/documentation/mcp
[neo4j]: https://graphacademy.neo4j.com/courses/genai-mcp-neo4j-tools/2-using-neo4j-mcp-tools/1-mcp-neo4j-cypher/
[neo4j-gr]: https://github.com/neo4j-field/mcp-neo4j-graphrag
[lc-mcp]: https://github.com/langchain-ai/langchain-mcp-adapters
[lc-migrate]: https://docs.langchain.com/oss/python/migrate/langchain-mcp-adapters
[oai-sdk]: https://openai.github.io/openai-agents-python/mcp/
[li-mcp]: https://developers.llamaindex.ai/python/framework/module_guides/mcp/llamaindex_mcp/
[adk]: https://adk.dev/tools-custom/mcp-tools/
[maf]: https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools
[spring]: https://docs.spring.io/spring-ai/reference/api/mcp/mcp-client-boot-starter-docs.html
[pyd]: https://mcphero.app/docs/programmatic-usage/pydantic-ai
[fastmcp]: https://gofastmcp.com/integrations/openapi
[inv-tp]: https://invariantlabs.ai/blog/mcp-security-notification-tool-poisoning-attacks
[tob]: https://blog.trailofbits.com/2025/04/21/jumping-the-line-how-mcp-servers-can-attack-you-before-you-ever-use-them/
[inv-gh]: https://invariantlabs.ai/blog/mcp-github-vulnerability
[trifecta]: https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/
[supabase]: https://generalanalysis.com/blog/supabase-mcp-blog
[asana]: https://www.bleepingcomputer.com/news/security/asana-warns-mcp-ai-feature-exposed-customer-data-to-other-orgs/
[jfrog]: https://jfrog.com/blog/2025-6514-critical-mcp-remote-rce-vulnerability/
[oligo]: https://www.oligo.security/blog/critical-rce-vulnerability-in-anthropic-mcp-inspector-cve-2025-49596
[postmark]: https://thehackernews.com/2025/09/first-malicious-mcp-server-found.html
[safety-audit]: https://arxiv.org/abs/2504.03767
[owasp-mcp]: https://owasp.org/www-project-mcp-top-10/
[owasp-llm08]: https://genai.owasp.org/llmrisk/llm082025-vector-and-embedding-weaknesses/
[cosai]: https://www.coalitionforsecureai.org/coalition-for-secure-ai-releases-extensive-taxonomy-for-model-context-protocol-security/
[docker-gw]: https://www.docker.com/blog/docker-mcp-gateway-secure-infrastructure-for-agentic-ai/
[ms-gw]: https://techcommunity.microsoft.com/blog/microsoft-security-blog/secure-model-context-protocol-mcp-implementation-with-azure-and-local-servers/4449660
[cf-portal]: https://blog.cloudflare.com/zero-trust-mcp-server-portals/
[mcp-scan]: https://invariantlabs.ai/blog/introducing-mcp-scan
[sovereignty]: https://neuraltrust.ai/blog/data-sovereignty-complete-guide
[securecloud]: https://www.securecloud.de/en/use-cases/mcp-server
[frends]: https://frends.com/insights/model-context-protocol-mcp-for-regulated-enterprises-eu-data-residency-gdpr-and-sovereign-ai-integration
[maxim]: https://www.getmaxim.ai/articles/top-5-mcp-gateways-for-regulated-industries-in-2026/
