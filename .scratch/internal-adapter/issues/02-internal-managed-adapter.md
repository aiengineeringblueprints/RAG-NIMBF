# 02: Internal pipeline becomes a registered managed adapter

**What to build:** Asking the adapter registry for "internal" returns a real managed adapter implementing the full lifecycle (capabilities, prepare, retrieve, generate, cleanup) instead of nothing. The internal pipeline's logic — chunking, embedding, indexing, retrieval, reranking, generation — moves behind that interface. Runs keep working exactly as before via the existing orchestrator branch (this is the expand half of expand–contract; the branch removal is ticket 03).

**Blocked by:** 01 (One shared index cache key).

**Status:** ready-for-agent

- [ ] `get_rag_adapter("internal")` returns a managed adapter; no registry consumer receives None for it
- [ ] The adapter supports the component-injection slots on par with the other adapters
- [ ] Stage-level behavior (index-only) is expressible through the lifecycle interface
- [ ] Adapter tests exercise it through the managed lifecycle interface only, in the style of the existing managed-adapter lifecycle tests (tiny in-memory corpus, stubbed generator)
- [ ] A full pytest run passes; benchmark output for an internal run is byte-for-byte unchanged in structure
