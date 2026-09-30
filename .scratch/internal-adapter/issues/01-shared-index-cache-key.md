# 01: One shared index cache key

**What to build:** A single index-cache-key function that both the internal pipeline path and the component-injection path use to derive vector-store collection identities. The key includes dataset name, dataset subset, sample size, corpus fingerprint, and embedding provider. Two experiments over different datasets or embeddings can never silently share a collection, regardless of whether components were injected.

**Blocked by:** None (can start immediately). This is the prefactor that fixes a latent correctness bug before the adapter extraction.

**Status:** ready-for-agent

- [ ] One public cache-key function exists; both the internal retrieval path and the injected-component factory use it (the weaker injected variant is gone)
- [ ] Distinct dataset name, subset, sample size, corpus fingerprint, or embedding provider each produce a distinct key; identical inputs produce identical keys (pure-function tests)
- [ ] Existing test suite passes; no behavioral change other than the strengthened key
