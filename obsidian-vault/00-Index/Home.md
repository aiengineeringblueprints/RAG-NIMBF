# Benchmarking Framework Vault

This vault is the working knowledge base for the RAG benchmarking project.

Start here before scanning the repository:

- [[Project Overview]]
- [[System Architecture]]
- [[Benchmark Pipeline]]
- [[Configuration Reference]]
- [[Operations Runbook]]
- [[Testing and Coverage]]
- [[Research Notes]]
- [[Known Stale Docs]]
- [[Agent Working Notes]]

Important source entrypoints:

- [main.py](../main.py)
- [config.py](../config.py)
- [benchmark/](../benchmark)
- [agentic/](../agentic)
- [EVAL_MATRIX.md](../EVAL_MATRIX.md)

Maintenance rule: whenever code behavior changes, update the affected vault note in the same change. If the vault and source disagree, source wins, then the vault should be corrected.
