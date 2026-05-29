Nutzung:

  python -m benchmark.clearml_task experiments/full-grid-example.yaml \
    --project-name "RAG Benchmarking" \
    --task-name rag_eval_baseline

  clearml-agent daemon --queue rag-benchmark-gpu

  Danach in ClearML UI: Task klonen, Hyperparameters ändern, Clone in rag-
  benchmark-gpu enqueuen.

  Zusätzlich aktualisiert:

  - requirements.txt: clearml>=1.16
  - README.md: ClearML-Agent-Workflow
  - .env.example: ClearML-Server/Agent-Hinweise
  - Vault-Doku unter obsidian-vault/

  Verifikation:

  - .venv/bin/python -m py_compile ... bestanden
  - .venv/bin/python -m pytest tests/test_orchestration.py bestanden: 7 passed

  Hinweis: Ich habe keinen echten ClearML-Run gestartet, weil dafür ein
  konfigurierter ClearML Server/Agent nötig ist.
