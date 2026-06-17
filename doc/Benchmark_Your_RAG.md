# Eigenes RAG mit dem Framework benchmarken

Diese Anleitung beschreibt die Schritte, die ein Nutzer des Frameworks durchführen muss, um ein eigenes RAG-System zu testen.

## 1. Integrationsart wählen

Es gibt drei sinnvolle Wege:

| Ziel | Empfohlener Weg |
| --- | --- |
| Bestehendes RAG hat einen HTTP-Endpunkt | HTTP-Adapter (`RAG_SYSTEM_ADAPTER=http`) |
| Bestehendes RAG ist Python-Code im Repo oder importierbar | Python-Adapter-Plugin |
| Du willst die Framework-Pipeline selbst benchmarken | Interner Modus (`RAG_SYSTEM_ADAPTER=internal`) |

Für die meisten externen Systeme ist der HTTP-Adapter der einfachste Einstieg. Ein Python-Adapter ist sinnvoll, wenn dein RAG keinen HTTP-Endpunkt hat oder du Framework-Komponenten wie Chunker, Embedder oder LLM direkt injizieren willst.

## 2. Datensatz vorbereiten

Der Benchmark braucht pro Sample mindestens:

- `question`: Frage an das RAG
- `ground_truth`: erwartete Referenzantwort
- `context`: Referenzkontext oder Dokumenttext, der indexiert oder für Metriken genutzt werden kann
- `metadata`: optional, aber empfohlen

Für eigene Daten ist JSONL am einfachsten:

```jsonl
{"question":"What is X?","ground_truth":"X is ...","context":"Document text containing the answer.","metadata":{"id":"1"}}
{"question":"Who created Y?","ground_truth":"Alice","context":"Y was created by Alice in 2024.","metadata":{"id":"2"}}
```

Lege die Datei zum Beispiel hier ab:

```text
examples/my_rag_dataset.jsonl
```

## 3. Manifest anlegen

Erstelle eine YAML-Datei, zum Beispiel:

```text
experiments/my_rag_eval.yaml
```

Minimalbeispiel für ein externes RAG:

```yaml
experiment_name: my-rag-eval

dataset:
  name: jsonl
  path: examples/my_rag_dataset.jsonl
  sample_size: [20]

settings:
  rag_system_adapter: http
  retrieval_top_k: 5
  eval_critic_llm: ollama:gemma3:12b
  eval_critic_embedding: nomic-embed-text:latest
  ragas_enabled: false
  custom_metrics_enabled: false
  vector_db_backend: chroma

matrix:
  llm_models: ["ollama:llama3.1:8b"]
  embedding_models: ["nomic-embed-text:latest"]
  chunking_strategies: ["recursive"]
  chunk_sizes: [512]
  chunk_overlaps: [64]
  prompt_templates: ["concise"]
```

Für Smoke-Tests sollten `ragas_enabled` und `custom_metrics_enabled` zuerst `false` sein. Aktiviere sie erst, wenn Antwort- und Kontextfelder korrekt ankommen.

## 4. HTTP-Adapter verwenden

Dein RAG-Service muss einen JSON `POST`-Endpunkt anbieten. Das Framework sendet:

```json
{
  "question": "What is the answer?",
  "metadata": {"id": "sample-1"},
  "ground_truth": "Expected answer",
  "config": {
    "retrieval_top_k": 5,
    "dataset_name": "jsonl",
    "prompt_template": "concise"
  }
}
```

Empfohlene Antwort:

```json
{
  "answer": "Generated answer",
  "contexts": ["Retrieved passage 1", "Retrieved passage 2"],
  "metadata": [{"doc_id": "doc-1", "score": 0.91}],
  "timings": {"total_seconds": 1.47, "token_count": 128}
}
```

Start:

```bash
set -a
source .env
set +a

RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=http://localhost:8000/query \
RAG_HTTP_ANSWER_FIELD=answer \
RAG_HTTP_CONTEXTS_FIELD=contexts \
BENCHMARK_CONFIG_FILE=experiments/my_rag_eval.yaml \
python main.py
```

Falls dein Endpoint Auth braucht:

```bash
RAG_HTTP_AUTH_HEADER=Authorization \
RAG_HTTP_AUTH_VALUE="Bearer $RAG_API_TOKEN" \
RAG_SYSTEM_ADAPTER=http \
RAG_HTTP_ENDPOINT_URL=https://example.com/query \
BENCHMARK_CONFIG_FILE=experiments/my_rag_eval.yaml \
python main.py
```

## 5. Python-Adapter verwenden

Wenn dein RAG importierbarer Python-Code ist, schreibe ein Adaptermodul:

```python
from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput

class MyRagAdapter:
    name = "myrag"

    def __init__(self, config):
        self.config = config
        self.rag = None

    def prepare(self, config, data, corpus=None):
        docs = corpus if corpus else data
        self.rag = build_my_rag(docs=docs, top_k=config.retrieval_top_k)

    def answer(self, sample, config):
        result = self.rag.query(sample["question"])
        return RagSystemOutput(
            answer=result.answer,
            contexts=result.contexts,
            metadata=result.metadata,
            total_seconds=result.total_seconds,
            token_count=result.token_count,
            answer_valid=bool(result.answer.strip()),
        )

register_rag_adapter("myrag", lambda config: MyRagAdapter(config))
```

Start:

```bash
set -a
source .env
set +a

PYTHONPATH=.:path/to/your/package \
RAG_ADAPTER_MODULES=my_package.my_adapter \
RAG_SYSTEM_ADAPTER=myrag \
BENCHMARK_CONFIG_FILE=experiments/my_rag_eval.yaml \
python main.py
```

## 6. Optional: Framework-Komponenten injizieren

Wenn dein Python-Adapter Framework-Komponenten nutzen soll, implementiere zusätzlich:

```python
def supports_components(self):
    return {
        "chunker": True,
        "embedder": True,
        "retriever": False,
        "reranker": False,
        "llm": True,
        "prompt": False,
    }

def set_components(self, bundle):
    self.bundle = bundle
```

Dann baut das Framework die gewünschten Komponenten aus YAML und `.env` und übergibt sie vor `prepare()`.

Beispiel: `examples/enterprise_rag_plugin/adapter.py`.

## 7. Lauf starten und Ergebnisse prüfen

Normaler Lauf:

```bash
BENCHMARK_CONFIG_FILE=experiments/my_rag_eval.yaml python main.py
```

Resumierbarer Lauf:

```bash
python -m benchmark.worker run experiments/my_rag_eval.yaml --keep-going
```

Ergebnisse liegen unter:

```text
results/runN/
```

Prüfe zuerst:

- ob `answer_valid` hoch ist
- ob `contexts` gefüllt sind
- ob Latenzen plausibel sind
- ob RAGAS/custom metrics erst dann aktiviert werden, wenn der Smoke-Test stabil ist

## 8. Typischer Ablauf für neue Nutzer

1. Erstelle oder wähle einen kleinen Datensatz mit 5 bis 20 Samples.
2. Starte mit `ragas_enabled=false` und `custom_metrics_enabled=false`.
3. Integriere dein RAG per HTTP oder Python-Adapter.
4. Prüfe `results/runN/configs/*_qa.json`, ob Fragen, Antworten und Kontexte stimmen.
5. Aktiviere RAGAS und Custom Metrics.
6. Erhöhe `sample_size`.
7. Nutze `benchmark.worker --keep-going` für längere Läufe.
