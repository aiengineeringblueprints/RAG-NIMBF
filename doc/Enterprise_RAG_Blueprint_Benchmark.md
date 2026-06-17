# Enterprise RAG Blueprint benchmarken

Diese Anleitung beschreibt den konkreten Start des `Enterprise_RAG_Blueprint`-Adapters mit dem Benchmarking Framework.

## Was dieser Adapter macht

Der Adapter `examples.enterprise_rag_plugin.adapter` registriert `RAG_SYSTEM_ADAPTER=enterprise_rag` und ruft den Blueprint direkt als Python-Code auf. Beim Start baut er die Blueprint-Chroma-Collection aus dem Datensatz, der in der YAML-Datei angegeben ist. Es wird also kein separat vorbereiteter `Enterprise_RAG_Blueprint/test_documents`-Corpus benötigt.

Der Ablauf ist:

1. Framework lädt `experiments/enterprise_rag_demo.yaml`.
2. Framework lädt den dort angegebenen Datensatz, zum Beispiel `squad`.
3. Adapter ersetzt die Blueprint-Chroma-Collection `INDEX_NAME` in `VECTORDB_DIR` durch diesen Datensatz.
4. Blueprint-Retriever und Blueprint-Chain beantworten die Benchmark-Fragen.
5. Framework schreibt Metriken, Reports und MLflow-Artefakte nach `results/runN/`.

## Voraussetzungen

Installiere die Framework- und Blueprint-Abhängigkeiten:

```bash
pip install -r requirements.txt
pip install -r Enterprise_RAG_Blueprint/chain/requirements.txt
pip install -r Enterprise_RAG_Blueprint/loader/requirements.txt
```

Stelle sicher, dass die benötigten Modelle und Endpoints erreichbar sind:

```bash
ollama list
```

Wenn du den Remote-Ollama-Host aus `.env` nutzt, prüfe Auth und Modellliste:

```bash
set -a
source .env
set +a

curl -s \
  -H "Authorization: Bearer $LLM_OLLAMA_API_KEY" \
  "$LLM_OLLAMA_BASE_URL/api/tags"
```

## Konfiguration

Die zentrale Manifest-Datei ist:

```text
experiments/enterprise_rag_demo.yaml
```

Wichtige Felder:

```yaml
experiment_name: enterprise-rag-demo

dataset:
  name: squad
  sample_size: [5]

settings:
  rag_system_adapter: enterprise_rag
  rag_adapter_accepts: "chunker,embedder,llm"
  retrieval_top_k: 2
  max_new_tokens: 128
  eval_critic_llm: openai:Qwen3.5-397-A17B
  eval_critic_embedding: nomic-embed-text:latest
  ragas_enabled: true
  custom_metrics_enabled: true
  vector_db_backend: chroma

matrix:
  llm_models: ["ollama:llama3.1:8b"]
  embedding_models: ["nomic-embed-text:latest"]
  chunking_strategies: ["recursive"]
  chunk_sizes: [512]
  chunk_overlaps: [64]
  prompt_templates: ["concise"]
```

Für den Remote-Ollama-Host müssen `LLM_OLLAMA_BASE_URL` und `LLM_OLLAMA_API_KEY` in `.env` gesetzt sein. Für lokale Embeddings oder Blueprint-Defaults kommen Werte aus `Enterprise_RAG_Blueprint/.env`, zum Beispiel `VECTORDB_DIR`, `INDEX_NAME`, `OLLAMA_BASE_URL` und `RETRIEVER_DISABLE_FILTER`.

## Startbefehl

Führe den Benchmark aus dem Repository-Root aus:

```bash
set -a
source .env
source Enterprise_RAG_Blueprint/.env
set +a

PYTHONPATH=.:Enterprise_RAG_Blueprint:Enterprise_RAG_Blueprint/chain \
RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.adapter \
BENCHMARK_CONFIG_FILE=experiments/enterprise_rag_demo.yaml \
python main.py
```

`set -a` exportiert alle Variablen aus den `.env`-Dateien in die Prozessumgebung. `set +a` schaltet dieses Verhalten danach wieder aus. Das ist wichtig, weil Python-Unterprozesse und LangChain/Ollama-Clients die Variablen als echte Environment-Variablen sehen müssen.

## Resumierbarer Lauf

Für längere Runs nutze den Worker:

```bash
set -a
source .env
source Enterprise_RAG_Blueprint/.env
set +a

PYTHONPATH=.:Enterprise_RAG_Blueprint:Enterprise_RAG_Blueprint/chain \
RAG_ADAPTER_MODULES=examples.enterprise_rag_plugin.adapter \
python -m benchmark.worker run experiments/enterprise_rag_demo.yaml --keep-going
```

## Ergebnisse

Nach dem Lauf findest du die Ergebnisse unter:

```text
results/runN/
```

Wichtige Dateien und Ordner:

- `benchmark_*.json`: vollständiger Ergebnisreport
- `configs/`: per-Konfiguration JSON und QA-Logs
- `reproducibility/`: Manifest, Git-Stand und Paketliste
- `plots/`, `reports/`, `results_summary.csv`: aggregierte Auswertung, wenn Reports erzeugt wurden

## Häufige Probleme

### 504 Gateway Time-out beim Remote-Ollama-Host

Wenn Logs wie diese erscheinen:

```text
POST http://141.39.193.218/ollama/api/chat "HTTP/1.1 504 Gateway Time-out"
```

ist Auth bereits erfolgreich, aber der Remote-Proxy oder das Modell antwortet nicht schnell genug. Reduziere zuerst:

```yaml
settings:
  retrieval_top_k: 2
  max_new_tokens: 128
```

Teste den Remote-Chat direkt:

```bash
curl -s \
  -H "Authorization: Bearer $LLM_OLLAMA_API_KEY" \
  -H "Content-Type: application/json" \
  "$LLM_OLLAMA_BASE_URL/api/chat" \
  -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Answer in one sentence: What is RAG?"}],
    "stream": false,
    "options": {"num_predict": 64}
  }'
```

Wenn dieser direkte Test ebenfalls 504 liefert, muss der Remote-Proxy-Timeout erhöht oder ein schnelleres Modell verwendet werden.

### Unauthorized beim Remote-Ollama-Host

Lade `.env` und sende den Bearer-Header:

```bash
set -a
source .env
set +a

curl -s \
  -H "Authorization: Bearer $LLM_OLLAMA_API_KEY" \
  "$LLM_OLLAMA_BASE_URL/api/tags"
```

### Falscher Corpus

Der Adapter indexiert den YAML-Datensatz zur Laufzeit. `Enterprise_RAG_Blueprint/test_documents` wird für diesen Benchmarkpfad nicht genutzt. Führe `scripts/index_blueprint_corpus.py` nur aus, wenn du explizit die Blueprint-Beispieldokumente außerhalb dieses Benchmarkpfads testen willst.
