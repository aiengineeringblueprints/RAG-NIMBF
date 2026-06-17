# Externes RAG-System evaluieren — Python-Adapter (ohne HTTP)

Wie du ein bestehendes RAG-System (in Unterordner / eigener Codebasis) ans
Framework anbindest, ohne den HTTP-Adapter zu nutzen.

> **Referenz-Impl:** `examples/python_rag_plugin/demo_adapter.py` — lauffähig,
> keine Dependencies außer stdlib. Copy-paste起点. Dazugehöriger Test:
> `tests/test_demo_python_rag_plugin.py`. Manifest:
> `experiments/external_rag_demo.yaml`.

---

## TL;DR

- Option 1 (Python-Plugin-Adapter) = **Black-Box-Evaluation**. Framework sieht
  dein RAG als ganzen Block, nicht als Komponenten.
- Framework kann **nicht** in dein RAG hineingreifen. Es ruft nur `prepare()`
  einmalig und dann `answer()` pro Sample.
- Komponenten-Sweep über Framework-Matrix geht nur im `internal`-Modus oder
  wenn dein Adapter Config-Felder ausliest.

---

## 1. Was das Framework tauschen kann

| Modus | Komponenten frei tauschbar? |
|---|---|
| `RAG_SYSTEM_ADAPTER=internal` (Framework-Pipeline) | **Ja.** Chunking, Embedding, Retrieval, LLM, Reranker, Prompt — alle via YAML-Matrix sweepbar. |
| `RAG_SYSTEM_ADAPTER=myrag` (Plugin) | **Nein.** Dein Adapter kapselt alles. Framework ruft nur `prepare()` + `answer()`. |

Bei Plugin-Modus kann Framework nicht in dein RAG hineingreifen. Willst du
z.B. Framework-Embedding mit deinem Retriever testen → geht nur, wenn dein
Adapter das explizit ausliest.

---

## 2. Anforderungen ans RAG-System

### Zwingend

- Python-importierbar (Modulpfad in `RAG_ADAPTER_MODULES`)
- Klasse mit `prepare(config, data, corpus)` + `answer(sample, config) -> RagSystemOutput`
- Liefert `answer: str` pro Sample

### Stark empfohlen (sonst Metriken eingeschränkt)

| Feld fehlt | Folge |
|---|---|
| `contexts: list[str]` | RAGAS faithfulness / context_precision / context_recall → `None`. Nur Answer-Metriken. |
| `metadata[].doc_id` | nDCG / recall@k → skippt, kein Gold-Match. |
| `total_seconds` | Latency = 0 in Reports. |
| `token_count` / `input_tokens` / `output_tokens` | Token-Accounting leer, kein Cost-Estimate. |
| `answer_valid=False` bei Fehler | Sample zählt als invalid statt Crash. |

### Optional

- `ttft_seconds` (Time-to-first-token)
- `tokens_per_second`
- `gpu_usage` dict
- `estimated_cost_usd`
- `raw_reasoning` (für Thinking-Modelle)
- `raw_response` (Debugging)

---

## 3. Adapter-Plugin schreiben

Siehe das lauffähige Referenz-Beispiel in
`examples/python_rag_plugin/demo_adapter.py` — Retrieval = TF-Cosinus, Generation
= extraktiv, alle `RagSystemOutput`-Felder gefüllt. Skeleton für eigenes RAG:

`my_rag_pkg/my_adapter.py`:

```python
from benchmark.adapters import register_rag_adapter
from benchmark.adapters.base import RagSystemOutput
from my_code.rag import MyRAG  # dein bestehendes System

class MyRagAdapter:
    name = "myrag"

    def __init__(self, config):
        self.config = config
        self.rag = None

    def prepare(self, config, data, corpus=None):
        # Einmalig: Index bauen, Modelle laden
        docs = [d["context"] for d in (corpus or data)]
        self.rag = MyRAG(corpus=docs, top_k=config.retrieval_top_k)

    def answer(self, sample, config):
        result = self.rag.query(sample["question"])
        return RagSystemOutput(
            answer=result.answer,
            contexts=[c.text for c in result.passages],
            metadata=[{"doc_id": c.doc_id, "score": c.score} for c in result.passages],
            total_seconds=result.latency,
            token_count=result.tokens,
        )

register_rag_adapter("myrag", lambda config: MyRagAdapter(config))
```

Ausführen (Referenz-Beispiel):

```bash
PYTHONPATH=examples \
RAG_ADAPTER_MODULES=python_rag_plugin.demo_adapter \
RAG_SYSTEM_ADAPTER=demo_python \
BENCHMARK_CONFIG_FILE=experiments/external_rag_demo.yaml \
python main.py
```

Ausführen (eigenes Plugin):

```bash
RAG_ADAPTER_MODULES=my_rag_pkg.my_adapter \
RAG_SYSTEM_ADAPTER=myrag \
BENCHMARK_CONFIG_FILE=experiments/my_rag_eval.yaml \
python main.py
```

---

## 4. Minimal-Manifest

YAML braucht nur Dataset + Evaluator-Settings. Chunk / Retrieval / LLM-Felder
werden vom Framework ignoriert (dein Adapter übernimmt die Pipeline), müssen
aber der Schema-Erwartung entsprechen.

`experiments/my_rag_eval.yaml`:

```yaml
experiment_name: my-rag-eval

dataset:
  name: squad          # oder jsonl + DATASET_PATH
  sample_size: [100]

settings:
  rag_system_adapter: myrag
  retrieval_top_k: 5            # landet in config, vom Adapter lesbar
  eval_critic_llm: ollama:gemma3:12b
  eval_critic_embedding: nomic-embed-text:latest
  ragas_enabled: true
  custom_metrics_enabled: true
  vector_db_backend: chroma     # wird ignoriert, aber Pflichtfeld

matrix:
  llm_models: ["n/a"]           # Dummy, Adapter übernimmt
  embedding_models: ["n/a"]
  chunking_strategies: ["none"]
  chunk_sizes: [0]
  chunk_overlaps: [0]
  prompt_templates: ["concise"]
```

---

## 5. Komponenten-Sweep — drei Wege

### Weg A — Eigenes RAG über Adapter-Config parametrisieren

Adapter liest Felder aus `config`:

```python
def prepare(self, config, data, corpus=None):
    self.rag = MyRAG(
        retriever_type=config.reranker_model or "bm25",  # zweckentfremdet
        top_k=config.retrieval_top_k,
        llm=config.llm_model,                           # deiner
    )
```

Dann via YAML-Matrix durchsweepen. **Hacky** — Framework-Felder werden
umbraucht. Sauberer: Weg B.

### Weg B — Eigene Config-Felder via `BenchmarkConfig` erweitern

`config.py` Dataclass erweitern um z.B. `my_rag_retriever`, `my_rag_llm`.
Adapter liest die direkt. Matrix-Expansion unterstützt sie automatisch. Mehr
Code, dafür sauber getrennt.

### Weg C — Hybrid: Framework für Sub-Komponente nutzen

Beispiel: Dein RAG nutzt Framework-Chunking + Framework-Embedding + eigenen
Retriever + eigene Generation. Baue Adapter, der intern `benchmark.chunking`,
`benchmark.retrieval` importiert und mischt. Möglich, aber dann bist du tief
in Framework-Internas — empfehle eher, das als `internal`-Pipeline-Variante zu
bauen (z.B. eigenen Retriever in `benchmark/retrieval.py` registrieren).

---

## 6. Bottom Line

| Ziel | Weg |
|---|---|
| Nur Evaluierung deines fertigen RAG | **Option 1** (Plugin) — Anforderungen aus Abschnitt 2 reichen |
| Sweep über RAG-Komponenten | `internal`-Pipeline nutzen, oder RAG über Config-Felder parametrisieren (Weg A/B) |
| Mischung | Adapter, der je nach Config-Feld eigene vs. Framework-Komponente nutzt |

---

## 7. Siehe auch

- `doc/External_RAG_System_Usage.md` — HTTP-Adapter-Variante
- `benchmark/adapters/base.py` — `RagSystemAdapter` Protocol + `RagSystemOutput` Dataclass
- `benchmark/adapters/__init__.py` — `register_rag_adapter` Registry
- `benchmark/adapters/components.py` — `ComponentBundle` + `build_components`
- `main.py:199-300` — Adapter-Wiring im Benchmark-Loop
- `examples/http_rag_server.py` — Beispiel-Endpoint (HTTP)
- `examples/enterprise_rag_plugin/adapter.py` — Referenz-Adapter mit Component Injection

---

## 8. Component Injection (optional)

Wenn dein RAG Framework-gebaute Komponenten (Chunker, Embedder, LLM, ...)
übernehmen kann, implementiere zwei zusätzliche Methoden auf deiner
Adapter-Klasse:

```python
def supports_components(self) -> dict[str, bool]:
    return {
        "chunker": True, "embedder": True, "retriever": False,
        "reranker": False, "llm": True, "prompt": False,
    }

def set_components(self, bundle: ComponentBundle) -> None:
    self.bundle = bundle
```

Das Framework ruft `supports_components()` auf, baut die angeforderten Slots
aus `.env`/YAML, und ruft `set_components()` einmal vor `prepare()` auf. Slots
mit `False` werden nicht gebaut; Slots mit `True`, aber in `.env` nicht
gesetzt, kommen als `None` an.

`.env` des Users:

```env
RAG_SYSTEM_ADAPTER=myrag
RAG_ADAPTER_MODULES=my_pkg.my_adapter
RAG_ADAPTER_ACCEPTS=chunker,embedder,llm
CHUNKING_STRATEGIES=recursive
EMBEDDING_MODELS=ollama:nomic-embed-text
LLM_MODELS=ollama:gemma3:4b
```

Implementiert dein Adapter nur `set_components()` (ohne `supports_components()`),
ist `RAG_ADAPTER_ACCEPTS` aus `.env` alleinige Wahrheitsquelle.

Referenz-Implementierung: `examples/enterprise_rag_plugin/adapter.py`.
Manifest: `experiments/enterprise_rag_demo.yaml`.
