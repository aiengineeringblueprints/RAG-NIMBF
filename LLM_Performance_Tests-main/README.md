# LLM Performance Benchmark

Vergleicht verschiedene LLM-Modelle (Cloud-APIs und Self-Hosted) hinsichtlich Latenz, Durchsatz, Streaming-Verhalten (TTFT/TPOT) und Skalierungsverhalten bei paralleler vs. sequenzieller Verarbeitung.

## Quickstart

```bash
# 1. Abhängigkeiten installieren
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Konfiguration anlegen
cp .env.example .env
# → .env anpassen: API-Keys, Modelle, Prompts, Parameter

# 3. Benchmark starten
python main.py
```

## Konfiguration (.env)

| Variable | Beschreibung | Beispiel |
|---|---|---|
| `MODELS` | JSON-Array mit Modell-Definitionen | siehe `.env.example` |
| `PROMPTS_FILE` | Pfad zu einer JSON-Datei mit Prompts | `prompts.json` |
| `CALL_COUNTS` | Anzahl verschiedener Prompts pro Experiment | `[1,3,6,10]` |
| `PROMPT_SELECTION` | `first` (erste N) oder `random` (zufällig N) | `first` |
| `MAX_TOKENS_LIST` | Token-Limits pro Experiment | `[256,1024]` |
| `TIMEOUT_S` | Timeout pro Call in Sekunden | `600` |

### Modell-Tags (optional, aber empfohlen)

Jedes Modell kann freie Metadaten als `tags` mitbringen. Die landen in MLflow,
in der Terminal-Vergleichstabelle und in den Master-Tabellen — und machen
Vergleiche zwischen verschiedenen Hardwares deutlich übersichtlicher:

```json
{
  "provider": "openai_compatible",
  "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4",
  "label": "Spark Nemotron-3-Nano",
  "base_url": "http://...",
  "api_key": "...",
  "tags": {
    "hardware": "DGX Spark",
    "params_total": "30B",
    "params_active": "3.5B",
    "architecture": "MoE Mamba2-Transformer",
    "quantization": "NVFP4",
    "kv_cache": "fp8"
  }
}
```

Übliche Tag-Keys: `hardware`, `params_total`, `params_active`, `architecture`,
`quantization`, `kv_cache`. Du kannst aber beliebige eigene Keys vergeben.

### Unterstutzte Provider

- **`anthropic`** — Anthropic API (Claude)
- **`openai`** — OpenAI API (GPT)
- **`openai_compatible`** — beliebiger OpenAI-kompatibler Server (vLLM, LM Studio, llama.cpp)
- **`ollama`** — Native Ollama API (`/api/chat` mit Streaming)

### Prompt-Pool

Prompts werden in `prompts.json` definiert (JSON-Array mit Strings). `CALL_COUNTS` bestimmt, wie viele **verschiedene** Prompts pro Experiment verwendet werden — jeder Prompt wird genau 1x aufgerufen, keine Wiederholungen (Cache-Vermeidung).

## Was passiert bei einem Run?

Fur jede Kombination aus `CALL_COUNTS` x `MAX_TOKENS_LIST`:

1. **N Prompts** werden aus dem Pool ausgewahlt (first/random)
2. **Sequenziell**: Alle N Prompts werden nacheinander an jedes Modell gesendet
3. **Parallel**: Alle N Prompts werden gleichzeitig an jedes Modell gesendet
4. **Auswertung**: Latenz (Ø, Median, P95), Wall-Zeit, Erfolgsrate, Tokens/s, TTFT, TPOT, Inter-Token-Latenz

### Streaming-Metriken

Da alle Provider (OpenAI-kompatibel, Ollama) per Streaming abgefragt werden, werden zusätzlich erfasst:

| Metrik | Bedeutung | Warum wichtig (z.B. für RAG/Chat) |
|---|---|---|
| **TTFT** (Time To First Token) | Zeit vom Request bis das erste Token eintrifft | "Wie schnell fühlt sich der Server an?" — größter UX-Faktor bei Streaming-UIs |
| **TPOT** (Time Per Output Token) | Mittlere Zeit zwischen aufeinanderfolgenden Tokens | Bestimmt das wahrgenommene Lese-/Tipp-Tempo |
| **ITL P50/P95** (Inter-Token-Latenz) | Median und 95. Perzentil der Token-Abstände | Zeigt Stabilität — hohes P95 = ruckelige Ausgabe |
| **Gen Tok/s** | Output-Tokens / (Latenz − TTFT) | Reine Generierungs-Rate ohne Anfangswartezeit |
| **Gesamt Tok/s** | Output-Tokens / Latenz | Klassischer Durchsatz inkl. TTFT |

### Terminal-Ausgabe

Jedes Experiment zeigt zwei Tabellen — eine Latenz-Übersicht und eine Streaming-Metriken-Tabelle:

```
ÜBERSICHT: Sequential vs. Parallel — pro Modell
┌────────────────────────┬───────┬───────┬────────┬────────┬────────────┐
│ Modell                 │ SEQ N │ SEQ Ø │ SEQ P95│ PAR Ø  │ PAR Wall   │
├────────────────────────┼───────┼───────┼────────┼────────┼────────────┤
│ RTX5090 Qwen3-32b-AWQ  │ 10    │ 4.21s │ 6.30s  │ 8.03s  │ 10.15s     │
│ Spark Nemotron-3-Nano  │ 10    │ 3.88s │ 5.10s  │ 6.52s  │  8.91s     │
└────────────────────────┴───────┴───────┴────────┴────────┴────────────┘

STREAMING-METRIKEN: TTFT, TPOT, Generierungs-Durchsatz
┌────────────────────────┬───────┬─────────┬──────────┬─────────┬───────────┬──────────────┐
│ Modell                 │ Modus │ TTFT Ø  │ TTFT P95 │ TPOT Ø  │ Gen Tok/s │ Gesamt Tok/s │
├────────────────────────┼───────┼─────────┼──────────┼─────────┼───────────┼──────────────┤
│ RTX5090 Qwen3-32b-AWQ  │ seq   │  450ms  │   620ms  │  16ms   │   62.4    │     58.7     │
│ RTX5090 Qwen3-32b-AWQ  │ par   │ 1540ms  │  2100ms  │  18ms   │   55.1    │     49.3     │
│ Spark Nemotron-3-Nano  │ seq   │  220ms  │   380ms  │  13ms   │   75.2    │     71.9     │
│ Spark Nemotron-3-Nano  │ par   │  890ms  │  1450ms  │  15ms   │   66.5    │     61.4     │
└────────────────────────┴───────┴─────────┴──────────┴─────────┴───────────┴──────────────┘
```

Am Ende folgt eine Trend-Ubersicht die zeigt, wie sich die Modelle bei steigender Parallelitat verhalten.

## Ergebnisse in MLflow anschauen

```bash
# MLflow UI starten (aus dem Projektverzeichnis)
mlflow ui --backend-store-uri "sqlite:///mlflow.db"
```

Dann `http://localhost:5000` im Browser offnen.

> **Pro Programmstart wird ein neues Experiment angelegt**, mit Zeitstempel im
> Namen (z.B. `llm-benchmark · 2026-04-07_15-22-30`). So bleiben verschiedene
> Test-Durchläufe sauber getrennt und die Vergleiche zwischen Konfigurationen
> sind eindeutig.

### MLflow-Struktur

**Experiment-Runs** (ein Run pro Parameterkombination):

- Run-Name: `N=6  |  T=1024`
- **Metrics** (pro Modell × Modus als separate Keys):
  - `<modell>_wall_s`, `<modell>_mean_s`, `<modell>_p95_s` — Latenz-Kennzahlen
  - `<modell>_tok_s` — Gesamt-Throughput inkl. TTFT
  - `<modell>_ttft_ms`, `<modell>_ttft_p95_ms` — Time To First Token
  - `<modell>_tpot_ms` — Time Per Output Token
  - `<modell>_gen_tok_s` — Reine Generierungs-Tokens/s ohne TTFT
  - `<modell>_speedup` — Wall-Zeit Speedup parallel vs. sequenziell
- **Artifacts**:
  - `modellvergleich_rag.json` — **Modell-zentrierte RAG-Sicht** mit Hardware-Spalte
  - `vergleich.json` — Übersichtstabelle mit allen Latenz- und Streaming-Metriken
  - `details.json` — Einzel-Calls mit TTFT/TPOT/Gen Tok/s pro Call
  - `antworten.json` — vollständige Frage-Antwort-Paare
  - `wall_zeit_vergleich.png` — Balkenplot Wall-Zeit seq vs. par

**Master-Run** (ein Run pro gesamtem Test-Durchlauf):

- Run-Name: `Modellvergleich · Master (RAG)`
- **Artifact**: `modellvergleich_rag_master.json` — Master-Tabelle mit allen
  Modellen × allen Konfigurationen × Hardware in **einer** Tabelle. Ideal um
  über mehrere Hardwares und Quantisierungen hinweg auf einen Blick zu sehen,
  welche Kombination für RAG am besten passt.

**Trend-Runs** (ein Run pro Modell, nur bei mehreren CALL_COUNTS):

- Run-Name: `Trend · <Modell>`
- **Metrics-Tab**: automatische Liniendiagramme uber n_calls:
  - `par_wall_s`, `par_latency_mean_s` — Wall-Zeit/Ø-Latenz parallel
  - `par_ttft_ms`, `par_tpot_ms` — Streaming-Latenz unter Last
  - `par_gen_tok_s` — Generierungs-Durchsatz
  - `par_speedup` — fallt ab = Sattigungspunkt erreicht
  - `seq_*` — gleiche Metriken sequenziell (Baseline)

### Modelle vergleichen

1. In der Runs-Liste mehrere Runs per Checkbox auswahlen
2. Oben auf **Compare** klicken
3. Metriken direkt nebeneinander sehen

### Sattigungspunkt finden

`CALL_COUNTS=[1,3,6,10,15,20]` setzen und den Trend-Run des Modells offnen. Im Speedup-Chart zeigt der Knickpunkt, ab wie vielen parallelen Calls das Modell nicht mehr sinnvoll skaliert.
