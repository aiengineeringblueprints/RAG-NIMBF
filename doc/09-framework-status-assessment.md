# Framework-Status & Professor-Abgleich

Snapshot: 2026-06-15. Basierend auf Code-Inspektion + WhatsApp-Chat mit Prof. Weeger (11:43–12:42).

---

## 1. Was der Professor gesagt hat (Quelle: Chat 15.6.2026)

| Zeit | Wer | Aussage |
|---|---|---|
| 11:43 | Prof | „Ich glaube das meiste ist dort schon drin, aber vielleicht nicht alles. Ja können wir morgen anschauen. Des Deployment: Ist das das was du dabei haben willst im Paper oder?“ |
| 11:46 | Prof | „um 14 uhr kann ich das deployment machen“ |
| 12:06 | Prof | „Denke ja, je nach aufwand“ (zu Deployment-im-Paper) |
| 12:39 | Prof | „je nach Aufwand für dich wäre es sicher nicht so schlecht da 2 der neuen guten Modelle zu testen, am Ende ist aber das wichtige das Framework und die Art der Ergebnisse, nicht nur die Ergebnisse selbst“ |

**Kernaussage Prof:** Framework + Art der Ergebnisse > rohe Ergebnisliste. Deployment + 2 neue Modelle = Bonus bei niedrigem Aufwand.

---

## 2. Eigene Aussage „Framework steht, nur Integration ist Knackpunkt"

**Bewertung: mehrdeutig, Prof hat mit „das heißt?“ nachgefragt.**

„Integration" konnte heißen:
- (a) Anbindung **externer RAG-Systeme** ans Framework (Plugin-API vorhanden)
- (b) **Komponenten-Integration** im Framework (Chunk↔Retrieve↔Generate↔Eval)
- (c) **Deployment/Packaging** als wiederverwendbares Artefakt
- (d) **CI/CD-Integration**

Vor Telefonat klären, welche (a–d) gemeint war. Empfehlung: (a) — siehe `External_RAG_Python_Adapter.md`.

---

## 3. Code-Befund: Framework-Kriterien erfüllt

| Kriterium | Datei | Status |
|---|---|---|
| Pluggable Komponenten | `dataset_adapters.py`, `prompt_templates/`, `providers.py` | ✅ Registry-Pattern, `provider:name` Routing |
| Config-Sweep (cartesisches Produkt) | `config.py` (23 KB), `experiments/*.yaml` | ✅ |
| Black-Box-Eval externer RAG | `benchmark/adapters/base.py`, `http.py` | ✅ Protocol `RagSystemAdapter`, `register_rag_adapter` |
| Metrik-Vielfalt | `evaluation.py`, `custom_metrics.py`, `costing.py`, `resource_monitor.py` | ✅ RAGAS + nDCG/recall@k + BERTScore + Token-Kosten + GPU/CPU |
| Reproduzierbarkeit | `reproducibility.py` | ✅ Manifest + Package-Freeze pro Run |
| Tracking | `tracking.py` (570 Zeilen) | ✅ MLflow params/metrics/artefakte |
| Resumable Matrix-Runner | `benchmark/orchestration/`, `worker.py` | ✅ `plan` / `run --keep-going`, `progress.json` |
| ClearML Remote | `benchmark/clearml_task.py` | ✅ |
| Autonomer Explorer | `agentic/` (LangGraph) | ✅ |
| Tests | `tests/` | ✅ 258/258 grün |
| Papier-Artefakte | `NGEN-AI/`, `Poster/` | ✅ kompilierbar |

---

## 4. Was für ein „echtes" Framework fehlt

| Lücke | Beweis | Priorität |
|---|---|---|
| **Kein Packaging** | kein `pyproject.toml` / `setup.py` im Repo-Root; Plugin-Discovery nur via `RAG_ADAPTER_MODULES`-Env | hoch |
| **Kein Container** | kein `Dockerfile` | mittel |
| **Keine Versionierung** | kein `__version__`, keine API-Stabilität | mittel |
| **Deprecations offen** | RAGAS v1.0-Migration, `ast.Num` in `generation.py:100` (Python 3.14 bricht) | hoch |
| **Keine Schema-Validierung** | `RagSystemOutput` frozen dataclass, kein Pydantic/JSON-Schema → kaputter Adapter crasht still | mittel |
| **Kein CI** | kein `.github/workflows/` | niedrig |
| **Papier-Duplikat** | `Paper/` (2026-05-28) vs `NGEN-AI/` (2026-06-14) → Verwirrung | niedrig |

---

## 5. Themen für Telefonat

1. **„Integration" disambiguieren** — Plugin-API für externe RAG (vorhanden) vs. Packaging vs. CI. Aktuelle Code-Lage: Plugin-API steht, reale Anbindung eines dritten RAG-Systems noch offen.
2. **Deployment-Scope im Paper** — was genau? Docker-Image, Pip-Package, Architekturdiagramm, ClearML-Doku, Reproduzierbarkeits-Anhang? Prof: „je nach Aufwand" → eng fassen.
3. **2 neue Modelle benennen** — welche? Qwen3-Next, gpt-oss-120b, Llama 3.3? Sonst keine konkrete Planung.
4. **Scope-Grenze Framework vs. Eval-Harness** — aktuell beides im Repo. Paper muss trennen:
   - Framework = Pipeline + Adapter-API
   - Harness = Metriken + Tracking + Reporting
5. **Reproducibility-Claim verkaufen** — `reproducibility.py` schreibt Manifest + Freeze → starkes Argument, prominent im Paper platzieren.
6. **EVAL_MATRIX.md** — 502 Zeilen Sweep-Plan + beste Konfigs → als Beleg für systematisches Vorgehen zeigen.

---

## 6. Was „Art der Ergebnisse" (Prof) im Code bedeutet

Prof will nicht nur Scores, sondern **Charakterisierung** der Ergebnisse. Framework liefert bereits:
- **Qualität**: RAGAS (faithfulness, answer_relevancy, answer_correctness, context_precision/recall, semantic_similarity)
- **Retrieval-Qualität**: nDCG@1/3/5, recall@k, custom IR relevance
- **Generations-Qualität**: BERTScore (`roberta-large`), NLG-Metriken
- **Effizienz**: Latenz (TTFT, total), Token-Throughput, GPU-Auslastung
- **Kosten**: Token-Accounting pro Rolle, `estimated_cost_usd`
- **Stabilität**: Konfidenzintervalle via `tools/ci_plots.py` (CI-by-Faktor-Plots)
- **Pareto-Sicht**: `tools/pareto_with_perf.py` (Qualität vs. Performance)

→ im Paper als **multi-dimensionale Charakterisierung** verkaufen, nicht als bloße Score-Tabelle.

---

## 7. Verifizierbare Befehle (für Anruf-Demo falls benötigt)

```bash
# Tests
.venv/bin/python -m pytest -q            # 258 passed

# Entrypoint-Import
.venv/bin/python -c "import main; print('ok')"

# Letzte Run-Daten prüfen
.venv/bin/python -c "import json; \
  d=json.load(open('results/run87/benchmark_20260610_092401.json')); \
  print(d['num_configs'], 'configs,', len(d['results'][0]['per_sample']), 'samples')"

# Paper kompiliert
cd NGEN-AI && latexmk -pdf main.tex

# Poster kompiliert
cd Poster && latexmk -pdf main.tex
```

---

## Quellen

- `PROJECT_STATUS.md` (2026-06-15)
- `benchmark/adapters/{base,http}.py`
- `benchmark/orchestration/{worker,matrix}.py`
- `doc/External_RAG_Python_Adapter.md`
- `EVAL_MATRIX.md`
- `requirements.txt`
- WhatsApp-Chat Niclas ↔ Nico Weeger, 15.6.2026 11:43–12:42
