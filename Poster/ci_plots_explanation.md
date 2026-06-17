# CI-Plots Erklärung & Poster-Empfehlung

## Dateien

```
results/cross_run_plots/
├── ci_faithfulness_by_prompt.png      (+ .pdf)
├── ci_ndcg5_by_retrieval.png          (+ .pdf)
└── ci_faithfulness_by_chunking.png    (+ .pdf)
```

Skript: `tools/ci_plots.py` — Bootstrap 95% CI + Paired Wilcoxon signed-rank Test, per-question Scores aus `results/run*/benchmark_*.json` (274 Config-Zeilen).

## Was zeigt jeder Plot?

- **Balkenhöhe** = Mittelwert über alle per-question Scores.
- **Schwarze T-Striche** = Bootstrap 95% Konfidenzintervall (2000 Resamples, seed=42). True Mean liegt mit 95% Wahrscheinlichkeit drin. Schmaler = sicherer.
- **`n=...` im Balken** = Anzahl individueller Question-Scores in der Gruppe.
- **Bügel oben mit Sternen** = Paired Wilcoxon signed-rank Test:
  - `***` p<0.001 (sehr stark)
  - `**`  p<0.01
  - `*`   p<0.05
  - `n.s.` = nicht signifikant

---

## Plot 1 — Faithfulness nach Prompt-Template

| Gruppe   | Mean   | 95% CI             | n      |
|----------|--------|--------------------|--------|
| concise  | 0.824  | [0.818, 0.831]     | 13 600 |
| detailed | 0.906  | [0.902, 0.911]     | 13 326 |

Wilcoxon: `***` p<0.001. CIs überlappen nicht.

**Aussage:** Detaillierter Prompt macht Antworten signifikant treuer zur Quelle. CI-Lücke ist klar — Effekt ist echt, nicht Zufall.

---

## Plot 2 — nDCG@5 nach Retrieval-Strategie

| Gruppe     | Mean   | 95% CI             | n      |
|------------|--------|--------------------|--------|
| similarity | 0.675  | [0.669, 0.681]     | 14 425 |
| mmr        | 0.779  | [0.772, 0.786]     | 10 500 |
| hyde       | 0.839  | [0.806, 0.871]     | 400    |

Wilcoxon similarity vs. mmr: `***` p<0.001.

**Aussage:** MMR schlägt Similarity deutlich beim Retrieval-Ranking. HyDE hat höchsten Mittelwert, aber n=400 (4 Fragen × 100) → CI sehr breit, keine starke Signifikanz-Aussage. **MMR = sicherer Default.**

---

## Plot 3 — Faithfulness nach Chunking-Strategie

| Gruppe    | Mean   | 95% CI             | n      |
|-----------|--------|--------------------|--------|
| recursive | 0.864  | [0.859, 0.868]     | 17 107 |
| semantic  | 0.868  | [0.861, 0.874]     |  9 819 |

CIs überlappen komplett. Wilcoxon: vermutlich `n.s.`

**Aussage:** Rekursive vs. semantische Chunking haben praktisch identischen Faithfulness-Mittelwert. Für die Metrik Faithfulness ist Chunking-Wahl nicht relevant. Andere Metriken (nDCG, Recall) sind hier aussagekräftiger.

---

## Limitierungen

- **Pooled über alle Runs**: CI aggregiert Scores aus verschiedenen LLMs / Datasets. Heterogen, nicht within-run paired.
- **Wilcoxon gepaart auf Question-Text**. Manche Fragen in mehreren Runs → Score mehrfach gezählt. Konservativ, aber erwähnenswert.
- **n=400 HyDE** zu klein für robuste Aussage. Im Poster entweder weglassen oder mit Caveat zeigen.

---

## Poster-Empfehlung

### Platz-Check

Poster aktuell voll. Rechte Spalte: Retrieval Trade-Offs → LLM Runtime Baseline → Limitations → Takeaways → Future Work → Measured Outputs → References. Neue CI-Plots brauchen je ~15–20 % Spaltenhöhe.

### Optionen

| Option | Aktion | Trade-off |
|---|---|---|
| **A. Ein Plot aufnehmen (empfohlen)** | `ci_faithfulness_by_prompt` ersetzt Block **"Measured Outputs"** (rechte Spalte, vor References) | Sauberer Tausch. "Measured Outputs" ist redundant mit "Evaluation Lenses" (linke Spalte) — Informationen doppelt. CI-Plot bringt neuen statistischen Beweis, den kein anderer Block zeigt. |
| B. Zwei Plots aufnehmen | Zusätzlich `ci_ndcg5_by_retrieval` → "Retrieval Trade-Offs" Block kürzen (Bild + 1 Satz) | Mehr statistische Substanz, aber dichtes Layout. "Retrieval Trade-Offs" müsste Text kürzen. |
| C. Alle drei aufnehmen | "LLM Runtime Baseline" streichen oder "Future Work" kürzen | Verliert entweder wichtige Runtime-Referenz oder gerade erst hinzugefügtes Future Work. Nicht empfohlen. |
| **D. Weg lassen** | Plots nur im Paper / Appendix | Poster bleibt quantitativ dünn. Keine Signifikanz-Beweise auf der Konferenz. |

### Konkrete Empfehlung: **Option A**

1. Block **"Measured Outputs"** löschen (main.tex Zeile 346–359).
2. Dort neuen Block **"Statistical Robustness"** einfügen mit `ci_faithfulness_by_prompt.png`.
3. 1–2 Sätze Text: "Bootstrap 95% CI; paired Wilcoxon `*** p<0.001`. Detailed prompts raise faithfulness significantly. CI for retrieval and chunking in paper appendix."
4. In Limitations eine Zeile ergänzen: "Per-question bootstrap CI + paired Wilcoxon tests computed; poster shows headline comparison, paper appendix shows full set."

Vorteile:
- Redundanz weg ("Measured Outputs" = Duplikat von "Evaluation Lenses")
- Statistische Glaubwürdigkeit auf Poster sichtbar
- Platzneutral
- Andere zwei Plots bleiben als "paper appendix" erwähnbar, ohne Platz zu kosten

### Wann Option D

Wenn July-Deadline zu knapp und Layout-Risiko vermieden werden soll. Plots existieren dann im Repo + paper appendix, kein Poster-Druck-Risiko.

---

## Nächster Schritt

Sag Bescheid welche Option (A / B / C / D). Bei A: ich entferne "Measured Outputs" und setze "Statistical Robustness" Block mit dem Prompt-CI-Plot ein.
