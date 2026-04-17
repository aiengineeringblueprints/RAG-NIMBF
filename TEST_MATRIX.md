# Benchmarking Matrix - Alle Kombinationen

## Feste Parameter
| Parameter | Wert |
|---|---|
| Embedding | nomic-embed-text:latest (Ollama) |
| Reranker | Deaktiviert |
| HyDE | Deaktiviert |
| Dataset | ragas-wikiqa |

## Kurznamen
| Kurzname | Original |
|---|---|
| Qwen3-32B | Qwen/Qwen3-32B-AWQ |
| Qwen3.5-397B | Qwen3.5-397B-A17B |
| GPT-OSS-20B | gpt-oss:20b |
| Qwen3.5-35B | qwen3.5:35b |

## Legende Scores
| Spalte | Metrik | Beschreibung |
|---|---|---|
| Faith | ragas_faithfulness | Treue gegenuber dem Kontext (0-1, hoeher besser) |
| Rel | ragas_answer_relevancy | Relevanz der Antwort (0-1, hoeher besser) |
| Corr | ragas_answer_correctness | Korrektheit der Antwort (0-1, hoeher besser) |
| Prec | ragas_context_precision | Prazision des Kontexts (0-1, hoeher besser) |
| Rec | ragas_context_recall | Recall des Kontexts (0-1, hoeher besser) |
| Zeit | total_time_seconds | Gesamtlaufzeit in Sekunden |
| TTFT | avg_ttft_seconds | Durchschn. Time-to-First-Token |
| TPS | avg_tokens_per_second | Durchschn. Tokens pro Sekunde |
| N | - | Anzahl Fragen im Test |

> **Hinweis:** Runs mit N=1 oder N=50 sind Testlaeufe. Runs mit N=100 oder N=150 sind vollstaendige Benchmarks.

## Matrix (128 Kombinationen)

| # | LLM | Chunking | Size | Overlap | Retrieval | Template | Faith | Rel | Corr | Prec | Rec | Zeit | TTFT | TPS | N | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Qwen3-32B | Recursive | 1000 | 100 | Similarity | Concise | 0.200 | 0.507 | 0.355 | 0.421 | 0.380 | 2840s | 0.53 | 11.5 | 50 | Test (N=50) |
| 2 | Qwen3-32B | Recursive | 1000 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 3 | Qwen3-32B | Recursive | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 4 | Qwen3-32B | Recursive | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 5 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Concise | 0.514 | 0.332 | 0.367 | 0.399 | 0.353 | 12347s | 15.54 | 61.6 | 150 | Getestet |
| 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 7 | Qwen3-32B | Recursive | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 8 | Qwen3-32B | Recursive | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 9 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 10 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 11 | Qwen3-32B | Recursive | 500 | 100 | MMR | Concise | 0.821 | 0.490 | 0.323 | 0.463 | 0.310 | 4926s | 0.81 | 0.0 | 100 | Getestet |
| 12 | Qwen3-32B | Recursive | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 13 | Qwen3-32B | Recursive | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 14 | Qwen3-32B | Recursive | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 15 | Qwen3-32B | Recursive | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 16 | Qwen3-32B | Recursive | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 17 | Qwen3-32B | Semantic | 1000 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 18 | Qwen3-32B | Semantic | 1000 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 19 | Qwen3-32B | Semantic | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 20 | Qwen3-32B | Semantic | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 21 | Qwen3-32B | Semantic | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 22 | Qwen3-32B | Semantic | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 23 | Qwen3-32B | Semantic | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 24 | Qwen3-32B | Semantic | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 25 | Qwen3-32B | Semantic | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 26 | Qwen3-32B | Semantic | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 27 | Qwen3-32B | Semantic | 500 | 100 | MMR | Concise | 0.877 | 0.560 | 0.386 | 0.625 | 0.470 | 5205s | 1.21 | 0.0 | 100 | Getestet |
| 28 | Qwen3-32B | Semantic | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 29 | Qwen3-32B | Semantic | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 30 | Qwen3-32B | Semantic | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 31 | Qwen3-32B | Semantic | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 32 | Qwen3-32B | Semantic | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 33 | Qwen3.5-397B | Recursive | 1000 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 34 | Qwen3.5-397B | Recursive | 1000 | 100 | Similarity | Detailed | 0.955 | 0.564 | 0.345 | 0.431 | 0.340 | 5162s | 0.75 | 0.0 | 50 | Test (N=50) |
| 35 | Qwen3.5-397B | Recursive | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 36 | Qwen3.5-397B | Recursive | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 37 | Qwen3.5-397B | Recursive | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 38 | Qwen3.5-397B | Recursive | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 39 | Qwen3.5-397B | Recursive | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 40 | Qwen3.5-397B | Recursive | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 41 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Concise | 0.888 | 0.525 | 0.374 | 0.441 | 0.430 | 6842s | 0.71 | 0.0 | 100 | Getestet |
| 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 0.931 | 0.564 | 0.338 | 0.402 | 0.360 | 4747s | 0.75 | 0.0 | 50 | Test (N=50) |
| 43 | Qwen3.5-397B | Recursive | 500 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 44 | Qwen3.5-397B | Recursive | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 45 | Qwen3.5-397B | Recursive | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 46 | Qwen3.5-397B | Recursive | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 47 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 48 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 49 | Qwen3.5-397B | Semantic | 1000 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 50 | Qwen3.5-397B | Semantic | 1000 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 51 | Qwen3.5-397B | Semantic | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 52 | Qwen3.5-397B | Semantic | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 53 | Qwen3.5-397B | Semantic | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 54 | Qwen3.5-397B | Semantic | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 55 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 56 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 57 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Concise | 0.921 | 0.559 | 0.420 | 0.599 | 0.590 | 12356s | 2.98 | 0.0 | 100 | Getestet |
| 58 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 59 | Qwen3.5-397B | Semantic | 500 | 100 | MMR | Concise | 0.887 | 0.521 | 0.386 | 0.607 | 0.470 | 5556s | 1.16 | 0.0 | 100 | Getestet |
| 60 | Qwen3.5-397B | Semantic | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 61 | Qwen3.5-397B | Semantic | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 62 | Qwen3.5-397B | Semantic | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 63 | Qwen3.5-397B | Semantic | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 64 | Qwen3.5-397B | Semantic | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 65 | GPT-OSS-20B | Recursive | 1000 | 100 | Similarity | Concise | 0.265 | 0.501 | 0.441 | 0.429 | 0.400 | 3416s | 9.47 | 51.2 | 50 | Test (N=50) |
| 66 | GPT-OSS-20B | Recursive | 1000 | 100 | Similarity | Detailed | 0.568 | 0.446 | 0.277 | 0.432 | 0.400 | 4194s | 0.85 | 51.7 | 50 | Test (N=50) |
| 67 | GPT-OSS-20B | Recursive | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 68 | GPT-OSS-20B | Recursive | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 69 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 70 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 71 | GPT-OSS-20B | Recursive | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 72 | GPT-OSS-20B | Recursive | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 73 | GPT-OSS-20B | Recursive | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 74 | GPT-OSS-20B | Recursive | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 75 | GPT-OSS-20B | Recursive | 500 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 76 | GPT-OSS-20B | Recursive | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 77 | GPT-OSS-20B | Recursive | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 78 | GPT-OSS-20B | Recursive | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 79 | GPT-OSS-20B | Recursive | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 80 | GPT-OSS-20B | Recursive | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 81 | GPT-OSS-20B | Semantic | 1000 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 82 | GPT-OSS-20B | Semantic | 1000 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 84 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 85 | GPT-OSS-20B | Semantic | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 86 | GPT-OSS-20B | Semantic | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 87 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 88 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 89 | GPT-OSS-20B | Semantic | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 90 | GPT-OSS-20B | Semantic | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 91 | GPT-OSS-20B | Semantic | 500 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 92 | GPT-OSS-20B | Semantic | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 93 | GPT-OSS-20B | Semantic | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 94 | GPT-OSS-20B | Semantic | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 95 | GPT-OSS-20B | Semantic | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 96 | GPT-OSS-20B | Semantic | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 97 | Qwen3.5-35B | Recursive | 1000 | 100 | Similarity | Concise | 0.160 | 0.520 | 0.405 | 0.432 | 0.360 | 2890s | 1.32 | 5.1 | 50 | Test (N=50) |
| 98 | Qwen3.5-35B | Recursive | 1000 | 100 | Similarity | Detailed | 0.810 | 0.513 | 0.229 | 0.407 | 0.360 | 4746s | 4.44 | 42.7 | 50 | Test (N=50) |
| 99 | Qwen3.5-35B | Recursive | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 100 | Qwen3.5-35B | Recursive | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 101 | Qwen3.5-35B | Recursive | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 102 | Qwen3.5-35B | Recursive | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 103 | Qwen3.5-35B | Recursive | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 104 | Qwen3.5-35B | Recursive | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 105 | Qwen3.5-35B | Recursive | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 106 | Qwen3.5-35B | Recursive | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 107 | Qwen3.5-35B | Recursive | 500 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 108 | Qwen3.5-35B | Recursive | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 109 | Qwen3.5-35B | Recursive | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 110 | Qwen3.5-35B | Recursive | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 111 | Qwen3.5-35B | Recursive | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 112 | Qwen3.5-35B | Recursive | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 113 | Qwen3.5-35B | Semantic | 1000 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 114 | Qwen3.5-35B | Semantic | 1000 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 115 | Qwen3.5-35B | Semantic | 1000 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 116 | Qwen3.5-35B | Semantic | 1000 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 117 | Qwen3.5-35B | Semantic | 1000 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 118 | Qwen3.5-35B | Semantic | 1000 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 119 | Qwen3.5-35B | Semantic | 1000 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 120 | Qwen3.5-35B | Semantic | 1000 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 121 | Qwen3.5-35B | Semantic | 500 | 100 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 122 | Qwen3.5-35B | Semantic | 500 | 100 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 123 | Qwen3.5-35B | Semantic | 500 | 100 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 124 | Qwen3.5-35B | Semantic | 500 | 100 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 125 | Qwen3.5-35B | Semantic | 500 | 200 | Similarity | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 126 | Qwen3.5-35B | Semantic | 500 | 200 | Similarity | Detailed | - | - | - | - | - | - | - | - | - | Offen |
| 127 | Qwen3.5-35B | Semantic | 500 | 200 | MMR | Concise | - | - | - | - | - | - | - | - | - | Offen |
| 128 | Qwen3.5-35B | Semantic | 500 | 200 | MMR | Detailed | - | - | - | - | - | - | - | - | - | Offen |

## Zusammenfassung
| Metrik | Wert |
|---|---|
| Gesamt Kombinationen | 128 |
| Getestet (mit Ergebnissen) | 13 |
| Offen | 115 |
