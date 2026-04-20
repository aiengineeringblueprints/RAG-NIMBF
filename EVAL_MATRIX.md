# Benchmarking Matrix - Alle Kombinationen

## Feste Parameter
| Parameter | Wert                             |
| --------- | -------------------------------- |
| Embedding | nomic-embed-text:latest (Ollama) |
| Reranker  | Deaktiviert                      |
| HyDE      | Deaktiviert                      |
| Dataset   | squad                            |
| Eval_LLM  | Qwen3.5-397B-A17B_No_Thinking    |

###  Zu testende Parameter

- Chunk Size
- Chunk Overlap
- Chunking Strategy
- Embedding Model
- Retrieval Method
- Retrieval top K
- LLM
- Reranker (mit/ohne) (Modell)
- Prompt-Template

> **Hinweis:** Runs mit N=1 oder N=50 sind Testläufe. Runs mit N=100 oder N=150 sind vollständige Benchmarks.

## Matrix 

| #        | LLM          | Chunking  | Size | Overlap | Retrieval  | Template | Retrieval_Top_K | Reranker | Embedding        | Dataset | Faith | Rel   | Corr  | Prec  | Rec   | N   | Status      |
| -------- | ------------ | --------- | ---- | ------- | ---------- | -------- | --------------- | -------- | ---------------- | ------- | ----- | ----- | ----- | ----- | ----- | --- | ----------- |
| 1        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Concise  |                 | no       | nomic-embed-text | wikiqa  | 0.200 | 0.507 | 0.355 | 0.421 | 0.380 | 50  | Test (N=50) |
| 2        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text | wikiqa  | 0.813 | -     | -     | -     | -     | 100 | Offen       |
| 3        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text | wikiqa  | -     | -     | -     | -     | -     | -   | Offen       |
| 4        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text | wikiqa  | -     | -     | -     | -     | -     | -   | Offen       |
| 5        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text | wikiqa  | 0.514 | 0.332 | 0.367 | 0.399 | 0.353 | 150 | Getestet    |
| 6        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text | wikiqa  | -     | -     | -     | -     | -     | -   | Offen       |
| 7        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 8        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 9        | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 10       | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 11       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | 0.821 | 0.490 | 0.323 | 0.463 | 0.310 | 100 | Getestet    |
| 12       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 13       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 14       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 15       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 16       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 17       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 18       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 19       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 20       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 21       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 22       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 23       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 24       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 25       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 26       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 27       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | 0.877 | 0.560 | 0.386 | 0.625 | 0.470 | 100 | Getestet    |
| 28       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 29       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 30       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 31       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 32       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 33       | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Concise  | 12              | No       | nomic-embed-text | squad   | 0.91  | -     | -     | -     | -     | 100 | getestet    |
| 34 rerun | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Detailed | 12              | No       | nomic-embed-text | squad   | 0.94  |       |       |       |       | 100 | getestet    |
| 35       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Concise  | 12              | No       | nomic-embed-text | squad   | 0.885 | -     | -     | -     | -     | -   | Offen       |
| 36       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Detailed | 12              | No       | nomic-embed-text | squad   | 0.92  | -     | -     | -     | -     | -   | Offen       |
| 37       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Concise  | 12              | No       | nomic-embed-text | squad   | 0.907 | -     | -     | -     | -     | 100 | Getestet    |
| 38       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 39       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Concise  |                 | No       | nomic-embed-text | squad   | 0.885 | -     | -     | -     | -     | 100 | Getestet    |
| 40       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Detailed |                 | No       | nomic-embed-text | squad   | 0.896 | -     | -     | -     | -     | 100 | Getestet    |
| 41       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | 0.888 | 0.525 | 0.374 | 0.441 | 0.430 | 100 | Getestet    |
| 42       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | 0.931 | 0.564 | 0.338 | 0.402 | 0.360 | 50  | Test (N=50) |
| 43       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 44       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 45       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Concise  | 12              | No       | nomic-embed-text | squad   | 0.910 | -     | -     | -     | -     | 100 | Getestet    |
| 46       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 47       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 48       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 49       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 50       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 51       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 52       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 53       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 54       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 55       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 56       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 57       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | 0.921 | 0.559 | 0.420 | 0.599 | 0.590 | 100 | Getestet    |
| 58       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 59       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | 0.887 | 0.521 | 0.386 | 0.607 | 0.470 | 100 | Getestet    |
| 60       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 61       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 62       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 63       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 64       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 65       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | 0.265 | 0.501 | 0.441 | 0.429 | 0.400 | 50  | Test (N=50) |
| 66       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | 0.568 | 0.446 | 0.277 | 0.432 | 0.400 | 50  | Test (N=50) |
| 67       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 68       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 69       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 70       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 71       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 72       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 73       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 74       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 75       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 76       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 77       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 78       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 79       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 80       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 81       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 82       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 83       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 84       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 85       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 86       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 87       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 88       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 89       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 90       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 91       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 92       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 93       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 94       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 95       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 96       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 97       | Qwen3.5-35B  | Recursive | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | 0.160 | 0.520 | 0.405 | 0.432 | 0.360 | 50  | Test (N=50) |
| 98       | Qwen3.5-35B  | Recursive | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | 0.810 | 0.513 | 0.229 | 0.407 | 0.360 | 50  | Test (N=50) |
| 99       | Qwen3.5-35B  | Recursive | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 100      | Qwen3.5-35B  | Recursive | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 101      | Qwen3.5-35B  | Recursive | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 102      | Qwen3.5-35B  | Recursive | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 103      | Qwen3.5-35B  | Recursive | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 104      | Qwen3.5-35B  | Recursive | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 105      | Qwen3.5-35B  | Recursive | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 106      | Qwen3.5-35B  | Recursive | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 107      | Qwen3.5-35B  | Recursive | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 108      | Qwen3.5-35B  | Recursive | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 109      | Qwen3.5-35B  | Recursive | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 110      | Qwen3.5-35B  | Recursive | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 111      | Qwen3.5-35B  | Recursive | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 112      | Qwen3.5-35B  | Recursive | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 113      | Qwen3.5-35B  | Semantic  | 1000 | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 114      | Qwen3.5-35B  | Semantic  | 1000 | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 115      | Qwen3.5-35B  | Semantic  | 1000 | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 116      | Qwen3.5-35B  | Semantic  | 1000 | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 117      | Qwen3.5-35B  | Semantic  | 1000 | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 118      | Qwen3.5-35B  | Semantic  | 1000 | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 119      | Qwen3.5-35B  | Semantic  | 1000 | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 120      | Qwen3.5-35B  | Semantic  | 1000 | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 121      | Qwen3.5-35B  | Semantic  | 500  | 100     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 122      | Qwen3.5-35B  | Semantic  | 500  | 100     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 123      | Qwen3.5-35B  | Semantic  | 500  | 100     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 124      | Qwen3.5-35B  | Semantic  | 500  | 100     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 125      | Qwen3.5-35B  | Semantic  | 500  | 200     | Similarity | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 126      | Qwen3.5-35B  | Semantic  | 500  | 200     | Similarity | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 127      | Qwen3.5-35B  | Semantic  | 500  | 200     | MMR        | Concise  |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 128      | Qwen3.5-35B  | Semantic  | 500  | 200     | MMR        | Detailed |                 |          | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |

## Mathematische Metriken
Für alle Metriken wird der mean Wert betrachtet

| ID  | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu  | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
| --- | ----- | ------ | -------- | ----- | ------ | -------- | ----- | ------ | -------- | ----------------- | ------------- | ----------------- | ------- | ----- | ------ | -------------------- | ----------------- | ------------- |
| 1   | 1.0   | 1.0    | 0.125    | 1.0   | 1.0    | 0.375    | 1.0   | 1.0    | 0.625    | 0.623             |               |                   | 0.183   | 0.052 | 0.326  | 0.803                | 0.815             | 0.809         |
| 2   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 3   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 4   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 5   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 6   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 8   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 9   |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 10  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 11  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 12  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 13  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 14  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 15  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 16  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 17  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 18  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 19  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 20  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 21  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 22  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 23  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 24  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 25  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 26  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 27  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 28  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 29  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 30  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 31  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 32  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 33  | 0.67  | 0.67   | 0.214    | 0.82  | 0.664  | 0.394    | 0.85  | 0.672  | 0.523    | 0.574             |               |                   | 0.352   | 0.051 | 0.518  | 0.872                | 0.887             | 0.879         |
| 34  | 0.67  | 0.67   | 0.205    | 0.82  | 0.658  | 0.386    | 0.85  | 0.672  | 0.521    | 0.574             |               |                   | 0.273   | 0.062 | 0.523  | 0.882                | 0.871             | 0.873         |
| 35  | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             |               |                   | 0.328   | 0.051 | 0.473  | 0.833                | 0.847             | 0.84          |
| 36  | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             |               |                   | 0.121   | 0.02  | 0.283  | 0.836                | 0.863             | 0.85          |
| 37  | 0.7   | 0.7    | 0.222    | 0.82  | 0.68   | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             |               |                   | 0.343   | 0.05  | 0.516  | 0.881                | 0.896             | 0.889         |
| 38  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 39  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.258             | 0.319   | 0.051 | 0.468  | 0.823                | 0.837             | 0.830         |
| 40  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.201             | 0.118   | 0.019 | 0.280  | 0.845                | 0.873             | 0.859         |
| 41  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 42  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 43  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 44  |       |        |          |       |        |          |       |        |          |                   |               |                   |         |       |        |                      |                   |               |
| 45  | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.725  | 0.553    | 0.605             |               |                   | 0.32    | 0.06  | 0.484  | 0.852                | 0.866             | 0.859         |



