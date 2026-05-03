# Benchmarking Matrix

## Feste Parameter


| Parameter | Wert                             |
| --------- | -------------------------------- |
| Embedding | nomic-embed-text:latest (Ollama) |
| Reranker  | Deaktiviert                      |
| HyDE      | Deaktiviert                      |
| Dataset   | squad                            |
| Eval_LLM  | Qwen3.5-397B-A17B_No_Thinking    |


### Zu testende Parameter

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


| #        | LLM          | Chunking  | Size | Overlap | Retrieval  | Template | Retrieval_Top_K | Reranker            | Embedding        | Dataset | Faith | Rel   | Corr  | Prec  | Rec   | N   | Status      |
| -------- | ------------ | --------- | ---- | ------- | ---------- | -------- | --------------- | ------------------- | ---------------- | ------- | ----- | ----- | ----- | ----- | ----- | --- | ----------- |
| 1        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.898 | -     | -     | -     | -     | 100 | Getestet    |
| 2        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.876 | -     | -     | -     | -     | 100 | Getestet    |
| 3        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.843 | -     | -     | -     | -     | 100 | Getestet    |
| 4        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.865 | -     | -     | -     | -     | 100 | Getestet    |
| 5        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.889 | -     | -     | -     | -     | 100 | Getestet    |
| 6        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.921 | -     | -     | -     | -     | 100 | Getestet    |
| 7        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.840 | -     | -     | -     | -     | 100 | Getestet    |
| 8        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.863 | -     | -     | -     | -     | 100 | Getestet    |
| 9        | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.872 | -     | -     | -     | -     | 100 | Getestet    |
| 10       | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.904 | -     | -     | -     | -     | 100 | Getestet    |
| 11       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.842 | -     | -     | -     | -     | 100 | Getestet    |
| 12       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.848 | -     | -     | -     | -     | 100 | Getestet    |
| 13       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.842 | -     | -     | -     | -     | 100 | Getestet    |
| 14       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.875 | -     | -     | -     | -     | 100 | Getestet    |
| 15       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.867 | -     | -     | -     | -     | 100 | Getestet    |
| 16       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.863 | -     | -     | -     | -     | 100 | Getestet    |
| 17       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.858 | -     | -     | -     | -     | 100 | Getestet    |
| 18       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.867 | -     | -     | -     | -     | 100 | Getestet    |
| 19       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.799 | -     | -     | -     | -     | 100 | Getestet    |
| 20       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.868 | -     | -     | -     | -     | 100 | Getestet    |
| 21       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.858 | -     | -     | -     | -     | 100 | Getestet    |
| 22       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.873 | -     | -     | -     | -     | 100 | Getestet    |
| 23       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.829 | -     | -     | -     | -     | 100 | Getestet    |
| 24       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.866 | -     | -     | -     | -     | 100 | Getestet    |
| 25       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.855 | -     | -     | -     | -     | 100 | Getestet    |
| 26       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.871 | -     | -     | -     | -     | 100 | Getestet    |
| 27       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.819 | -     | -     | -     | -     | 100 | Getestet    |
| 28       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.865 | -     | -     | -     | -     | 100 | Getestet    |
| 29       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.853 | -     | -     | -     | -     | 100 | Getestet    |
| 30       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.870 | -     | -     | -     | -     | 100 | Getestet    |
| 31       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.809 | -     | -     | -     | -     | 100 | Getestet    |
| 32       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.846 | -     | -     | -     | -     | 100 | Getestet    |
| 33       | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Concise  | 12              | Yes (MiniLM-L-6-v2) | nomic-embed-text | squad   | 0.840 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 34 rerun | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Detailed | 12              | Yes (MiniLM-L-6-v2) | nomic-embed-text | squad   | 0.940 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 35       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.885 | -     | -     | -     | -     | -   | Offen       |
| 36       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.92  | -     | -     | -     | -     | -   | Offen       |
| 37       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.907 | -     | -     | -     | -     | 100 | Getestet    |
| 38       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.944 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 39       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.885 | -     | -     | -     | -     | 100 | Getestet    |
| 40       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.896 | -     | -     | -     | -     | 100 | Getestet    |
| 41       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.825 | -     | -     | -     | 0.930 | 100 | Getestet    |
| 42       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.955 | -     | -     | -     | 0.930 | 100 | Getestet    |
| 43       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.767 | -     | -     | 0.90  | 0.708 | 100 | Getestet    |
| 44       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.936 | -     | -     | 0.90  | 0.654 | 100 | Getestet    |
| 45       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.910 | -     | -     | -     | -     | 100 | Getestet    |
| 46       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Detailed | 12              | Yes (MiniLM-L-6-v2) | nomic-embed-text | squad   | 0.945 | -     | -     | -     | 0.940 | 100 | Getestet    |
| 47       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Concise  |                 | Yes (MiniLM-L-6-v2) | nomic-embed-text | squad   | 0.812 | -     | -     | 0.89  | 0.738 | 100 | Getestet    |
| 48       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Detailed |                 | Yes (MiniLM-L-6-v2) | nomic-embed-text | squad   | 0.949 | -     | -     | 0.89  | 0.650 | 100 | Getestet    |
| 49       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.852 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 50       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.941 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 51       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 52       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 53       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 54       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.941 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 55       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 56       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 57       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 58       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.948 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 59       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | 0.887 | 0.521 | 0.386 | 0.607 | 0.470 | 100 | Getestet    |
| 60       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 61       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 62       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | -     | -     | -     | 0.960 | 100 | Getestet    |
| 63       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 64       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 65       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 66       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 67       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 68       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 69       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 70       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 71       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 72       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 73       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 74       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 75       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 76       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 77       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 78       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 79       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 80       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 81       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 82       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 83       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 84       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 85       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 86       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 87       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 88       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 89       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 90       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 91       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 92       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 93       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 94       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 95       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 96       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 97       | Qwen3.5-35B  | Recursive | 1000 | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | 0.160 | 0.520 | 0.405 | 0.432 | 0.360 | 50  | Test (N=50) |
| 98       | Qwen3.5-35B  | Recursive | 1000 | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | 0.810 | 0.513 | 0.229 | 0.407 | 0.360 | 50  | Test (N=50) |
| 99       | Qwen3.5-35B  | Recursive | 1000 | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 100      | Qwen3.5-35B  | Recursive | 1000 | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 101      | Qwen3.5-35B  | Recursive | 1000 | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 102      | Qwen3.5-35B  | Recursive | 1000 | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 103      | Qwen3.5-35B  | Recursive | 1000 | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 104      | Qwen3.5-35B  | Recursive | 1000 | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 105      | Qwen3.5-35B  | Recursive | 500  | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 106      | Qwen3.5-35B  | Recursive | 500  | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 107      | Qwen3.5-35B  | Recursive | 500  | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 108      | Qwen3.5-35B  | Recursive | 500  | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 109      | Qwen3.5-35B  | Recursive | 500  | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 110      | Qwen3.5-35B  | Recursive | 500  | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 111      | Qwen3.5-35B  | Recursive | 500  | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 112      | Qwen3.5-35B  | Recursive | 500  | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 113      | Qwen3.5-35B  | Semantic  | 1000 | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 114      | Qwen3.5-35B  | Semantic  | 1000 | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 115      | Qwen3.5-35B  | Semantic  | 1000 | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 116      | Qwen3.5-35B  | Semantic  | 1000 | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 117      | Qwen3.5-35B  | Semantic  | 1000 | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 118      | Qwen3.5-35B  | Semantic  | 1000 | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 119      | Qwen3.5-35B  | Semantic  | 1000 | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 120      | Qwen3.5-35B  | Semantic  | 1000 | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 121      | Qwen3.5-35B  | Semantic  | 500  | 100     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 122      | Qwen3.5-35B  | Semantic  | 500  | 100     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 123      | Qwen3.5-35B  | Semantic  | 500  | 100     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 124      | Qwen3.5-35B  | Semantic  | 500  | 100     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 125      | Qwen3.5-35B  | Semantic  | 500  | 200     | Similarity | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 126      | Qwen3.5-35B  | Semantic  | 500  | 200     | Similarity | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 127      | Qwen3.5-35B  | Semantic  | 500  | 200     | MMR        | Concise  |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |
| 128      | Qwen3.5-35B  | Semantic  | 500  | 200     | MMR        | Detailed |                 |                     | nomic-embed-text |         | -     | -     | -     | -     | -     | -   | Offen       |


## Mathematische Metriken

Für alle Metriken wird der mean Wert betrachtet


| ID  | hit@1 | ndcg@1 | recall@1 | hit@3 | ndcg@3 | recall@3 | hit@5 | ndcg@5 | recall@5 | context_relevance | vec_dist_q_gt | vec_dist_q_answer | rouge_l | bleu  | meteor | bert_score_precision | bert_score_recall | bert_score_f1 |
| --- | ----- | ------ | -------- | ----- | ------ | -------- | ----- | ------ | -------- | ----------------- | ------------- | ----------------- | ------- | ----- | ------ | -------------------- | ----------------- | ------------- |
| 1   | 0.67  | 0.67   | 0.205    | 0.82  | 0.658  | 0.386    | 0.85  | 0.672  | 0.519    | 0.574             | 0.474         | 0.304             | 0.517   | 0.071 | 0.568  | 0.926                | 0.938             | 0.932         |
| 2   | 0.67  | 0.67   | 0.205    | 0.82  | 0.658  | 0.386    | 0.85  | 0.672  | 0.519    | 0.574             | 0.474         | 0.155             | 0.214   | 0.028 | 0.406  | 0.928                | 0.952             | 0.940         |
| 3   | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             | 0.474         | 0.374             | 0.605   | 0.084 | 0.585  | 0.852                | 0.861             | 0.856         |
| 4   | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             | 0.474         | 0.155             | 0.213   | 0.031 | 0.396  | 0.900                | 0.923             | 0.911         |
| 5   | 0.7   | 0.7    | 0.222    | 0.82  | 0.68   | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.314             | 0.546   | 0.071 | 0.586  | 0.917                | 0.928             | 0.923         |
| 6   | 0.7   | 0.7    | 0.222    | 0.82  | 0.68   | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.159             | 0.214   | 0.026 | 0.409  | 0.929                | 0.952             | 0.940         |
| 7   | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.367             | 0.592   | 0.069 | 0.583  | 0.852                | 0.860             | 0.856         |
| 8   | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.156             | 0.212   | 0.030 | 0.397  | 0.881                | 0.903             | 0.892         |
| 9   | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.9   | 0.728  | 0.553    | 0.600             | 0.474         | 0.344             | 0.595   | 0.096 | 0.602  | 0.891                | 0.899             | 0.895         |
| 10  | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.9   | 0.728  | 0.553    | 0.600             | 0.474         | 0.153             | 0.211   | 0.030 | 0.402  | 0.890                | 0.913             | 0.902         |
| 11  | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.360             | 0.607   | 0.067 | 0.579  | 0.892                | 0.900             | 0.896         |
| 12  | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.150             | 0.215   | 0.031 | 0.395  | 0.882                | 0.903             | 0.892         |
| 13  | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.725  | 0.553    | 0.605             | 0.474         | 0.329             | 0.570   | 0.079 | 0.590  | 0.899                | 0.909             | 0.904         |
| 14  | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.725  | 0.553    | 0.605             | 0.474         | 0.155             | 0.216   | 0.030 | 0.418  | 0.910                | 0.932             | 0.921         |
| 15  | 0.72  | 0.72   | 0.352    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.369             | 0.592   | 0.076 | 0.561  | 0.872                | 0.880             | 0.876         |
| 16  | 0.72  | 0.72   | 0.352    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.152             | 0.222   | 0.033 | 0.399  | 0.863                | 0.883             | 0.873         |
| 17  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.324             | 0.573   | 0.064 | 0.602  | 0.909                | 0.919             | 0.914         |
| 18  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.150             | 0.226   | 0.031 | 0.426  | 0.930                | 0.952             | 0.941         |
| 19  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.362             | 0.571   | 0.067 | 0.565  | 0.862                | 0.870             | 0.866         |
| 20  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.152             | 0.219   | 0.028 | 0.410  | 0.910                | 0.933             | 0.921         |
| 21  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.324             | 0.573   | 0.064 | 0.602  | 0.909                | 0.919             | 0.914         |
| 22  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.150             | 0.226   | 0.031 | 0.426  | 0.930                | 0.952             | 0.941         |
| 23  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.362             | 0.571   | 0.067 | 0.565  | 0.862                | 0.870             | 0.866         |
| 24  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.152             | 0.219   | 0.028 | 0.410  | 0.910                | 0.933             | 0.921         |
| 25  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.324             | 0.573   | 0.064 | 0.602  | 0.909                | 0.919             | 0.914         |
| 26  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.150             | 0.226   | 0.031 | 0.426  | 0.930                | 0.952             | 0.941         |
| 27  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.362             | 0.571   | 0.067 | 0.565  | 0.862                | 0.870             | 0.866         |
| 28  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.152             | 0.219   | 0.028 | 0.410  | 0.910                | 0.933             | 0.921         |
| 29  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.324             | 0.573   | 0.064 | 0.602  | 0.909                | 0.919             | 0.914         |
| 30  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.4      | 0.9   | 0.719  | 0.542    | 0.596             | 0.474         | 0.150             | 0.226   | 0.031 | 0.426  | 0.930                | 0.952             | 0.941         |
| 31  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.362             | 0.571   | 0.067 | 0.565  | 0.862                | 0.870             | 0.866         |
| 32  | 0.74  | 0.74   | 0.41     | 0.86  | 0.809  | 0.86     | 0.86  | 0.809  | 0.86     | 0.670             | 0.474         | 0.152             | 0.219   | 0.028 | 0.410  | 0.910                | 0.933             | 0.921         |
| 33  | 0.77  | 0.77   | 0.472    | 0.86  | 0.822  | 0.86     | 0.86  | 0.822  | 0.86     | 0.631             | 0.474         | 0.299             | 0.455   | 0.084 | 0.559  | 0.858                | 0.869             | 0.863         |
| 34  | 0.77  | 0.77   | 0.472    | 0.86  | 0.822  | 0.86     | 0.86  | 0.822  | 0.86     | 0.631             | 0.474         | 0.162             | 0.244   | 0.040 | 0.437  | 0.876                | 0.895             | 0.885         |
| 35  | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             |               |                   | 0.328   | 0.051 | 0.473  | 0.833                | 0.847             | 0.84          |
| 36  | 0.67  | 0.67   | 0.372    | 0.82  | 0.763  | 0.82     | 0.82  | 0.763  | 0.82     | 0.645             |               |                   | 0.121   | 0.02  | 0.283  | 0.836                | 0.863             | 0.85          |
| 37  | 0.7   | 0.7    | 0.222    | 0.82  | 0.68   | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             |               |                   | 0.343   | 0.05  | 0.516  | 0.881                | 0.896             | 0.889         |
| 38  | 0.7   | 0.7    | 0.222    | 0.82  | 0.680  | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.165             | 0.240   | 0.040 | 0.432  | 0.876                | 0.895             | 0.885         |
| 39  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.258             | 0.319   | 0.051 | 0.468  | 0.823                | 0.837             | 0.830         |
| 40  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.201             | 0.118   | 0.019 | 0.280  | 0.845                | 0.873             | 0.859         |
| 41  | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.9   | 0.728  | 0.553    | 0.600             | 0.474         | 0.298             | 0.442   | 0.084 | 0.536  | 0.838                | 0.849             | 0.843         |
| 42  | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.9   | 0.728  | 0.553    | 0.600             | 0.474         | 0.166             | 0.235   | 0.039 | 0.421  | 0.856                | 0.875             | 0.865         |
| 43  | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.312             | 0.412   | 0.077 | 0.509  | 0.789                | 0.800             | 0.794         |
| 44  | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.164             | 0.232   | 0.037 | 0.413  | 0.837                | 0.855             | 0.846         |
| 45  | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.725  | 0.553    | 0.605             |               |                   | 0.32    | 0.06  | 0.484  | 0.852                | 0.866             | 0.859         |
| 46  | 0.78  | 0.78   | 0.412    | 0.86  | 0.825  | 0.86     | 0.86  | 0.825  | 0.86     | 0.666             | 0.474         | 0.162             | 0.243   | 0.038 | 0.431  | 0.876                | 0.895             | 0.885         |
| 47  | 0.77  | 0.77   | 0.397    | 0.85  | 0.814  | 0.85     | 0.85  | 0.814  | 0.85     | 0.678             | 0.474         | 0.318             | 0.466   | 0.090 | 0.552  | 0.829                | 0.839             | 0.834         |
| 48  | 0.77  | 0.77   | 0.397    | 0.85  | 0.814  | 0.85     | 0.85  | 0.814  | 0.85     | 0.678             | 0.474         | 0.163             | 0.239   | 0.041 | 0.424  | 0.847                | 0.865             | 0.856         |
| 49  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.314             | 0.485   | 0.092 | 0.574  | 0.859                | 0.869             | 0.864         |
| 50  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.162             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 53  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.305             | 0.497   | 0.092 | 0.583  | 0.878                | 0.889             | 0.884         |
| 54  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.160             | 0.235   | 0.037 | 0.427  | 0.866                | 0.885             | 0.875         |
| 57  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.309             | 0.497   | 0.092 | 0.584  | 0.878                | 0.889             | 0.884         |
| 58  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.162             | 0.237   | 0.038 | 0.433  | 0.875                | 0.895             | 0.885         |
| 61  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.309             | 0.497   | 0.092 | 0.584  | 0.878                | 0.889             | 0.884         |
| 62  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.161             | 0.235   | 0.037 | 0.427  | 0.866                | 0.885             | 0.875         |


### Empfohlene Scores:

Retrieval:   ndcg@5
Context:     context_relevance
Generation:  bert_score_f1 + meteor
