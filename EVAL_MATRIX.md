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


| #        | LLM          | Chunking  | Size | Overlap | Retrieval  | Template | Retrieval_Top_K | Reranker            | Embedding        | Dataset | Faith | ndcg@5 | ctx_rel | meteor | bert_f1   | Semsim | N   | Status   |
| -------- | ------------ | --------- | ---- | ------- | ---------- | -------- | --------------- | ------------------- | ---------------- | ------- | ----- | ------ | ------- | ------ | --------- | ------ | --- | -------- |
| 1        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.898 | 0.672 | 0.574 | 0.568 | 0.932 | - | 100 | Getestet |
| 2        | Qwen3-32B    | Recursive | 1000 | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.876 | 0.672 | 0.574 | 0.406 | 0.940 | - | 100 | Getestet |
| 3        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.843 | 0.763 | 0.645 | 0.585 | 0.856 | - | 100 | Getestet |
| 4        | Qwen3-32B    | Recursive | 1000 | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.865 | 0.763 | 0.645 | 0.396 | 0.911 | - | 100 | Getestet |
| 5        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.889 | 0.678 | 0.574 | 0.586 | 0.923 | - | 100 | Getestet |
| 6        | Qwen3-32B    | Recursive | 1000 | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.921 | 0.678 | 0.574 | 0.409 | 0.940 | - | 100 | Getestet |
| 7        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.840 | 0.772 | 0.645 | 0.583 | 0.856 | - | 100 | Getestet |
| 8        | Qwen3-32B    | Recursive | 1000 | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.863 | 0.772 | 0.645 | 0.397 | 0.892 | - | 100 | Getestet |
| 9        | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.872 | 0.728 | 0.600 | 0.602 | 0.895 | - | 100 | Getestet |
| 10       | Qwen3-32B    | Recursive | 500  | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.904 | 0.728 | 0.600 | 0.402 | 0.902 | - | 100 | Getestet |
| 11       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.842 | 0.805 | 0.672 | 0.579 | 0.896 | - | 100 | Getestet |
| 12       | Qwen3-32B    | Recursive | 500  | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.848 | 0.805 | 0.672 | 0.395 | 0.892 | - | 100 | Getestet |
| 13       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.842 | 0.725 | 0.605 | 0.590 | 0.904 | - | 100 | Getestet |
| 14       | Qwen3-32B    | Recursive | 500  | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.875 | 0.725 | 0.605 | 0.418 | 0.921 | - | 100 | Getestet |
| 15       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.867 | 0.793 | 0.678 | 0.561 | 0.876 | - | 100 | Getestet |
| 16       | Qwen3-32B    | Recursive | 500  | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.863 | 0.793 | 0.678 | 0.399 | 0.873 | - | 100 | Getestet |
| 17       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.858 | 0.719 | 0.596 | 0.602 | 0.914 | - | 100 | Getestet |
| 18       | Qwen3-32B    | Semantic  | 1000 | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.867 | 0.719 | 0.596 | 0.426 | 0.941 | - | 100 | Getestet |
| 19       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.799 | 0.809 | 0.670 | 0.565 | 0.866 | - | 100 | Getestet |
| 20       | Qwen3-32B    | Semantic  | 1000 | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.868 | 0.809 | 0.670 | 0.410 | 0.921 | - | 100 | Getestet |
| 21       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.858 | 0.719 | 0.596 | 0.602 | 0.914 | - | 100 | Getestet |
| 22       | Qwen3-32B    | Semantic  | 1000 | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.873 | 0.719 | 0.596 | 0.426 | 0.941 | - | 100 | Getestet |
| 23       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.829 | 0.809 | 0.670 | 0.565 | 0.866 | - | 100 | Getestet |
| 24       | Qwen3-32B    | Semantic  | 1000 | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.866 | 0.809 | 0.670 | 0.410 | 0.921 | - | 100 | Getestet |
| 25       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.855 | 0.719 | 0.596 | 0.602 | 0.914 | - | 100 | Getestet |
| 26       | Qwen3-32B    | Semantic  | 500  | 100     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.871 | 0.719 | 0.596 | 0.426 | 0.941 | - | 100 | Getestet |
| 27       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.819 | 0.809 | 0.670 | 0.565 | 0.866 | - | 100 | Getestet |
| 28       | Qwen3-32B    | Semantic  | 500  | 100     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.865 | 0.809 | 0.670 | 0.410 | 0.921 | - | 100 | Getestet |
| 29       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.853 | 0.719 | 0.596 | 0.602 | 0.914 | - | 100 | Getestet |
| 30       | Qwen3-32B    | Semantic  | 500  | 200     | Similarity | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.870 | 0.719 | 0.596 | 0.426 | 0.941 | - | 100 | Getestet |
| 31       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Concise  | 12              | no                  | nomic-embed-text | squad   | 0.809 | 0.809 | 0.670 | 0.565 | 0.866 | - | 100 | Getestet |
| 32       | Qwen3-32B    | Semantic  | 500  | 200     | MMR        | Detailed | 12              | no                  | nomic-embed-text | squad   | 0.846 | 0.809 | 0.670 | 0.410 | 0.921 | - | 100 | Getestet |
| 33       | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.840 | 0.822 | 0.631 | 0.559 | 0.863 | - | 100 | Getestet |
| 34 rerun | Qwen3.5-397B | Recursive | 1000 | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.940 | 0.822 | 0.631 | 0.437 | 0.885 | - | 100 | Getestet |
| 35       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.885 | 0.763 | 0.645 | 0.473 | 0.84 | - |-|Offen|
| 36       | Qwen3.5-397B | Recursive | 1000 | 100     | MMR        | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.92  | 0.763 | 0.645 | 0.283 | 0.85 | - |-|Offen|
| 37       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.907 | 0.678 | 0.574 | 0.516 | 0.889 | - | 100 | Getestet |
| 38       | Qwen3.5-397B | Recursive | 1000 | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.944 | 0.678 | 0.574 | 0.432 | 0.885 | - | 100 | Getestet |
| 39       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.885 | 0.772 | 0.645 | 0.468 | 0.830 | - | 100 | Getestet |
| 40       | Qwen3.5-397B | Recursive | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.896 | 0.772 | 0.645 | 0.280 | 0.859 | - | 100 | Getestet |
| 41       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.825 | 0.728 | 0.600 | 0.536 | 0.843 | - | 100 | Getestet |
| 42       | Qwen3.5-397B | Recursive | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.955 | 0.728 | 0.600 | 0.421 | 0.865 | - | 100 | Getestet |
| 43       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.767 | 0.805 | 0.672 | 0.509 | 0.794 | - | 100 | Getestet |
| 44       | Qwen3.5-397B | Recursive | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.936 | 0.805 | 0.672 | 0.413 | 0.846 | - | 100 | Getestet |
| 45       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.910 | 0.725 | 0.605 | 0.484 | 0.859 | - | 100 | Getestet |
| 46       | Qwen3.5-397B | Recursive | 500  | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.825 | 0.666 | 0.431 | 0.885 | - | 100 | Getestet |
| 47       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.812 | 0.814 | 0.678 | 0.552 | 0.834 | - | 100 | Getestet |
| 48       | Qwen3.5-397B | Recursive | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.949 | 0.814 | 0.678 | 0.424 | 0.856 | - | 100 | Getestet |
| 49       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.852 | 0.719 | 0.596 | 0.574 | 0.864 | - | 100 | Getestet |
| 50       | Qwen3.5-397B | Semantic  | 1000 | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.941 | 0.719 | 0.596 | 0.424 | 0.866 | - | 100 | Getestet |
| 51       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.809 | 0.670 | 0.516 | 0.785 | - | 100 | Getestet |
| 52       | Qwen3.5-397B | Semantic  | 1000 | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.944 | 0.809 | 0.670 | 0.415 | 0.837 | - | 100 | Getestet |
| 53       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | 0.719 | 0.596 | 0.583 | 0.884 | - | 100 | Getestet |
| 54       | Qwen3.5-397B | Semantic  | 1000 | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.941 | 0.719 | 0.596 | 0.427 | 0.875 | - | 100 | Getestet |
| 55       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.782 | 0.809 | 0.670 | 0.517 | 0.785 | - | 100 | Getestet |
| 56       | Qwen3.5-397B | Semantic  | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.947 | 0.809 | 0.670 | 0.415 | 0.837 | - | 100 | Getestet |
| 57       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | 0.719 | 0.596 | 0.584 | 0.884 | - | 100 | Getestet |
| 58       | Qwen3.5-397B | Semantic  | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.948 | 0.719 | 0.596 | 0.433 | 0.885 | - | 100 | Getestet |
| 59       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.809 | 0.670 | 0.517 | 0.785 | - | 100 | Getestet |
| 60       | Qwen3.5-397B | Semantic  | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.944 | 0.809 | 0.670 | 0.415 | 0.837 | - | 100 | Getestet |
| 61       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.872 | 0.719 | 0.596 | 0.584 | 0.884 | - | 100 | Getestet |
| 62       | Qwen3.5-397B | Semantic  | 500  | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.719 | 0.596 | 0.427 | 0.875 | - | 100 | Getestet |
| 63       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.809 | 0.670 | 0.517 | 0.785 | - | 100 | Getestet |
| 64       | Qwen3.5-397B | Semantic  | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.944 | 0.809 | 0.670 | 0.415 | 0.837 | - | 100 | Getestet |
| 65       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.827 | 0.675 | 0.574 | 0.575 | 0.932 | - | 100 | Getestet |
| 66       | GPT-OSS-20B  | Recursive | 1000 | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.805 | 0.675 | 0.574 | 0.316 | 0.912 | - | 100 | Getestet |
| 67       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.763 | 0.767 | 0.645 | 0.545 | 0.874 | - | 100 | Getestet |
| 68       | GPT-OSS-20B  | Recursive | 1000 | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.774 | 0.767 | 0.645 | 0.347 | 0.932 | - | 100 | Getestet |
| 69       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.836 | 0.678 | 0.574 | 0.581 | 0.932 | - | 100 | Getestet |
| 70       | GPT-OSS-20B  | Recursive | 1000 | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.792 | 0.678 | 0.574 | 0.319 | 0.960 | - | 100 | Getestet |
| 71       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.773 | 0.772 | 0.645 | 0.563 | 0.884 | - | 100 | Getestet |
| 72       | GPT-OSS-20B  | Recursive | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.769 | 0.772 | 0.645 | 0.342 | 0.903 | - | 100 | Getestet |
| 73       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.809 | 0.724 | 0.600 | 0.575 | 0.893 | - | 100 | Getestet |
| 74       | GPT-OSS-20B  | Recursive | 500  | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.724 | 0.600 | 0.358 | 0.912 | - | 100 | Getestet |
| 75       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.802 | 0.802 | 0.672 | 0.571 | 0.875 | - | 100 | Getestet |
| 76       | GPT-OSS-20B  | Recursive | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.813 | 0.802 | 0.672 | 0.341 | 0.894 | - | 100 | Getestet |
| 77       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.826 | 0.724 | 0.605 | 0.583 | 0.922 | - | 100 | Getestet |
| 78       | GPT-OSS-20B  | Recursive | 500  | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.807 | 0.724 | 0.605 | 0.324 | 0.922 | - | 100 | Getestet |
| 79       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.772 | 0.793 | 0.678 | 0.564 | 0.875 | - | 100 | Getestet |
| 80       | GPT-OSS-20B  | Recursive | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.802 | 0.793 | 0.678 | 0.334 | 0.903 | - | 100 | Getestet |
| 81       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.821 | 0.713 | 0.596 | 0.597 | 0.914 | - | 100 | Getestet |
| 82       | GPT-OSS-20B  | Semantic  | 1000 | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.822 | 0.713 | 0.596 | 0.372 | 0.913 | - | 100 | Getestet |
| 83       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.837 | 0.804 | 0.671 | 0.580 | 0.885 | - | 100 | Getestet |
| 84       | GPT-OSS-20B  | Semantic  | 1000 | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.797 | 0.804 | 0.671 | 0.347 | 0.873 | - | 100 | Getestet |
| 85       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.821 | 0.713 | 0.596 | 0.597 | 0.914 | - | 100 | Getestet |
| 86       | GPT-OSS-20B  | Semantic  | 1000 | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.822 | 0.713 | 0.596 | 0.372 | 0.913 | - | 100 | Getestet |
| 87       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.83  | 0.804 | 0.671 | 0.571 | 0.894 | - | 100 | Getestet |
| 88       | GPT-OSS-20B  | Semantic  | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.813 | 0.804 | 0.671 | 0.363 | 0.893 | - | 100 | Getestet |
| 89       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.829 | 0.713 | 0.596 | 0.573 | 0.932 | - | 100 | Getestet |
| 90       | GPT-OSS-20B  | Semantic  | 500  | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.822 | 0.713 | 0.596 | 0.358 | 0.893 | - | 100 | Getestet |
| 91       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.813 | 0.804 | 0.671 | 0.554 | 0.884 | - | 100 | Getestet |
| 92       | GPT-OSS-20B  | Semantic  | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.801 | 0.804 | 0.671 | 0.374 | 0.894 | - | 100 | Getestet |
| 93       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.829 | 0.713 | 0.596 | 0.573 | 0.932 | - | 100 | Getestet |
| 94       | GPT-OSS-20B  | Semantic  | 500  | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.822 | 0.713 | 0.596 | 0.358 | 0.893 | - | 100 | Getestet |
| 95       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.813 | 0.804 | 0.671 | 0.554 | 0.884 | - | 100 | Getestet |
| 96       | GPT-OSS-20B  | Semantic  | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.803 | 0.804 | 0.671 | 0.374 | 0.894 | - | 100 | Getestet |
| 97       | qwen3.5-think  | Recursive | 1000 | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.87  | 0.675 | 0.574 | 0.552 | 0.873 | - | 100 | Getestet |
| 98       | qwen3.5-think  | Recursive | 1000 | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.95  | 0.675 | 0.574 | 0.433 | 0.885 | - | 100 | Getestet |
| 99       | qwen3.5-think  | Recursive | 1000 | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.785 | 0.767 | 0.645 | 0.513 | 0.804 | - | 100 | Getestet |
| 100      | qwen3.5-think  | Recursive | 1000 | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.939 | 0.767 | 0.645 | 0.411 | 0.837 | - | 100 | Getestet |
| 101      | qwen3.5-think  | Recursive | 1000 | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.865 | 0.678 | 0.574 | 0.558 | 0.873 | - | 100 | Getestet |
| 102      | qwen3.5-think  | Recursive | 1000 | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.942 | 0.678 | 0.574 | 0.435 | 0.885 | - | 100 | Getestet |
| 103      | qwen3.5-think  | Recursive | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.775 | 0.772 | 0.645 | 0.506 | 0.804 | - | 100 | Getestet |
| 104      | qwen3.5-think  | Recursive | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.939 | 0.772 | 0.645 | 0.411 | 0.837 | - | 100 | Getestet |
| 105      | qwen3.5-think  | Recursive | 500  | 100     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.837 | 0.728 | 0.600 | 0.545 | 0.853 | - | 100 | Getestet |
| 106      | qwen3.5-think  | Recursive | 500  | 100     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.955 | 0.728 | 0.600 | 0.421 | 0.865 | - | 100 | Getestet |
| 107      | qwen3.5-think  | Recursive | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.767 | 0.805 | 0.672 | 0.508 | 0.794 | - | 100 | Getestet |
| 108      | qwen3.5-think  | Recursive | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.939 | 0.805 | 0.672 | 0.412 | 0.846 | - | 100 | Getestet |
| 109      | qwen3.5-think  | Recursive | 500  | 200     | Similarity | Concise  |                 | No                  | nomic-embed-text | squad   | 0.797 | 0.726 | 0.605 | 0.553 | 0.844 | - | 100 | Getestet |
| 110      | qwen3.5-think  | Recursive | 500  | 200     | Similarity | Detailed |                 | No                  | nomic-embed-text | squad   | 0.914 | 0.726 | 0.605 | 0.429 | 0.866 | - | 100 | Getestet |
| 111      | qwen3.5-think  | Recursive | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.807 | 0.793 | 0.678 | 0.544 | 0.825 | - | 100 | Getestet |
| 112      | qwen3.5-think  | Recursive | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.944 | 0.793 | 0.678 | 0.415 | 0.846 | - | 100 | Getestet |
| 113      | qwen3.5-think  | Semantic  | 1000 | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | - | 100 | Getestet |
| 114      | qwen3.5-think  | Semantic  | 1000 | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | - | 100 | Getestet |
| 115      | qwen3.5-think  | Semantic  | 1000 | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.810 | 0.671 | 0.517 | 0.785 | - | 100 | Getestet |
| 116      | qwen3.5-think  | Semantic  | 1000 | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.940 | 0.810 | 0.671 | 0.414 | 0.837 | - | 100 | Getestet |
| 117      | qwen3.5-think  | Semantic  | 1000 | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | - | 100 | Getestet |
| 118      | qwen3.5-think  | Semantic  | 1000 | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | - | 100 | Getestet |
| 119      | qwen3.5-think  | Semantic  | 1000 | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.810 | 0.671 | 0.517 | 0.785 | - | 100 | Getestet |
| 120      | qwen3.5-think  | Semantic  | 1000 | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.940 | 0.810 | 0.671 | 0.414 | 0.837 | - | 100 | Getestet |
| 121      | qwen3.5-think  | Semantic  | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | - | 100 | Getestet |
| 122      | qwen3.5-think  | Semantic  | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | - | 100 | Getestet |
| 123      | qwen3.5-think  | Semantic  | 500  | 100     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.810 | 0.671 | 0.517 | 0.785 | - | 100 | Getestet |
| 124      | qwen3.5-think  | Semantic  | 500  | 100     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.940 | 0.810 | 0.671 | 0.414 | 0.837 | - | 100 | Getestet |
| 125      | qwen3.5-think  | Semantic  | 500  | 200     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | - | 100 | Getestet |
| 126      | qwen3.5-think  | Semantic  | 500  | 200     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | - | 100 | Getestet |
| 127      | qwen3.5-think  | Semantic  | 500  | 200     | MMR        | Concise  |                 | No                  | nomic-embed-text | squad   | 0.787 | 0.810 | 0.671 | 0.517 | 0.785 | - | 100 | Getestet |
| 128      | qwen3.5-think  | Semantic  | 500  | 200     | MMR        | Detailed |                 | No                  | nomic-embed-text | squad   | 0.940 | 0.810 | 0.671 | 0.414 | 0.837 | - | 100 | Getestet |
| 129      | gpt-4o-mini   | Semantic  | 1000 | 100     | -          | Detailed |                 | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | 0.659 | 100 | Getestet |
| 130      | gpt-4o-mini   | Semantic  | 1000 | 100     | -          | Concise  |                 | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | 0.737 | 100 | Getestet |
| 131      | gpt-4o-mini   | Semantic  | 1000 | 200     | -          | Detailed |                 | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | 0.659 | 100 | Getestet |
| 132      | gpt-4o-mini   | Semantic  | 1000 | 200     | -          | Concise  |                 | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | 0.737 | 100 | Getestet |
| 133      | gpt-4o-mini   | Semantic  | 500  | 100     | -          | Detailed |                 | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | 0.659 | 100 | Getestet |
| 134      | gpt-4o-mini   | Semantic  | 500  | 100     | -          | Concise  |                 | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | 0.737 | 100 | Getestet |
| 135      | gpt-4o-mini   | Semantic  | 500  | 200     | -          | Detailed |                 | No                  | nomic-embed-text | squad   | 0.945 | 0.718 | 0.596 | 0.424 | 0.866 | 0.659 | 100 | Getestet |
| 136      | gpt-4o-mini   | Semantic  | 500  | 200     | -          | Concise  |                 | No                  | nomic-embed-text | squad   | 0.847 | 0.718 | 0.596 | 0.573 | 0.864 | 0.737 | 100 | Getestet |
| 137      | deepseek-v4-pro | Recursive | 500  | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.812 | 0.728 | 0.600 | 0.658 | 0.909 | 0.833 | 100 | Getestet |
| 138      | deepseek-v4-pro | Recursive | 500  | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.938 | 0.728 | 0.600 | 0.448 | 0.914 | 0.670 | 100 | Getestet |
| 139      | deepseek-v4-pro | Recursive | 1000 | 100     | Similarity | Concise  | 12              | No                  | nomic-embed-text | squad   | 0.845 | 0.675 | 0.574 | 0.665 | 0.919 | -     | 100 | Getestet |
| 140      | deepseek-v4-pro | Recursive | 1000 | 100     | Similarity | Detailed | 12              | No                  | nomic-embed-text | squad   | 0.927 | 0.675 | 0.574 | 0.452 | 0.914 | -     | 100 | Getestet |



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
| 51  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.322             | 0.422   | 0.082 | 0.516  | 0.780                | 0.790             | 0.785         |
| 52  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.161             | 0.234   | 0.038 | 0.415  | 0.828                | 0.846             | 0.837         |
| 53  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.305             | 0.497   | 0.092 | 0.583  | 0.878                | 0.889             | 0.884         |
| 54  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.160             | 0.235   | 0.037 | 0.427  | 0.866                | 0.885             | 0.875         |
| 55  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 56  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.161             | 0.234   | 0.038 | 0.415  | 0.828                | 0.846             | 0.837         |
| 57  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.309             | 0.497   | 0.092 | 0.584  | 0.878                | 0.889             | 0.884         |
| 58  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.162             | 0.237   | 0.038 | 0.433  | 0.875                | 0.895             | 0.885         |
| 59  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 60  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.161             | 0.234   | 0.038 | 0.415  | 0.828                | 0.846             | 0.837         |
| 61  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.309             | 0.497   | 0.092 | 0.584  | 0.878                | 0.889             | 0.884         |
| 62  | 0.74  | 0.74   | 0.222    | 0.86  | 0.703  | 0.400    | 0.90  | 0.719  | 0.542    | 0.596             | 0.474         | 0.161             | 0.235   | 0.037 | 0.427  | 0.866                | 0.885             | 0.875         |
| 63  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 64  | 0.74  | 0.74   | 0.410    | 0.86  | 0.809  | 0.860    | 0.86  | 0.809  | 0.860    | 0.670             | 0.474         | 0.161             | 0.234   | 0.038 | 0.415  | 0.828                | 0.846             | 0.837         |
| 65  | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.325             | 0.539   | 0.044 | 0.575  | 0.927                | 0.937             | 0.932         |
| 66  | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.225             | 0.215   | 0.007 | 0.316  | 0.901                | 0.922             | 0.912         |
| 67  | 0.68  | 0.68   | 0.382    | 0.82  | 0.767  | 0.82     | 0.82  | 0.767  | 0.82     | 0.645             | 0.474         | 0.357             | 0.545   | 0.040 | 0.545  | 0.870                | 0.878             | 0.874         |
| 68  | 0.68  | 0.68   | 0.382    | 0.82  | 0.767  | 0.82     | 0.82  | 0.767  | 0.82     | 0.645             | 0.474         | 0.229             | 0.247   | 0.017 | 0.347  | 0.922                | 0.942             | 0.932         |
| 69  | 0.70  | 0.70   | 0.222    | 0.82  | 0.680  | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.326             | 0.553   | 0.046 | 0.581  | 0.927                | 0.937             | 0.932         |
| 70  | 0.70  | 0.70   | 0.222    | 0.82  | 0.680  | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.220             | 0.229   | 0.010 | 0.319  | 0.949                | 0.971             | 0.960         |
| 71  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.372             | 0.571   | 0.043 | 0.563  | 0.881                | 0.888             | 0.884         |
| 72  | 0.7   | 0.7    | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.226             | 0.240   | 0.011 | 0.342  | 0.893                | 0.913             | 0.903         |
| 73  | 0.73  | 0.73   | 0.195    | 0.86  | 0.713  | 0.408    | 0.90  | 0.724  | 0.553    | 0.600             | 0.474         | 0.340             | 0.551   | 0.045 | 0.575  | 0.889                | 0.898             | 0.893         |
| 74  | 0.73  | 0.73   | 0.195    | 0.86  | 0.713  | 0.408    | 0.90  | 0.724  | 0.553    | 0.600             | 0.474         | 0.232             | 0.256   | 0.012 | 0.358  | 0.902                | 0.922             | 0.912         |
| 75  | 0.73  | 0.73   | 0.365    | 0.86  | 0.802  | 0.86     | 0.86  | 0.802  | 0.86     | 0.672             | 0.474         | 0.385             | 0.601   | 0.053 | 0.571  | 0.872                | 0.879             | 0.875         |
| 76  | 0.73  | 0.73   | 0.365    | 0.86  | 0.802  | 0.86     | 0.86  | 0.802  | 0.86     | 0.672             | 0.474         | 0.237             | 0.255   | 0.012 | 0.341  | 0.885                | 0.903             | 0.894         |
| 77  | 0.72  | 0.72   | 0.202    | 0.85  | 0.721  | 0.412    | 0.89  | 0.724  | 0.553    | 0.605             | 0.474         | 0.319             | 0.551   | 0.048 | 0.583  | 0.917                | 0.926             | 0.922         |
| 78  | 0.72  | 0.72   | 0.202    | 0.85  | 0.721  | 0.412    | 0.89  | 0.724  | 0.553    | 0.605             | 0.474         | 0.217             | 0.212   | 0.009 | 0.324  | 0.912                | 0.932             | 0.922         |
| 79  | 0.72  | 0.72   | 0.353    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.373             | 0.564   | 0.052 | 0.564  | 0.871                | 0.879             | 0.875         |
| 80  | 0.72  | 0.72   | 0.353    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.225             | 0.240   | 0.013 | 0.334  | 0.894                | 0.912             | 0.903         |
| 81  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.345             | 0.575   | 0.044 | 0.597  | 0.909                | 0.918             | 0.914         |
| 82  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.229             | 0.252   | 0.011 | 0.372  | 0.903                | 0.922             | 0.913         |
| 83  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.369             | 0.581   | 0.055 | 0.580  | 0.881                | 0.889             | 0.885         |
| 84  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.235             | 0.239   | 0.009 | 0.347  | 0.864                | 0.883             | 0.873         |
| 85  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.345             | 0.575   | 0.044 | 0.597  | 0.909                | 0.918             | 0.914         |
| 86  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.229             | 0.252   | 0.011 | 0.372  | 0.903                | 0.922             | 0.913         |
| 87  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.364             | 0.570   | 0.048 | 0.571  | 0.890                | 0.898             | 0.894         |
| 88  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.229             | 0.259   | 0.013 | 0.363  | 0.884                | 0.903             | 0.893         |
| 89  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.339             | 0.539   | 0.041 | 0.573  | 0.928                | 0.936             | 0.932         |
| 90  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.220             | 0.225   | 0.011 | 0.358  | 0.883                | 0.903             | 0.893         |
| 91  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.363             | 0.555   | 0.049 | 0.554  | 0.880                | 0.888             | 0.884         |
| 92  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.231             | 0.266   | 0.011 | 0.374  | 0.884                | 0.903             | 0.894         |
| 93  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.339             | 0.539   | 0.041 | 0.573  | 0.928                | 0.936             | 0.932         |
| 94  | 0.75  | 0.75   | 0.223    | 0.85  | 0.697  | 0.392    | 0.89  | 0.713  | 0.534    | 0.596             | 0.474         | 0.220             | 0.225   | 0.011 | 0.358  | 0.883                | 0.903             | 0.893         |
| 95  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.363             | 0.555   | 0.049 | 0.554  | 0.880                | 0.888             | 0.884         |
| 96  | 0.75  | 0.75   | 0.415    | 0.85  | 0.804  | 0.85     | 0.85  | 0.804  | 0.85     | 0.671             | 0.474         | 0.231             | 0.266   | 0.011 | 0.374  | 0.884                | 0.903             | 0.894         |
| 115 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 116 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.160             | 0.234   | 0.038 | 0.414  | 0.828                | 0.846             | 0.837         |
| 119 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 120 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.160             | 0.234   | 0.038 | 0.414  | 0.828                | 0.846             | 0.837         |
| 123 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 124 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.160             | 0.234   | 0.038 | 0.414  | 0.828                | 0.846             | 0.837         |
| 127 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.324             | 0.425   | 0.088 | 0.517  | 0.780                | 0.790             | 0.785         |
| 128 | 0.75  | 0.75   | 0.415    | 0.86  | 0.810  | 0.86     | 0.86  | 0.810  | 0.86     | 0.671             | 0.474         | 0.160             | 0.234   | 0.038 | 0.414  | 0.828                | 0.846             | 0.837         |
| 99  | 0.68  | 0.68   | 0.382    | 0.82  | 0.767  | 0.82     | 0.82  | 0.767  | 0.82     | 0.645             | 0.474         | 0.317             | 0.427   | 0.090 | 0.513  | 0.799                | 0.810             | 0.804         |
| 100 | 0.68  | 0.68   | 0.382    | 0.82  | 0.767  | 0.82     | 0.82  | 0.767  | 0.82     | 0.645             | 0.474         | 0.169             | 0.230   | 0.041 | 0.411  | 0.828                | 0.846             | 0.837         |
| 103 | 0.70  | 0.70   | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.321             | 0.424   | 0.082 | 0.506  | 0.799                | 0.810             | 0.804         |
| 104 | 0.70  | 0.70   | 0.387    | 0.82  | 0.772  | 0.82     | 0.82  | 0.772  | 0.82     | 0.645             | 0.474         | 0.172             | 0.230   | 0.041 | 0.411  | 0.828                | 0.846             | 0.837         |
| 107 | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.308             | 0.411   | 0.077 | 0.508  | 0.789                | 0.799             | 0.794         |
| 108 | 0.74  | 0.74   | 0.368    | 0.86  | 0.805  | 0.86     | 0.86  | 0.805  | 0.86     | 0.672             | 0.474         | 0.164             | 0.230   | 0.037 | 0.412  | 0.837                | 0.855             | 0.846         |
| 111 | 0.72  | 0.72   | 0.352    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.315             | 0.459   | 0.082 | 0.544  | 0.820                | 0.830             | 0.825         |
| 112 | 0.72  | 0.72   | 0.352    | 0.85  | 0.793  | 0.85     | 0.85  | 0.793  | 0.85     | 0.678             | 0.474         | 0.166             | 0.233   | 0.039 | 0.415  | 0.837                | 0.855             | 0.846         |
| 97  | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.287             | 0.440   | 0.077 | 0.552  | 0.867                | 0.879             | 0.873         |
| 98  | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.166             | 0.239   | 0.040 | 0.433  | 0.875                | 0.895             | 0.885         |
| 101 | 0.70  | 0.70   | 0.222    | 0.82  | 0.680  | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.288             | 0.442   | 0.078 | 0.558  | 0.867                | 0.879             | 0.873         |
| 102 | 0.70  | 0.70   | 0.222    | 0.82  | 0.680  | 0.397    | 0.85  | 0.678  | 0.508    | 0.574             | 0.474         | 0.164             | 0.243   | 0.040 | 0.435  | 0.876                | 0.895             | 0.885         |
| 105 | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.90  | 0.728  | 0.554    | 0.600             | 0.474         | 0.297             | 0.444   | 0.087 | 0.545  | 0.848                | 0.859             | 0.853         |
| 106 | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.90  | 0.728  | 0.554    | 0.600             | 0.474         | 0.165             | 0.234   | 0.038 | 0.421  | 0.856                | 0.875             | 0.865         |
| 109 | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.726  | 0.554    | 0.605             | 0.474         | 0.305             | 0.453   | 0.087 | 0.553  | 0.839                | 0.849             | 0.844         |
| 110 | 0.72  | 0.72   | 0.202    | 0.85  | 0.723  | 0.413    | 0.89  | 0.726  | 0.554    | 0.605             | 0.474         | 0.167             | 0.239   | 0.038 | 0.429  | 0.856                | 0.875             | 0.866         |
| 113 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 114 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 117 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 118 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 121 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 122 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 125 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 126 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 129 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 130 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 131 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 132 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 133 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 134 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 135 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.160             | 0.233   | 0.037 | 0.424  | 0.856                | 0.875             | 0.866         |
| 136 | 0.75  | 0.75   | 0.223    | 0.86  | 0.704  | 0.401    | 0.90  | 0.718  | 0.542    | 0.596             | 0.474         | 0.311             | 0.485   | 0.092 | 0.573  | 0.859                | 0.869             | 0.864         |
| 137 | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.90  | 0.728  | 0.554    | 0.600             | 0.474         | 0.434             | 0.725   | 0.101 | 0.658  | 0.907                | 0.912             | 0.909         |
| 138 | 0.74  | 0.74   | 0.197    | 0.86  | 0.718  | 0.409    | 0.90  | 0.728  | 0.554    | 0.600             | 0.474         | 0.179             | 0.262   | 0.039 | 0.448  | 0.904                | 0.924             | 0.914         |
| 139 | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.431             | 0.735   | 0.116 | 0.665  | 0.917                | 0.922             | 0.919         |
| 140 | 0.68  | 0.68   | 0.214    | 0.82  | 0.662  | 0.385    | 0.85  | 0.675  | 0.518    | 0.574             | 0.474         | 0.168             | 0.248   | 0.035 | 0.452  | 0.904                | 0.924             | 0.914         |


### Empfohlene Scores:

Retrieval:   ndcg@5
Context:     context_relevance
Generation:  bert_score_f1 + meteor

## Beste Konfigurationen bisher

Auswahl nach `Faith` innerhalb jedes LLMs. Bei sehr aehnlichen Scores werden Varianten mit besserer Abdeckung fuer Retrieval/Chunking bevorzugt. Leere `Retrieval_Top_K`-Felder in der alten Matrix sollten als fehlende Dokumentation behandelt und vor Re-Runs explizit gesetzt werden.

| Base_ID | LLM | Chunking | Size | Overlap | Retrieval | Template | Retrieval_Top_K | Reranker | Embedding | Dataset | Faith | Recall | N | Kommentar |
| ------- | --- | -------- | ---- | ------- | --------- | -------- | --------------- | -------- | --------- | ------- | ----- | ------ | --- | --------- |
| 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | no | nomic-embed-text | squad | 0.921 | - | 100 | Bester Qwen3-32B Lauf; starker Baseline-Kandidat. |
| 10 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Detailed | 12 | no | nomic-embed-text | squad | 0.904 | - | 100 | Kleinere Chunks, weiterhin stark. |
| 1 | Qwen3-32B | Recursive | 1000 | 100 | Similarity | Concise | 12 | no | nomic-embed-text | squad | 0.898 | - | 100 | Beste Concise-Variante fuer Qwen3-32B. |
| 5 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Concise | 12 | no | nomic-embed-text | squad | 0.889 | - | 100 | Gleiche Chunk-Laenge wie ID 6, anderes Template. |
| 2 | Qwen3-32B | Recursive | 1000 | 100 | Similarity | Detailed | 12 | no | nomic-embed-text | squad | 0.876 | - | 100 | Nahe Baseline fuer Overlap-Vergleich. |
| 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.955 | 0.930 | 100 | Bester Qwen3.5-397B Lauf. |
| 48 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Detailed | TBD | No | nomic-embed-text | squad | 0.949 | 0.650 | 100 | Hohe Faith, aber Recall deutlich niedriger; gut fuer Reranker/Top-K-Test. |
| 58 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.948 | 0.960 | 100 | Beste Semantic-Similarity-Variante. |
| 56 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Detailed | TBD | No | nomic-embed-text | squad | 0.947 | 0.653 | 100 | MMR mit hoher Faith, aber niedriger Recall. |
| 62 | Qwen3.5-397B | Semantic | 500 | 200 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Stabile Semantic-Alternative. |
| 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | TBD | No | nomic-embed-text | squad | 0.837 | 0.910 | 100 | Bester GPT-OSS-20B Lauf; anderes Muster als Qwen. |
| 69 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Concise | TBD | No | nomic-embed-text | squad | 0.836 | 0.960 | 100 | Sehr nah an ID 83 mit besserem Recall. |
| 87 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Concise | TBD | No | nomic-embed-text | squad | 0.830 | 0.910 | 100 | MMR/Overlap-Variante. |
| 93 | GPT-OSS-20B | Semantic | 500 | 200 | Similarity | Concise | TBD | No | nomic-embed-text | squad | 0.829 | 0.960 | 100 | Kleinere Semantic-Similarity-Variante. |
| 89 | GPT-OSS-20B | Semantic | 500 | 100 | Similarity | Concise | TBD | No | nomic-embed-text | squad | 0.829 | 0.960 | 100 | Kontrollvariante zu ID 93. |
| 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | TBD | No | nomic-embed-text | squad | 0.955 | 0.930 | 100 | Bester Thinking-Lauf. |
| 98 | qwen3.5-think | Recursive | 1000 | 100 | Similarity | Detailed | TBD | No | nomic-embed-text | squad | 0.950 | 0.960 | 100 | Beste lange Recursive-Variante. |
| 126 | qwen3.5-think | Semantic | 500 | 200 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Semantic mit hohem Recall. |
| 122 | qwen3.5-think | Semantic | 500 | 100 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Kontrollvariante zu ID 126. |
| 118 | qwen3.5-think | Semantic | 1000 | 200 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Lange Semantic-Variante. |
| 129 | gpt-4o-mini | Semantic | 1000 | 100 | - | Detailed | - | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Bester gpt-4o-mini Lauf; Faith auf Qwen3.5-Niveau. |
| 133 | gpt-4o-mini | Semantic | 500 | 100 | - | Detailed | - | No | nomic-embed-text | squad | 0.945 | 0.960 | 100 | Kleinere Chunks, identische Metriken zu ID 129. |
| 138 | deepseek-v4-pro | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | nomic-embed-text | squad | 0.938 | 0.930 | 100 | Bester deepseek-v4-pro Lauf; Faith auf hohem Niveau. |
| 137 | deepseek-v4-pro | Recursive | 500 | 100 | Similarity | Concise | 12 | No | nomic-embed-text | squad | 0.812 | 0.930 | 100 | Concise-Variante; niedrigere Faith aber hoechste meteor-Scores. |


## Phase-2 Variantenmatrix

Ziel: Nicht die gesamte Matrix als Cartesian Product vergroessern, sondern die besten Basiskonfigurationen gezielt auf Generalisierung und Robustheit testen. Primaere Gewinner pro LLM: `6`, `42`, `83`, `106`. Sekundaere Kandidaten fuer Cross-Checks: `10`, `58`, `69`, `98`.

| Set | Base_IDs | Zu testende Variante | Werte / Modelle | Zweck | Erwartete Runs |
| --- | -------- | -------------------- | --------------- | ----- | -------------- |
| P2.1 Stability | 6, 42, 83, 106 | Re-run auf `squad` | `N=300` oder `3x N=100`; gleiche Config; `Retrieval_Top_K=12`; HyDE off; Reranker off | Varianz schaetzen und pruefen, ob die Gewinner stabil sind. | 4-12 |
| P2.2 Dataset generalization | 6, 42, 83, 106, 10, 58, 69, 98 | Dataset | `squad`, `ragperf-wikipedia-nq`, `t2-ragbench/FinQA` | Pruefen, ob SQuAD-Gewinner auf offenen und finanznahen QA-Daten halten. | 24 |
| P2.3 Embedding sensitivity | 6, 42, 83, 106 | Embedding Model | `nomic-embed-text:latest`, `ollama:mxbai-embed-large:latest`, `huggingface:BAAI/bge-m3` | Klaeren, ob Retrieval-Ergebnisse am Embedding haengen. | 12 |
| P2.4 Top-K sweep | 6, 42, 83, 106 | Retrieval_Top_K | `5`, `12`, `20` | Trade-off zwischen Kontextmenge, Faithfulness und Recall messen. | 12 |
| P2.5 Reranker light | 6, 42, 83, 106 | Reranker | `Retrieval_Top_K=20`, `RERANKER_TOP_K=5`, `huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2` | Schneller Cross-Encoder-Test gegen No-Reranker-Baseline. | 4 |
| P2.6 Reranker strong | 6, 42, 83, 106 | Reranker | `Retrieval_Top_K=20`, `RERANKER_TOP_K=5`, `huggingface:BAAI/bge-reranker-base` | Staerkerer Reranker fuer finale Kandidaten. | 4 |
| P2.7 HyDE | 6, 42, 83, 106 | HyDE | `RETRIEVAL_USE_HYDE=true`, `Retrieval_Top_K=12`, Reranker off | Testen, ob Query Expansion bei Short-Question-Datasets hilft. | 4 |
| P2.8 HyDE + rerank | 6, 42, 83, 106 | HyDE + Reranker | `RETRIEVAL_USE_HYDE=true`, `Retrieval_Top_K=20`, `RERANKER_TOP_K=5`, `huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2` | Pruefen, ob mehr Recall durch HyDE per Reranker wieder bereinigt wird. | 4 |
| P2.9 MMR lambda | 48, 56, 83, 87 | MMR lambda | `RETRIEVAL_MMR_LAMBDA=0.25`, `0.5`, `0.75`; `Retrieval_Top_K=12` | MMR ist gemischt: Faith oft gut, Recall teils schwach. Lambda isoliert Diversitaet vs. Relevanz. | 12 |
| P2.10 Final combined | 6, 42, 83, 106 | Best-of stack | `huggingface:BAAI/bge-m3`, `Retrieval_Top_K=20`, `huggingface:BAAI/bge-reranker-base`, `RERANKER_TOP_K=5`, HyDE off; Datasets `squad`, `ragperf-wikipedia-nq`, `t2-ragbench/FinQA` | Finaler robuster Vergleich gegen die einfache Baseline. | 12 |

### Phase-2 Prioritaet

1. Zuerst `P2.1`, `P2.2`, `P2.4`: Stabilitaet, Dataset-Generalisation und Top-K sind die wichtigsten Validitaetsluecken.
2. Danach `P2.5` und `P2.6`: Reranker nur auf Gewinnern testen, weil er teuer ist.
3. Danach `P2.3`, `P2.7`, `P2.8`, `P2.9`: Embedding, HyDE und MMR-Lambda erklaeren die Ursachen der Unterschiede.
4. Zum Schluss `P2.10`: Finaler Stack fuer Paper-/Poster-Claims.

## API-Kosten (Frontier-Modelle)

Nur fuer Runs mit kostenpflichtigen API-Modellen (nicht lokale Ollama-Modelle).

| Run | IDs | LLM | Kosten (USD) | Hinweis |
| --- | --- | ------------ | ------------ | ------- |
| 75 | 137, 138 | deepseek-v4-pro | $0.12 | API-Key, 2 Configs a 100 Samples |
| 76 | 139, 140 | deepseek-v4-pro | $0.18 | API-Key, 2 Configs a 100 Samples |
