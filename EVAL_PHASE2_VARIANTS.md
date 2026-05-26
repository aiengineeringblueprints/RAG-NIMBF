# Phase-2 Benchmark Variantenmatrix

Diese Datei expandiert die in `EVAL_MATRIX.md` geplante Phase-2-Variantenmatrix in einzelne ausfuehrbare Benchmark-Konfigurationen. Ergebniswerte bleiben leer, bis die Runs abgeschlossen sind.

## Parameter-Tabelle

| Nr | Set | Base_ID | LLM | Chunking | Size | Overlap | Retrieval | Template | Retrieval_Top_K | Reranker | Reranker_Top_K | HyDE | MMR_Lambda | Embedding | Dataset | Faith | ndcg@5 | ctx_rel | meteor | bert_f1 | Semsim | N | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | P2.1 Stability | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 300 | Offen |
| 2 | P2.1 Stability | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | 0.959 | 0.702 | 0.634 | 0.043 | 0.835 | 0.648 | 300 | Getestet |
| 3 | P2.1 Stability | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 300 | Offen |
| 4 | P2.1 Stability | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 300 | Offen |
| 5 | P2.2 Dataset generalization | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 6 | P2.2 Dataset generalization | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 7 | P2.2 Dataset generalization | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 8 | P2.2 Dataset generalization | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | 0.955 | 0.728 | 0.600 | 0.038 | 0.865 | 0.655 | 100 | Getestet |
| 9 | P2.2 Dataset generalization | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 10 | P2.2 Dataset generalization | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | 0.743 | 0.204 | 0.716 | 0.0 | 0.549 | 0.457 | 100 | Getestet |
| 11 | P2.2 Dataset generalization | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 12 | P2.2 Dataset generalization | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 13 | P2.2 Dataset generalization | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 14 | P2.2 Dataset generalization | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 15 | P2.2 Dataset generalization | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 16 | P2.2 Dataset generalization | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 17 | P2.2 Dataset generalization | 10 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 18 | P2.2 Dataset generalization | 10 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 19 | P2.2 Dataset generalization | 10 | Qwen3-32B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 20 | P2.2 Dataset generalization | 58 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 21 | P2.2 Dataset generalization | 58 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 22 | P2.2 Dataset generalization | 58 | Qwen3.5-397B | Semantic | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 23 | P2.2 Dataset generalization | 69 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Concise | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 24 | P2.2 Dataset generalization | 69 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Concise | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 25 | P2.2 Dataset generalization | 69 | GPT-OSS-20B | Recursive | 1000 | 200 | Similarity | Concise | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 26 | P2.2 Dataset generalization | 98 | qwen3.5-think | Recursive | 1000 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 27 | P2.2 Dataset generalization | 98 | qwen3.5-think | Recursive | 1000 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 28 | P2.2 Dataset generalization | 98 | qwen3.5-think | Recursive | 1000 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 29 | P2.3 Embedding sensitivity | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 30 | P2.3 Embedding sensitivity | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | ollama:mxbai-embed-large:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 31 | P2.3 Embedding sensitivity | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 32 | P2.3 Embedding sensitivity | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 33 | P2.3 Embedding sensitivity | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | ollama:mxbai-embed-large:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 34 | P2.3 Embedding sensitivity | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 35 | P2.3 Embedding sensitivity | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 36 | P2.3 Embedding sensitivity | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | ollama:mxbai-embed-large:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 37 | P2.3 Embedding sensitivity | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 38 | P2.3 Embedding sensitivity | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 39 | P2.3 Embedding sensitivity | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | ollama:mxbai-embed-large:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 40 | P2.3 Embedding sensitivity | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 41 | P2.4 Top-K sweep | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 5 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 42 | P2.4 Top-K sweep | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 43 | P2.4 Top-K sweep | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 44 | P2.4 Top-K sweep | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 5 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 45 | P2.4 Top-K sweep | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 46 | P2.4 Top-K sweep | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 47 | P2.4 Top-K sweep | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 5 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 48 | P2.4 Top-K sweep | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 49 | P2.4 Top-K sweep | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 50 | P2.4 Top-K sweep | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 5 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 51 | P2.4 Top-K sweep | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 52 | P2.4 Top-K sweep | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | No | - | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 53 | P2.5 Reranker light | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 54 | P2.5 Reranker light | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 55 | P2.5 Reranker light | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 56 | P2.5 Reranker light | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 57 | P2.6 Reranker strong | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 58 | P2.6 Reranker strong | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 59 | P2.6 Reranker strong | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 60 | P2.6 Reranker strong | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 61 | P2.7 HyDE | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 12 | No | - | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 62 | P2.7 HyDE | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 63 | P2.7 HyDE | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 64 | P2.7 HyDE | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 12 | No | - | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 65 | P2.8 HyDE + rerank | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 66 | P2.8 HyDE + rerank | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 67 | P2.8 HyDE + rerank | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 68 | P2.8 HyDE + rerank | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:cross-encoder/ms-marco-MiniLM-L-6-v2 | 5 | true | - | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 69 | P2.9 MMR lambda | 48 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Detailed | 12 | No | - | false | 0.25 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 70 | P2.9 MMR lambda | 48 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Detailed | 12 | No | - | false | 0.5 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 71 | P2.9 MMR lambda | 48 | Qwen3.5-397B | Recursive | 500 | 200 | MMR | Detailed | 12 | No | - | false | 0.75 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 72 | P2.9 MMR lambda | 56 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Detailed | 12 | No | - | false | 0.25 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 73 | P2.9 MMR lambda | 56 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Detailed | 12 | No | - | false | 0.5 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 74 | P2.9 MMR lambda | 56 | Qwen3.5-397B | Semantic | 1000 | 200 | MMR | Detailed | 12 | No | - | false | 0.75 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 75 | P2.9 MMR lambda | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | 0.25 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 76 | P2.9 MMR lambda | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | 0.5 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 77 | P2.9 MMR lambda | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 12 | No | - | false | 0.75 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 78 | P2.9 MMR lambda | 87 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Concise | 12 | No | - | false | 0.25 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 79 | P2.9 MMR lambda | 87 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Concise | 12 | No | - | false | 0.5 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 80 | P2.9 MMR lambda | 87 | GPT-OSS-20B | Semantic | 1000 | 200 | MMR | Concise | 12 | No | - | false | 0.75 | nomic-embed-text:latest | squad | - | - | - | - | - | - | 100 | Offen |
| 81 | P2.10 Final combined | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 82 | P2.10 Final combined | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 83 | P2.10 Final combined | 6 | Qwen3-32B | Recursive | 1000 | 200 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 84 | P2.10 Final combined | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 85 | P2.10 Final combined | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 86 | P2.10 Final combined | 42 | Qwen3.5-397B | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 87 | P2.10 Final combined | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 88 | P2.10 Final combined | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 89 | P2.10 Final combined | 83 | GPT-OSS-20B | Semantic | 1000 | 100 | MMR | Concise | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |
| 90 | P2.10 Final combined | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | squad | - | - | - | - | - | - | 100 | Offen |
| 91 | P2.10 Final combined | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | ragperf-wikipedia-nq | - | - | - | - | - | - | 100 | Offen |
| 92 | P2.10 Final combined | 106 | qwen3.5-think | Recursive | 500 | 100 | Similarity | Detailed | 20 | huggingface:BAAI/bge-reranker-base | 5 | false | - | huggingface:BAAI/bge-m3 | t2-ragbench/FinQA | - | - | - | - | - | - | 100 | Offen |

## Hinweise

- `Faith`, `Rel`, `Corr`, `Prec` und `Rec` sind absichtlich leer und werden nach den Runs befuellt.
- `P2.1` nutzt `N=300`, damit Stabilitaet ohne separate Repeat-Zeilen geprueft wird.
- `Retrieval_Top_K` ist fuer alle Varianten explizit gesetzt; alte `TBD`/leere Werte aus `EVAL_MATRIX.md` werden hier nicht uebernommen.
- `Reranker_Top_K=-` bedeutet: kein Reranker aktiv.
