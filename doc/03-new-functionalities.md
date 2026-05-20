# New Functionalities

## 1. Adaptive Retrieval

### Self-RAG
Implement Self-RAG (Asai et al., 2023):
- Model learns when to retrieve, when not to
- Reflection tokens for retrieval necessity, relevance, support, utility
- Requires fine-tuned model or prompt-based approximation

### Corrective RAG (CRAG)
Implement CRAG (Yan et al., 2024):
- Evaluate retrieval quality before generation
- If retrieval is poor, fall back to web search or refuse
- Add `retrieval_evaluator` that scores retrieved context

### Adaptive Chunking
Currently fixed chunk sizes. Add adaptive chunking:
- Vary chunk size based on document structure (paragraphs, sections, sentences)
- Document-type-aware chunking (code, tables, prose, lists)
- Multi-granularity indexing (same document at multiple chunk sizes)

## 2. Multi-Modal RAG

### Image/Table Understanding
Current framework is text-only. Add:
- Image extraction from PDF documents
- Table extraction and structured representation
- Multi-modal embedding (CLIP, Llava)

### Document Processing Pipeline
No document ingestion pipeline. Add:
- PDF parser (PyMuPDF, unstructured.io)
- HTML parser
- DOCX parser
- OCR for scanned documents

## 3. RAG Pipeline Optimization

### Automatic Prompt Optimization
`prompts.md` has 9 manual prompt templates. Add:
- Automatic prompt tuning via DSPy or similar
- Prompt performance tracking
- A/B test prompts automatically

### Retrieval Parameter Auto-Tuning
Currently manual grid search. Add:
- Bayesian optimization for chunk_size, overlap, top_k, lambda
- Early stopping for clearly suboptimal configs
- Multi-objective optimization (speed vs quality)

### Cost Estimation
No cost tracking. Add:
- Token counting per provider
- Cost estimation (per-query and per-run)
- Budget limits and alerts

## 4. Production Readiness

### REST API
No API for triggering/monitoring benchmarks. Add:
- FastAPI server for benchmark management
- WebSocket for real-time progress
- Authentication and multi-user support

### Containerization
No Docker support. Add:
- Dockerfile with all dependencies
- docker-compose for MLflow + benchmark runner
- GPU passthrough configuration

### Configuration Management
Only env vars. Add:
- YAML/TOML config files
- Config validation with Pydantic
- Config profiles (local, remote, cloud)
- Config inheritance (base config + overrides)

## 5. Analysis Tools

### Error Analysis
No systematic error analysis. Add:
- Per-sample failure categorization (retrieval failure, generation failure, etc.)
- Automatic clustering of failure modes
- Example-based debugging (show worst samples)

### Metric Correlation Analysis
No correlation between metrics. Add:
- Correlation matrix across all metrics
- Redundancy detection (which metrics provide unique signal?)
- Metric sensitivity analysis (which metrics change most with config changes?)

### Learning Curve Analysis
No analysis of how metrics scale with sample size. Add:
- Learning curve plots (metric vs N)
- Confidence interval narrowing visualization
- Minimum sample size estimation

## 6. Collaboration Features

### Experiment Sharing
No mechanism to share experiments. Add:
- Export/import run results
- Shareable report URLs
- Benchmark result database

### Leaderboard
No leaderboard system. Add:
- Per-dataset leaderboard
- Per-model comparison table
- Historical performance tracking

## 7. Agentic Enhancements

### Multi-Agent Benchmarking
Current agentic runner is single-agent. Add:
- Separate agents for retrieval optimization, generation optimization, evaluation
- Agent communication protocol
- Meta-agent for orchestration

### Automated Research
No automated hypothesis generation. Add:
- Pattern detection in results
- Automatic hypothesis generation ("MMR seems to hurt faithfulness for large models")
- Experimental design for hypothesis validation

### Continuous Benchmarking
No continuous monitoring. Add:
- Scheduled benchmark runs
- Performance regression alerts
- Model update detection (when Ollama model changes)
