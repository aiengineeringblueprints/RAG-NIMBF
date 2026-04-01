# Deep Dive: Optimization Report

## CRITICAL Issues (must fix)

### 1. Parallel configs overwhelm single-GPU Ollama
**Location:** `main.py:167`
`ThreadPoolExecutor(max_workers=len(configs))` launches all configs simultaneously against one local Ollama instance. This causes GPU memory thrashing, request queuing, timeouts, and *lower* throughput than sequential execution. The `max_workers=1` safeguard in `evaluation.py:47` is undermined because multiple benchmarks call it concurrently.

### 2. Embeddings recomputed redundantly
**Location:** `main.py:38-40`
Every config re-embeds all chunks from scratch via Ollama HTTP calls. With 3 embedding models x 4 chunk sizes, that's 12 embedding passes where many share the same model on the same data. Hundreds of HTTP calls per pass.

### 3. ChromaDB memory leak
**Location:** `retrieval.py:9-18`
`EphemeralClient` is a process-lifetime global. Every config creates a new collection that is never cleaned up. With 12 configs x ~1000 embedded chunks each, this consumes GBs of RAM.

### 4. Same model judges its own answers
**Location:** `evaluation.py:36-38`
RAGAS uses `config.llm_model` as the critic LLM -- the exact same model that generated the answers. A 4B parameter model evaluating its own work introduces severe self-consistency bias.

### 5. RAGAS failures silently swallowed
**Location:** `main.py:67-79`
`except Exception` catches all RAGAS errors, prints a message, then continues with empty scores. The report shows `None` for all RAGAS fields with no indication that evaluation *failed* vs. *was not run*.

### 6. Thread-unsafe global ChromaDB state
**Location:** `retrieval.py:28-38`
The lock protects `delete_collection` but not the subsequent `Chroma(...)` constructor or `add_documents()`. Two threads can race on collection creation/deletion.

---

## HIGH Issues (should fix)

| # | Issue | Location |
|---|-------|----------|
| 7 | `nvidia-smi` subprocess spawned per question (600+ invocations) | `metrics.py:6-15` |
| 8 | TTFT field holds total latency, not time-to-first-token | `generation.py:51` |
| 9 | `tokens_per_second` averaged via arithmetic mean instead of harmonic mean | `main.py:134` |
| 10 | Ranking composite score treats `None` metrics as `0.0` | `analysis.py:96-99` |
| 11 | `worst_metrics` uses `v == len(results)` -- wrong with ties/None | `analysis.py:122` |
| 12 | GPU stat dict keys accessed without `.get()` | `main.py:106-115` |
| 13 | CSV per-sample export silently truncates data | `exports.py:152-154` |
| 14 | No env var validation -- nonsensical values crash or produce garbage | `config.py:38-48` |
| 15 | No timeout on `llm.invoke()` -- hangs indefinitely | `generation.py:35` |
| 16 | `load_dotenv()` runs at module import time | `config.py:6` |
| 17 | `BenchmarkConfig` and `GenerationResult` should be `frozen=True` | `config.py:18`, `generation.py:11` |
| 18 | `asyncio.get_event_loop()` deprecated since Python 3.10 | `main.py:166` |
| 19 | No `logging` module -- no log files, no verbosity control | All files |
| 20 | LLM/embedding instances recreated in `evaluation.py` | `evaluation.py:35-41` |

---

## MEDIUM Issues

| # | Issue | Location |
|---|-------|----------|
| 21 | Dataset loaded from `configs[0]` only | `main.py:159-162` |
| 22 | Shared RAGAS metric objects may have race conditions | `evaluation.py:43` |
| 23 | `.env.example` has `RETRIEVAL_TOP_K=3`, code defaults to `5` | `.env.example:7` vs `config.py:44` |
| 24 | Magic numbers hardcoded (temperature, timeout, thresholds) | Multiple files |
| 25 | `evaluate_results` has 7 params -- should accept config object | `evaluation.py:13-21` |
| 26 | DRY violations: 3 truncation helpers, 2 float-format helpers | `terminal.py`, `exports.py`, `visualization.py` |
| 27 | Unused imports: `asdict`, `datetime` | `exports.py:4-5` |
| 28 | `compute_stats` filters NaN but not `inf` | `models.py:74` |
| 29 | No atomic file writes | `exports.py:94` |
| 30 | Markdown report missing system info and wall time | `reporting/__init__.py:62-68` |
| 31 | Metric label dicts duplicated | `analysis.py` and `visualization.py` |
| 32 | Dependencies pinned with `>=` only | `requirements.txt` |

---

## Architecture & Feature Gaps

| Gap | Impact |
|-----|--------|
| No Protocol/ABC abstractions -- locked to Ollama | Can't benchmark cloud APIs |
| Hardcoded RAGAS metrics -- 8 fields per metric | Adding a 5th metric touches 8+ files |
| God function `run_single_benchmark` (126 lines) | Hard to maintain or extend |
| No caching -- identical embedding work repeated | Wastes hours of compute |
| No resumability -- crash loses all progress | Painful for long runs |
| No prompt template config | Can't benchmark prompting strategies |
| No CLI arguments | No overrides without editing `.env` |
| No experiment tracking | No regression tracking |
| No statistical significance testing | Can't tell if differences are meaningful |
| No tests (0%) | Untestable without running infrastructure |
