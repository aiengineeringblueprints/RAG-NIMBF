# RAGBench Integration Notes

## Source

- **Paper**: RAGBench (arXiv:2407.11005v2)
- **HuggingFace**: `rungalileo/ragbench`
- **Schema**: 12 component datasets under one HF dataset, each selected via the
  `name=` (config) argument to `datasets.load_dataset`.

All 12 components share an identical column schema (verified against every
component's `test` split). Question text, ground-truth answer, and document
context are always under the same keys, so a single parameterised adapter
covers all 12.

## Component datasets

| Domain        | Adapter name           | Component   | Source dataset     | Typical context size |
| ------------- | ---------------------- | ----------- | ------------------ | -------------------- |
| Bio-medical   | `ragbench_pubmedqa`    | `pubmedqa`  | PubMedQA           | medium               |
| Bio-medical   | `ragbench_covidqa`     | `covidqa`   | CovidQA            | medium               |
| General       | `ragbench_hotpotqa`    | `hotpotqa`  | HotpotQA           | multi-doc            |
| General       | `ragbench_msmarco`     | `msmarco`   | MS Marco           | medium               |
| General       | `ragbench_hagrid`      | `hagrid`    | HAGRID             | medium               |
| General       | `ragbench_expertqa`    | `expertqa`  | ExperQA            | long                 |
| Legal         | `ragbench_cuad`        | `cuad`      | CUAD               | ~11k tokens (long)   |
| Technical     | `ragbench_emanual`     | `emanual`   | EManual            | medium               |
| Technical     | `ragbench_techqa`      | `techqa`    | TechQA             | medium               |
| Financial     | `ragbench_finqa`       | `finqa`     | FinQA              | tabular              |
| Financial     | `ragbench_tatqa`       | `tatqa`     | TAT-QA             | tabular/numerical    |
| Conversation  | `ragbench_delucionqa`  | `delucionqa`| DelucionQA         | conversational       |

### Split sizes (test split, approximate)

- `pubmedqa`: 24.5k total
- `tatqa`: 33.1k total (largest)
- `finqa`: 16.6k total
- `hagrid`: 4.53k total
- `hotpotqa`: 2.7k total
- `msmarco`: 2.69k total
- `cuad`: 2.55k total
- `expertqa`: 2.03k total
- `delucionqa`: 1.83k total
- `techqa`: 1.81k total
- `covidqa`: 1.77k total
- `emanual`: 1.32k total

Each component ships `train`, `validation`, and `test` splits. The adapters
default to `test`; override with `dataset.split` in YAML or `DATASET_SPLIT`
in env.

## License notes

RAGBench is released under the MIT license (see the HuggingFace dataset card).
The underlying component datasets carry their own licenses (CUAD = CC BY 4.0,
HotpotQA = CC BY-SA 4.0, etc.). RAGBench's license covers the augmented
annotations and the LLM-generated responses; downstream users remain
responsible for compliance with the original component licenses.

## HuggingFace download requirements

- **Network**: outbound HTTPS to `huggingface.co` is required on first load.
- **Auth**: `rungalileo/ragbench` is public — no HF token required.
- **Cache**: HuggingFace caches downloads under `~/.cache/huggingface/`. The
  `datasets` library handles this automatically; the framework does **not**
  re-download on subsequent runs.
- **TRACe sidecar**: in addition to the HF cache, the framework writes a
  per-component/split sidecar JSON of sentence-level gold annotations to
  `datasets/ragbench/<component>/<split>/trace_gold.json` (root configurable
  via `DATASET_CACHE_ROOT`). This sidecar is for downstream TRACe metric
  scoring (separate worktree) and is rewritten on each load — it is cheap to
  regenerate.

### Fail-fast behaviour

If the HF download fails (no network, hub outage, etc.), the loader raises a
`RuntimeError` with a clear message:

> `Failed to download dataset 'rungalileo/ragbench' (subset='cuad'). Network
> access to HuggingFace is required. Original error: ...`

## Schema mapping (RAGBench → framework)

| Framework field     | RAGBench column                  | Notes |
| ------------------- | -------------------------------- | ----- |
| `question`          | `question`                       | str |
| `ground_truth`      | `response`                       | LLM-generated reference answer |
| `context`           | `documents` (joined by `\n\n`)   | falls back to `unannotated_context`, `gpt3_context`, `context` |
| `metadata.id`       | `id`                             | per-example ID |
| `metadata.dataset_name`        | `dataset_name`       | e.g. `cuad_test` |
| `metadata.generation_model_name`  | `generation_model_name` | e.g. `gpt-3.5-turbo-1106` |
| `metadata.annotating_model_name`  | `annotating_model_name` | e.g. `gpt-4o` |
| `metadata.ragbench_trace`        | (see below)          | TRACe gold annotations |

### TRACe annotations (`metadata.ragbench_trace`)

Stored as a dict and JSON-serialised by the Chroma metadata coercion when the
sample flows through `load_corpus_and_questions`. Contains:

- `documents_sentences`, `response_sentences` — sentence-keyed breakdown
- `sentence_support_information` — per-response-sentence support evidence
- `unsupported_response_sentence_keys`
- `all_relevant_sentence_keys`, `all_utilized_sentence_keys` — TRACe gold spans
- `relevance_score`, `utilization_score`, `completeness_score` (0–1)
- `adherence_score` (bool)

## Sample YAML

```yaml
experiment_name: ragbench-cuad-smoke

dataset:
  name: ragbench_cuad
  split: test
  max_examples: 20

settings:
  rag_system_adapter: internal
  retrieval_mode: retrieval
  retrieval_top_k: 8
  max_new_tokens: 512
  prompt_template: concise
  vector_db_backend: chroma
  custom_retrieval_metrics_mode: gold_doc

matrix:
  llm_models:
    - ollama:gpt-oss:20b
  embedding_models:
    - nomic-embed-text:latest
  chunking_strategies:
    - recursive
  chunk_sizes:
    - 512
  chunk_overlaps:
    - 64
  reranker_models:
    - null
```

### Selecting a split other than `test`

RAGBench splits are `train`, `validation`, `test`. Use `validation` for
benchmark/dev runs:

```yaml
dataset:
  name: ragbench_pubmedqa
  split: validation
  max_examples: 100
```

(Domain papers and the original RAGBench release also use the name `dev`;
this framework uses the literal HuggingFace split name `validation`.)

### Running a grid across components

```yaml
dataset:
  name:
    - ragbench_cuad
    - ragbench_pubmedqa
    - ragbench_finqa
  split: test
  max_examples: 50
```

## Adding a new RAGBench component

If a future RAGBench release adds a 13th component:

1. Edit `benchmark/ragbench_adapter.py`.
2. Add the component to `RAGBENCH_COMPONENTS` (name → display name).
3. The adapter `ragbench_<name>` is registered automatically on import; no
   other code changes are needed as long as the new component uses the same
   column schema (`question`, `documents`, `response`,
   `all_relevant_sentence_keys`, etc.).

If the schema ever diverges per component, replace the parameterised
registration with per-component `build_context` / `metadata_keys` overrides.

## Adapter module placement

The task brief suggested `benchmark/adapters/ragbench_adapter.py`. That path
conflicts with the existing `benchmark/adapters/` namespace, which is
exclusively for RAG-system adapters (`base.py`, `http.py`, `mcp.py`,
`components.py`). Placing a dataset adapter there would mislead readers. The
implementation lives at `benchmark/ragbench_adapter.py` (sibling to the
existing `benchmark/dataset_adapters.py`), matching the repo's flat-module
convention for dataset adapter code. It is imported for its side effect from
the bottom of `benchmark/dataset_adapters.py` so registration happens on the
first `resolve_adapter` lookup.
