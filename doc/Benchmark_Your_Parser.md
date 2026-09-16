# Benchmarking a document parser with the framework

This guide describes how to benchmark a document parser (OCR / PDF-to-Markdown
system) against document ground-truth datasets such as OmniDocBench, DP-Bench,
and olmOCR-Bench. For benchmarking RAG systems, see
[doc/Benchmark_Your_RAG.md](Benchmark_Your_RAG.md).

## 1. Choose an integration flavor

| Goal | Recommended way |
| --- | --- |
| Parser has an OpenAI-compatible vision `chat/completions` endpoint | HTTP adapter (`parser_adapters: ["http"]`) |
| Parser is Python code in the repo or importable | Plugin adapter (`PARSER_PLUGIN_MODULE` / `PARSER_PLUGIN_ATTRIBUTE`) |
| One-off experimentation without a manifest | Env vars (`PARSER_ADAPTER`, see README table) |

Adapters implement the `DocumentParser` protocol in
`benchmark/parsing/base.py`: attributes `name` and `parser_version`, plus
`parse(document, config=None) -> ParseResult`. A `ParseResult` carries
`pages` (`ParsedPage(page_number, markdown)`), timing, and diagnostics.

### HTTP adapter

Each page is sent as one `chat/completions` request (pages with `image_base64`
data are sent as `image_url` content parts). Required env:

```bash
PARSER_HTTP_ENDPOINT_URL=https://your-endpoint/v1/chat/completions
# optional:
PARSER_HTTP_MODEL=your-vlm-id
PARSER_HTTP_PROMPT="Convert this page to Markdown."
PARSER_HTTP_TIMEOUT_SECONDS=120
PARSER_HTTP_HEADERS='{"Authorization": "Bearer ..."}'
```

### Plugin adapter

Point the framework at an importable class that accepts the benchmark config
as an optional constructor argument (mirroring the RAG plugin convention):

```bash
PARSER_PLUGIN_MODULE=my_parsers
PARSER_PLUGIN_ATTRIBUTE=MyParser
```

Custom adapter flavors can be registered with
`register_parser_adapter(name, factory)` from `benchmark.parsing`.

## 2. Prepare a ground-truth dataset

All three v1 datasets are local-file loaders — nothing is downloaded. Point
`DATASET_PATH` at a local export:

| Dataset | Manifest | GT fields | License |
| --- | --- | --- | --- |
| OmniDocBench | `experiments/ocr_parser_omnidocbench.yaml` | layout_dets text + HTML tables | research-only |
| DP-Bench | `experiments/ocr_parser_dp_bench.yaml` | `gt_markdown` / `gt_html` | MIT |
| olmOCR-Bench | `experiments/ocr_parser_olmocr_bench.yaml` | `expected_text` (rows without it are skipped) | Apache-2.0 |

Every sample carries `dataset_license` / `dataset_research_only` metadata that
is pinned into run manifests, MLflow tags, and the leaderboard report.

## 3. Run the benchmark

```bash
BENCHMARK_STAGE=parsing \
DATASET_PATH=/local/omnidocbench.json \
PARSER_HTTP_ENDPOINT_URL=... \
python -m benchmark.worker run experiments/ocr_parser_omnidocbench.yaml --run-dir results/runN
```

The manifest declares the matrix (`parser_adapters`, multiple datasets, etc.);
the worker expands it, parses and scores every document, and is resumable via
per-document checkpoints (`run_dir/configs/<cell>_parsing/<doc_id>.json`).

Use `python -m benchmark.worker plan <manifest>` to inspect the expanded
matrix without running anything.

## 4. Scores and reports

- **Text metrics:** CER, WER, normalized edit distance (alignment via the
  vendored OmniDocBench `quick_match` matcher).
- **Tables:** TEDS and structure-only TEDS on Markdown tables converted to
  HTML.
- **Overall:** combined score; the primary leaderboard ranking key.

Outputs per run:

- `parsing_leaderboard_<timestamp>.json` — parsers ranked by `overall`, with
  per-category breakdown, match-algorithm metadata, and dataset license flags.
- `<run_dir>/configs/<cell>_parsing/page_details.json` — per-page scores for
  every document.
- MLflow: one nested child run per parser × dataset cell, plus the aggregate
  leaderboard artifact on the parent run.

## 5. Using a parser for RAG corpus ingestion

To build a RAG corpus through a parser instead of feeding pre-chunked text,
set `CORPUS_PARSER` (see README env-var table). This requires the
`jsonl-shared` dataset format and `DATASET_CORPUS_PATH`; documents are parsed
to Markdown before chunking.
