# Northstar Facilities v1

Northstar Facilities is a deterministic synthetic corpus for testing RAG
ingestion, retrieval, generation, provenance mapping, and multi-source recall.
It contains no claims about real facilities.

## Contents

- `corpus/`: 60 Markdown documents to upload to a RAG system.
- `questions.jsonl`: 360 Benchmark Framework-compatible QA cases.
- `manifest.json`: checksums, distributions, intended use, and limitations.
- `validation_report.json`: machine-readable quality-gate result.
- `LICENSE.md`: CC0-1.0 dedication and canonical legal-code link.
- `generate.py`: deterministic corpus and question generator.
- `validate.py`: schema, evidence, integrity, and reproducibility checks.

Each case has the framework's required `question`, `ground_truth`, `context`,
and `metadata` fields. It also carries top-level compatibility aliases and
explicit gold fields: `case_id`, `reference_answer`, `relevant_source_ids`,
`gold_chunk_ids`, and line-addressed `evidence`. Stable semantic chunk IDs use
`SOURCE_ID#section`; an external adapter should retain the source filename/code
when mapping provider-generated document and chunk IDs back to gold evidence.
Documents contain roughly 460–550 words across operational and distractor
sections, so common 128/256/512-token chunk settings produce different
boundaries and chunk counts.

Generate and validate:

```bash
python datasets/northstar_facilities/generate.py
python datasets/northstar_facilities/validate.py
```

Use it with the existing local JSONL loader:

```bash
DATASET_NAME=jsonl \
DATASET_PATH=datasets/northstar_facilities/questions.jsonl \
python main.py
```

Set the run's sample size to `360` (or `0`, if the caller uses zero for all
rows). Upload the Markdown files under `corpus/` when benchmarking an external
system that owns ingestion.

## Scope and limitations

This dataset is intentionally controlled and deterministic. It is a strong
integration and regression fixture because every answer is traceable to exact
source lines. It is not a substitute for a human-reviewed, domain-specific
evaluation set: the questions are template-generated, the entities are
fictional, and the language distribution is English-only. The dataset-specific
materials are released under [CC0-1.0](LICENSE.md).
