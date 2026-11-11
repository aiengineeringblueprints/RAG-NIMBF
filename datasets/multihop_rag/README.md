# MultiHop-RAG

[MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) (Tang & Yang, COLM 2024,
[arXiv:2401.15391](https://arxiv.org/abs/2401.15391)) tests retrieval and reasoning
across documents: 2556 queries over 609 English news articles (Sep–Dec 2023),
each answerable query needing evidence from 2–4 articles.

| question_type | Queries | Gold documents |
|---|---|---|
| `comparison_query` | 856 | 2–3 |
| `inference_query` | 816 | 2–4 |
| `temporal_query` | 583 | 2–3 |
| `null_query` | 301 | none (expects a decline) |

## Prepare

The articles are not committed (third-party news content). Generate the
`jsonl-shared` layout from the pinned Hugging Face revision:

```bash
python datasets/multihop_rag/prepare.py
```

This writes `corpus/MHR-001.md` … `MHR-609.md` and `questions.jsonl`.

## Conversion

- Each article file starts with title, source, author, publication date and
  category, because many queries name the outlet or the date.
- Evidence articles become `metadata.gold_doc_ids` (deduplicated), so
  `custom_retrieval_metrics_mode: gold_doc` and
  `scripts/retrieval_by_category.py` work.
- `question_type` is also stored as `metadata.category` for the adapters'
  per-category answer metrics.
- `answer_patterns` holds the reference answer. Yes/no answers match only when
  the opposite word is absent, so a hedged "yes or no" counts for neither.
- Null queries carry `expected_behavior: decline` and no gold documents.
- Questions are shuffled with a fixed seed, so any `sample_size` prefix mixes
  all question types.

## License

Dataset: ODC-BY (as published on Hugging Face). The articles remain the
property of their publishers.
