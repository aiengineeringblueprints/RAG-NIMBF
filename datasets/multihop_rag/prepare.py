"""Convert MultiHop-RAG (HF yixuantt/MultiHopRAG) into a jsonl-shared dataset.

Downloads the pinned dataset revision and writes
  corpus/MHR-001.md ... MHR-609.md   one news article per file
  questions.jsonl                    2556 queries in seeded random order

Each article starts with its title, source, author, date and category: many
queries name the outlet or the publication date ("as reported by The Verge").
Evidence articles become ``gold_doc_ids``; ``question_type`` (inference,
comparison, temporal, null) is also stored as ``category`` for the adapters'
per-category metrics. Null queries expect a decline.

Usage: python datasets/multihop_rag/prepare.py [--out datasets/multihop_rag]
"""

from __future__ import annotations

import argparse
import json
import random
import urllib.request
from pathlib import Path

REPO = "yixuantt/MultiHopRAG"
REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
SEED = 20240129

# Yes/no answers match only when the opposite word is absent, so a hedged
# "yes or no" answer does not count for either label.
_YES_NO = {
    "yes": r"re:^(?!.*\bno\b).*\byes\b",
    "no": r"re:^(?!.*\byes\b).*\bno\b",
}


def _doc_id(index: int) -> str:
    return f"MHR-{index + 1:03d}"


def _document(article: dict) -> str:
    author = article.get("author")
    lines = [
        f"# {article['title']}",
        "",
        f"Source: {article['source']}",
        *([f"Author: {author}"] if author and author != "None" else []),
        f"Published: {article['published_at']}",
        f"Category: {article['category']}",
        "",
        article["body"],
    ]
    return "\n".join(lines) + "\n"


def _question(query: dict, id_by_url: dict[str, str], index: int) -> dict:
    question_type = query["question_type"]
    gold = list(dict.fromkeys(id_by_url[e["url"]] for e in query["evidence_list"]))
    metadata = {
        "case_id": f"mhr-{index + 1:04d}",
        "question_type": question_type,
        "category": question_type,
        "gold_doc_ids": gold,
        "source_dataset": f"{REPO}@{REVISION[:7]}",
        "language": "en",
    }
    if question_type == "null_query":
        metadata["expected_behavior"] = "decline"
    else:
        answer = query["answer"].strip()
        pattern = _YES_NO.get(answer.lower().rstrip("."))
        metadata["answer_patterns"] = [[pattern or answer]]
    return {"question": query["query"], "ground_truth": query["answer"], "context": "", "metadata": metadata}


def build(corpus: list[dict], queries: list[dict], seed: int | None = SEED) -> tuple[dict[str, str], list[dict]]:
    """Return (documents by doc id, question rows); ``seed=None`` keeps the source order."""
    documents = {_doc_id(i): _document(article) for i, article in enumerate(corpus)}
    id_by_url = {article["url"]: _doc_id(i) for i, article in enumerate(corpus)}
    questions = [_question(query, id_by_url, i) for i, query in enumerate(queries)]
    if seed is not None:
        random.Random(seed).shuffle(questions)
    return documents, questions


def write(corpus: list[dict], queries: list[dict], out_dir: Path, seed: int | None = SEED) -> None:
    documents, questions = build(corpus, queries, seed)
    corpus_dir = Path(out_dir) / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for doc_id, text in documents.items():
        (corpus_dir / f"{doc_id}.md").write_text(text, encoding="utf-8")
    with open(Path(out_dir) / "questions.jsonl", "w", encoding="utf-8") as fh:
        for row in questions:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _download(filename: str) -> list[dict]:
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{filename}"
    with urllib.request.urlopen(url, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MultiHop-RAG into a jsonl-shared dataset.")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    corpus, queries = _download("corpus.json"), _download("MultiHopRAG.json")
    write(corpus, queries, args.out)
    print(f"Wrote {len(corpus)} documents and {len(queries)} questions to {args.out}")


if __name__ == "__main__":
    main()
