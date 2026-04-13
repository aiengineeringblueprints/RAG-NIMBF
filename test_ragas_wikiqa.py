"""Quick exploration script for the ragas-wikiqa dataset.

Downloads a few samples from vibrantlabsai/ragas-wikiqa and prints them as
pretty JSON so you can inspect question, context, and answer structure.

Usage:
    python test_ragas_wikiqa.py          # 3 random samples
    python test_ragas_wikiqa.py 10       # 10 random samples
"""

import json
import sys

from datasets import load_dataset
from rich import print_json


def main():
    n_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print(f"Loading vibrantlabsai/ragas-wikiqa (train split) ...")
    ds = load_dataset("vibrantlabsai/ragas-wikiqa", split="train")
    print(f"Total rows: {len(ds)}")
    print(f"Columns:    {ds.column_names}\n")

    sampled = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    for i, row in enumerate(sampled):
        # Build the normalised view (same as the adapter would produce)
        context = row["context"]
        if isinstance(context, list):
            context_joined = "\n\n".join(str(c) for c in context)
        else:
            context_joined = str(context)

        entry = {
            "index": i,
            "question": row["question"],
            "correct_answer": row["correct_answer"],
            "context_chunks": row["context"],  # raw: list of strings
            "context_joined": context_joined,   # joined for RAG input
            "generated_with_rag": row.get("generated_with_rag"),
            "generated_without_rag": row.get("generated_without_rag"),
        }

        print(f"{'='*70}")
        print(f"  SAMPLE {i+1}/{n_samples}")
        print(f"{'='*70}")
        print_json(json.dumps(entry, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
