#!/usr/bin/env python3
"""LLM-judged answer correctness per question category and run variant.

Re-scores the stored answers of finished runs with the critic LLM
(EVAL_CRITIC_LLM and its endpoint from .env, or --critic). Run directories are
only read; verdicts go to the --cache JSONL so an interrupted run resumes.

Usage:
    python scripts/judge_answers.py results/Enterprise-RAG/multihop/run1 \
        --dataset datasets/multihop_rag/questions.jsonl \
        --cache /tmp/multihop-run1-verdicts.jsonl [--csv out.csv] [--workers 8]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.answer_match import make_answer_judge  # noqa: E402
from benchmark.providers import get_chat_model, parse_model_id  # noqa: E402
from benchmark.reporting.answer_judging import VERDICTS, judge_runs  # noqa: E402
from config import get_env_combinations  # noqa: E402

COLUMNS = ["variant", "category", "n", "judged"] + [f"judge_{v}" for v in VERDICTS] + ["answer_match"]


def _critic(model_id: str | None):
    config = get_env_combinations()[0]
    provider, model_name = parse_model_id(model_id or config.eval_critic_llm)
    return make_answer_judge(get_chat_model(
        provider=provider,
        model_name=model_name,
        base_url=config.eval_critic_base_url(provider),
        api_key=config.eval_critic_api_key(provider),
        max_tokens=16,
        temperature=config.eval_critic_temperature,
        thinking_control=config.eval_critic_thinking_control,
    )), config.eval_critic_max_workers


def _markdown(rows: list[dict]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for row in sorted(rows, key=lambda r: (r["category"] == "ALL", r["category"], r["variant"])):
        cells = []
        for column in COLUMNS:
            value = row.get(column)
            cells.append(f"{value:.3f}" if isinstance(value, float) else ("–" if value is None else str(value)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path, help="JSONL verdict cache (created if missing)")
    parser.add_argument("--category-field", default="question_type")
    parser.add_argument("--critic", help="critic model id, e.g. openai:deepseek-v4-flash (default: EVAL_CRITIC_LLM)")
    parser.add_argument("--workers", type=int, help="parallel critic calls (default: EVAL_CRITIC_MAX_WORKERS)")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    judge, default_workers = _critic(args.critic)
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    rows = judge_runs(
        args.run_dirs, args.dataset, judge,
        category_field=args.category_field,
        cache_path=args.cache,
        max_workers=args.workers or default_workers,
    )
    if not rows:
        sys.exit("No per-sample results found in the given run directories.")
    print(_markdown(rows))
    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nCSV written to {args.csv}")


if __name__ == "__main__":
    main()
