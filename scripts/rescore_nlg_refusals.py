"""Re-score NLG metrics of a finished run with the fixed refusal check (issue #57).

Before #57, ``is_refusal_answer()`` matched broad substrings ("insufficient",
"provided context") and set ROUGE-L, BLEU, METEOR and BERTScore of genuine
answers to 0. This script reads every ``configs/*.samples.json.gz`` of a run
directory and:

* recomputes the NLG metrics of samples the old check zeroed (stored
  ``bert_score_f1 == 0``) that the new check no longer counts as refusals,
* sets the NLG metrics of samples the new check counts as refusals to 0,
* keeps all other stored scores.

The run's files are not modified (``results/`` is append-only); the old and new
per-config means are written to ``<run_dir>/nlg_rescore_issue57.json``.

Usage (repo root, on the DGX Spark):
    python scripts/rescore_nlg_refusals.py results/Enterprise-RAG/run8 [--bert-model roberta-large]
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from statistics import mean

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark.custom_metrics import (  # noqa: E402
    bert_score,
    bleu,
    is_refusal_answer,
    meteor_score,
    rouge_l,
)

NLG_METRICS = ("rouge_l", "bleu", "meteor",
               "bert_score_precision", "bert_score_recall", "bert_score_f1")


def rescore(samples: list[dict], bert_model) -> tuple[list[dict[str, float]], int, int]:
    """Return per-sample NLG scores plus the counts of rescored and newly zeroed samples."""
    out, rescored, zeroed = [], 0, 0
    for s in samples:
        stored = s.get("custom_scores") or {}
        old = {m: stored.get(m) for m in NLG_METRICS}
        answer, truth = s.get("answer") or "", s.get("ground_truth") or ""
        if is_refusal_answer(answer):
            zeroed += old["bert_score_f1"] not in (0, 0.0)
            out.append({m: 0.0 for m in NLG_METRICS})
        elif old["bert_score_f1"] == 0.0:
            rescored += 1
            p, r, f1 = bert_score(answer, truth, bert_model)
            out.append({"rouge_l": rouge_l(answer, truth), "bleu": bleu(answer, truth),
                        "meteor": meteor_score(answer, truth), "bert_score_precision": p,
                        "bert_score_recall": r, "bert_score_f1": f1})
        else:
            out.append(old)
    return out, rescored, zeroed


def _means(rows: list[dict]) -> dict[str, float]:
    return {m: mean(r[m] for r in rows) for m in NLG_METRICS
            if all(r.get(m) is not None for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dir")
    parser.add_argument("--bert-model", default="roberta-large")
    args = parser.parse_args()

    from sentence_transformers import SentenceTransformer
    bert_model = SentenceTransformer(args.bert_model, device="cpu")

    report = {"issue": 57, "bert_model": args.bert_model, "configs": {}}
    for path in sorted(glob.glob(os.path.join(args.run_dir, "configs", "*.samples.json.gz"))):
        with gzip.open(path, "rt") as f:
            samples = json.load(f)["results"][0]
        new, rescored, zeroed = rescore(samples, bert_model)
        old = [{m: (s.get("custom_scores") or {}).get(m) for m in NLG_METRICS} for s in samples]
        name = os.path.basename(path).removesuffix(".samples.json.gz")
        report["configs"][name] = {
            "n": len(samples), "rescored": rescored, "newly_zeroed": zeroed,
            "refusals": sum(is_refusal_answer(s.get("answer") or "") for s in samples),
            "old_means": _means(old), "new_means": _means(new),
        }
        o, n = report["configs"][name]["old_means"], report["configs"][name]["new_means"]
        print(f"{name}\n  rescored={rescored} newly_zeroed={zeroed} "
              f"bert_f1 {o.get('bert_score_f1', float('nan')):.4f} -> {n.get('bert_score_f1', float('nan')):.4f} "
              f"rouge_l {o.get('rouge_l', float('nan')):.4f} -> {n.get('rouge_l', float('nan')):.4f}")

    out_path = os.path.join(args.run_dir, "nlg_rescore_issue57.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
