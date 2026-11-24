"""Enterprise RAG Blueprint 2x2 ablation analysis (run8, issue #22).

Reads the per-config ``*.samples.json.gz`` of a run directory whose configs
differ in query refinement (managed-adapter hash) and hybrid alpha, and prints
refusal rates, answer latency and exact paired McNemar tests.

Usage (repo root): python scripts/analyze_blueprint_ablation.py results/Enterprise-RAG/run8
The hash -> refinement mapping below is specific to run8's manifest.
"""
import glob
import gzip
import json
import statistics as st
import sys
from math import comb

sys.path.insert(0, ".")
from benchmark.answer_match import declines  # noqa: E402
from benchmark.custom_metrics import is_refusal_answer  # noqa: E402

RUN = sys.argv[1] if len(sys.argv) > 1 else "results/Enterprise-RAG/run8"
LABEL = {"a6b2f4cd80": "refine_on", "2c9672c978": "refine_off"}


def load():
    out = {}
    for f in sorted(glob.glob(f"{RUN}/configs/*.samples.json.gz")):
        h = f.split("managed-")[1][:10]
        a = f.split("hybrid-a")[1].split("_")[0]
        out[f"{LABEL[h]}_a{a}"] = json.load(gzip.open(f))["results"][0]
    return out


def p95(xs):
    xs = sorted(xs)
    return xs[int(0.95 * (len(xs) - 1))]


def mcnemar(a, b):
    """Exact two-sided McNemar on paired booleans; returns (b01, b10, p)."""
    b01 = sum(1 for x, y in zip(a, b) if not x and y)
    b10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return b01, b10, p


data = load()
order = ["refine_on_a0.5", "refine_on_a0.3", "refine_off_a0.5", "refine_off_a0.3"]
qs = [s["question"] for s in data[order[0]]]
for k in order:
    assert [s["question"] for s in data[k]] == qs, f"question order differs: {k}"

rows = {}
print("config | n | declines | is_refusal | refused_with_gold@10 | nlg_zero | t_mean | t_median | t_p95")
for k in order:
    S = data[k]
    dec = [declines(s["answer"] or "") for s in S]
    ref = [is_refusal_answer(s["answer"] or "") for s in S]
    gold10 = [s["custom_scores"].get("hit@10", 0) >= 1 for s in S]
    rwg = sum(1 for d, g in zip(dec, gold10) if d and g)
    nlg0 = sum(1 for s in S if s["custom_scores"].get("bert_score_f1", 0) == 0)
    t = [s["total_seconds"] for s in S if s.get("total_seconds") is not None]
    rows[k] = dict(dec=dec, hit1=[s["custom_scores"].get("hit@1", 0) >= 1 for s in S], hit10=gold10)
    print(f"{k} | {len(S)} | {sum(dec)/len(S):.3f} ({sum(dec)}) | {sum(ref)/len(S):.3f} | {rwg} "
          f"| {nlg0} | {st.mean(t):.2f} | {st.median(t):.2f} | {p95(t):.2f}")

print("\nPaired exact McNemar vs baseline refine_on_a0.5 (b01 = only other correct, b10 = only baseline correct)")
base = rows["refine_on_a0.5"]
for k in order[1:]:
    for m in ("hit1", "hit10", "dec"):
        b01, b10, p = mcnemar(base[m], rows[k][m])
        print(f"{k} {m}: b01={b01} b10={b10} p={p:.4f}")
print("\nRefinement effect at fixed alpha")
for a in ("0.5", "0.3"):
    on, off = rows[f"refine_on_a{a}"], rows[f"refine_off_a{a}"]
    for m in ("hit1", "hit10", "dec"):
        b01, b10, p = mcnemar(on[m], off[m])
        print(f"alpha {a} {m}: off-only={b01} on-only={b10} p={p:.4f}")
