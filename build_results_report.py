import json, glob, os
from datetime import datetime

base = "LLM_Performance_Tests-main/benchmark_checkpoints"
out = "LLM_Performance_Tests-main/RESULTS.md"

rows = []
for run in sorted(os.listdir(base)):
    rdir = os.path.join(base, run)
    if not os.path.isdir(rdir):
        continue
    for f in sorted(glob.glob(os.path.join(rdir, "*.json"))):
        d = json.load(open(f))
        seq = [r for r in d["results"] if r["mode"] == "sequential"]
        par = [r for r in d["results"] if r["mode"] == "parallel"]

        def agg(xs, key):
            vs = [x.get(key) for x in xs if x.get(key) is not None]
            return sum(vs) / len(vs) if vs else None

        def ok(xs):
            return f"{sum(1 for x in xs if x['success'])}/{len(xs)}"

        fname = os.path.basename(f).replace(".json", "")
        parts = fname.split("__")
        n, t = parts[0].split("_")
        rows.append({
            "run": run, "N": int(n[1:]), "T": int(t[1:]),
            "model": parts[1].replace("_", " "),
            "seq_lat": agg(seq, "latency_s"),
            "par_lat": agg(par, "latency_s"),
            "seq_tput": agg(seq, "gen_tok_s"),
            "par_tput": agg(par, "gen_tok_s"),
            "seq_ttft": agg(seq, "ttft_s"),
            "par_ttft": agg(par, "ttft_s"),
            "seq_ok": ok(seq), "par_ok": ok(par),
        })


def fmt(v, p=2):
    if v is None:
        return "—"
    return f"{v:.{p}f}"


with open(out, "w") as fh:
    fh.write("# Benchmark Results\n\n")
    fh.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")
    fh.write("Source: `LLM_Performance_Tests-main/benchmark_checkpoints/`\n\n")
    fh.write("Legende: **N** = Anzahl Calls, **T** = max_tokens, "
             "**Lat** = Ø Latenz (s), **TPS** = Ø Tokens/s, "
             "**TTFT** = Ø Time-to-First-Token (s), **OK** = erfolgreiche Calls.\n\n")

    for run in sorted(set(r["run"] for r in rows)):
        fh.write(f"## Run {run}\n\n")
        rr = [r for r in rows if r["run"] == run]
        rr.sort(key=lambda x: (x["model"], x["N"], x["T"]))
        fh.write("| Model | N | T | Seq Lat (s) | Seq TPS | Seq TTFT | Seq OK | "
                 "Par Lat (s) | Par TPS | Par TTFT | Par OK |\n")
        fh.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rr:
            fh.write(
                f"| {r['model']} | {r['N']} | {r['T']} | "
                f"{fmt(r['seq_lat'], 2)} | {fmt(r['seq_tput'], 1)} | "
                f"{fmt(r['seq_ttft'], 3)} | {r['seq_ok']} | "
                f"{fmt(r['par_lat'], 2)} | {fmt(r['par_tput'], 1)} | "
                f"{fmt(r['par_ttft'], 3)} | {r['par_ok']} |\n"
            )
        fh.write("\n")

    fh.write("## Peak Throughput (Tokens/s) per Model\n\n")
    fh.write("| Model | Best Seq TPS | (Run, N, T) | Best Par TPS | (Run, N, T) |\n")
    fh.write("|---|---|---|---|---|\n")
    for m in sorted(set(r["model"] for r in rows)):
        mr = [r for r in rows if r["model"] == m and r["seq_tput"] is not None]
        pr = [r for r in rows if r["model"] == m and r["par_tput"] is not None]
        bs = max(mr, key=lambda x: x["seq_tput"]) if mr else None
        bp = max(pr, key=lambda x: x["par_tput"]) if pr else None
        bss = f"{bs['seq_tput']:.1f}"
        bsi = f"{bs['run']}, N={bs['N']}, T={bs['T']}" if bs else "—"
        bps = f"{bp['par_tput']:.1f}"
        bpi = f"{bp['run']}, N={bp['N']}, T={bp['T']}" if bp else "—"
        if not bs:
            bss = "—"
        if not bp:
            bps = "—"
        fh.write(f"| {m} | {bss} | {bsi} | {bps} | {bpi} |\n")

print("written:", out, os.path.getsize(out), "bytes")
