import json, os, sys, time, logging
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), ".env"))
from benchmark import providers
from benchmark.evaluation import evaluate_results

calls = []
T0 = time.time()
orig = providers._TokenCountingChatModel._agenerate
async def timed(self, messages, stop=None, run_manager=None, **kw):
    t = time.time()
    r = await orig(self, messages, stop=stop, run_manager=run_manager, **kw)
    for g in r.generations:
        m = g.message; md = getattr(m, "response_metadata", {}) or {}
        um = getattr(m, "usage_metadata", None) or {}
        calls.append(dict(t0=round(t-T0,1), dt=round(time.time()-t,1), out=um.get("output_tokens"),
                          fin=md.get("finish_reason")))
    return r
providers._TokenCountingChatModel._agenerate = timed

recs = json.load(open(sys.argv[1]))
sel = [r for r in recs.values() if r["index"] in map(int, sys.argv[2].split(","))]
gts = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else None
t0 = time.time()
res = evaluate_results(
    [r["question"] for r in sel],
    [gts[str(r["index"])] if gts else r["gen"]["answer"] for r in sel],
    [r["gen"]["answer"] for r in sel],
    [r["contexts"] for r in sel],
    critic_llm_model=os.environ["EVAL_CRITIC_LLM"],
    critic_embedding_model=os.environ["EVAL_CRITIC_EMBEDDING"],
    critic_openai_compat_base_url=os.environ["EVAL_CRITIC_OPENAI_COMPAT_BASE_URL"],
    critic_openai_compat_api_key=os.environ["EVAL_CRITIC_OPENAI_COMPAT_API_KEY"],
    critic_max_tokens=int(os.environ["EVAL_CRITIC_MAX_TOKENS"]),
)
print("SUMCALLS", round(sum(c['dt'] for c in calls),1)); print("TOTAL", round(time.time()-t0,1), "s", res.error, res.metric_means)
for c in calls: print(c)
