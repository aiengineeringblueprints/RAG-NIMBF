"""
Benchmark-Läufe: sequentiell und parallel.
"""

import asyncio
import time

from .clients import make_call
from .types import CallResult, ModelDef


def _to_result(raw: dict, model_def: ModelDef,
               p_idx: int, c_idx: int, mode: str) -> CallResult:
    if isinstance(raw, Exception):
        raw = {"latency": 0.0, "success": False, "error": str(raw)}
    return CallResult(
        model=model_def.label,
        provider=model_def.provider,
        prompt_idx=p_idx,
        mode=mode,
        call_idx=c_idx,
        latency_s=raw["latency"],
        success=raw["success"],
        tokens_in=raw.get("tokens_in"),
        tokens_out=raw.get("tokens_out"),
        error=raw.get("error"),
        output_text=raw.get("output_text"),
        ttft_s=raw.get("ttft_s"),
        tpot_s=raw.get("tpot_s"),
        itl_p50_s=raw.get("itl_p50_s"),
        itl_p95_s=raw.get("itl_p95_s"),
        gen_tok_s=raw.get("gen_tok_s"),
    )


def _snippet(text: str | None, width: int = 100) -> str:
    if not text:
        return ""
    return text.replace("\n", " ")[:width] + ("…" if len(text) > width else "")


# ── Warmup ────────────────────────────────────────────────────────────────────

def warmup_model(model_def: ModelDef, prompt: str, config: dict) -> None:
    """Sendet einen einzelnen 'Wegwerf-Call' an das Modell, damit es geladen
    und alle Kernels kompiliert werden. Das Ergebnis wird NICHT gemessen.

    Dies eliminiert die Verzerrung des ersten Messwerts durch Modell-
    Loading-Overhead (besonders bei Ollama, wo Modelle bei Bedarf aus dem
    Speicher geladen werden).

    Über die .env kann der Warmup mit `WARMUP_ENABLED=false` deaktiviert
    werden (z.B. um explizit den Cold-Start zu messen).
    """
    if not config.get("warmup_enabled", True):
        return

    warmup_tokens = int(config.get("warmup_max_tokens", 16))
    print(f"\n[WARMUP] {model_def.label} — lade Modell in den Speicher …",
          flush=True)
    t0 = time.perf_counter()
    try:
        raw = make_call(model_def, prompt, warmup_tokens, config)
        elapsed = time.perf_counter() - t0
        if raw.get("success"):
            print(f"  → Warmup OK in {elapsed:.1f}s "
                  f"(Modell ist jetzt geladen, Ergebnis wird verworfen)")
        else:
            print(f"  → Warmup fehlgeschlagen in {elapsed:.1f}s: "
                  f"{raw.get('error', '')[:80]}")
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"  → Warmup Exception nach {elapsed:.1f}s: {str(exc)[:80]}")


# ── Sequentieller Lauf ────────────────────────────────────────────────────────

def run_sequential(model_def: ModelDef, prompts: list[str],
                   n_calls: int, max_tokens: int, config: dict,
                   ) -> tuple[list[CallResult], float]:
    """Führt n_calls × len(prompts) Calls streng nacheinander aus."""
    print(f"\n[SEQ] {model_def.label}  ({model_def.provider})")
    results: list[CallResult] = []
    t0 = time.perf_counter()

    for p_idx, prompt in enumerate(prompts):
        print(f"  Prompt {p_idx + 1}/{len(prompts)}: \"{_snippet(prompt, 60)}\"")
        for c_idx in range(n_calls):
            raw = make_call(model_def, prompt, max_tokens, config)
            if raw["success"]:
                snippet = _snippet(raw.get("output_text"), 100)
                print(f"    Call {c_idx + 1}/{n_calls}: ✓  {raw['latency']:.3f}s"
                      + (f"  │  {snippet}" if snippet else ""))
            else:
                print(f"    Call {c_idx + 1}/{n_calls}: ✗  {raw.get('error', '')[:80]}")
            results.append(_to_result(raw, model_def, p_idx, c_idx, "sequential"))

    return results, time.perf_counter() - t0


# ── Paralleler Lauf ───────────────────────────────────────────────────────────

async def _async_call(model_def: ModelDef, prompt: str,
                      max_tokens: int, config: dict) -> dict:
    """Wrapper, der make_call() im Thread-Pool ausführt (non-blocking)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, make_call, model_def, prompt, max_tokens, config
    )


async def run_parallel(model_def: ModelDef, prompts: list[str],
                       n_calls: int, max_tokens: int, config: dict,
                       ) -> tuple[list[CallResult], float]:
    """Feuert n_calls × len(prompts) Calls gleichzeitig ab."""
    print(f"\n[PAR] {model_def.label}  ({model_def.provider})")

    tasks = []
    meta: list[tuple[int, int]] = []
    for p_idx, prompt in enumerate(prompts):
        for c_idx in range(n_calls):
            tasks.append(_async_call(model_def, prompt, max_tokens, config))
            meta.append((p_idx, c_idx))

    t_wall = time.perf_counter()
    raws = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.perf_counter() - t_wall

    n_ok = sum(1 for r in raws if isinstance(r, dict) and r.get("success"))
    print(f"  {len(tasks)} Calls abgeschlossen in {elapsed:.2f}s Wallzeit  "
          f"({n_ok}/{len(tasks)} erfolgreich)\n")

    # Ergebnisse nach Prompts gruppiert ausgeben
    results = []
    prompt_groups: dict[int, list] = {}
    for (p_idx, c_idx), raw in zip(meta, raws):
        prompt_groups.setdefault(p_idx, []).append((c_idx, raw))

    for p_idx in sorted(prompt_groups):
        print(f"  Prompt {p_idx + 1}/{len(prompts)}: \"{_snippet(prompts[p_idx], 60)}\"")
        for c_idx, raw in sorted(prompt_groups[p_idx]):
            if isinstance(raw, Exception) or not (isinstance(raw, dict) and raw.get("success")):
                err = str(raw) if isinstance(raw, Exception) else raw.get("error", "")
                print(f"    Call {c_idx + 1}: ✗  {err[:80]}")
            else:
                snippet = _snippet(raw.get("output_text"), 100)
                print(f"    Call {c_idx + 1}: ✓  {raw['latency']:.3f}s"
                      + (f"  │  {snippet}" if snippet else ""))
            results.append(_to_result(raw, model_def, p_idx, c_idx, "parallel"))

    return results, elapsed
