"""
API-Clients für alle unterstützten Provider.

Jede Funktion gibt ein einheitliches Dict zurück:
  success=True  → {latency, success, tokens_in, tokens_out, output_text}
  success=False → {latency, success, error}
"""

import json
import time
from statistics import median
from typing import Any

from .types import ModelDef


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _percentile(values: list[float], p: float) -> float | None:
    """Einfache Quantilberechnung ohne numpy."""
    if not values:
        return None
    sv = sorted(values)
    idx = min(int(len(sv) * p), len(sv) - 1)
    return sv[idx]


def _stream_metrics(
    inter_token_times: list[float],
    t0: float,
    t_first_token: float | None,
    t_end: float,
    tokens_out: int | None,
) -> dict:
    """Berechnet TTFT/TPOT/ITL/gen_tps aus den gemessenen Zeitpunkten."""
    ttft = (t_first_token - t0) if t_first_token is not None else None
    gen_time = (t_end - t_first_token) if t_first_token is not None else None
    tpot = (sum(inter_token_times) / len(inter_token_times)
            if inter_token_times else None)
    itl_p50 = median(inter_token_times) if inter_token_times else None
    itl_p95 = _percentile(inter_token_times, 0.95)
    # gen_tps: Tokens pro Sekunde reine Generierung (nach TTFT).
    # Berechnung: tokens_out / latency_total (NICHT / gen_time!)
    # Grund: Bei Thinking-Modellen ist TTFT irreführend hoch, weil das
    # Modell intern schon Tokens generiert (Thinking-Phase) bevor der
    # erste Streaming-Chunk gesendet wird. Wenn wir tokens_out/gen_time
    # nehmen, bekommen wir unrealistische Werte (z.B. 2000+ tok/s).
    # Stattdessen nutzen wir die Gesamt-Latenz, was der realen User-
    # Experience entspricht: "Wie viele Tokens bekomme ich pro Sekunde
    # insgesamt?" Das ist identisch mit tok/s (Gesamt-Throughput) und
    # wird nicht mehr separat als gen_tok_s geführt.
    total_latency = t_end - t0
    gen_tps = (tokens_out / total_latency
               if (tokens_out and total_latency > 0) else None)
    return {
        "ttft_s": ttft,
        "tpot_s": tpot,
        "itl_p50_s": itl_p50,
        "itl_p95_s": itl_p95,
        "gen_tok_s": gen_tps,
    }


def _ok(latency: float, tokens_in: int | None, tokens_out: int | None,
        output_text: str | None = None,
        stream_metrics: dict | None = None) -> dict:
    base = {"latency": latency, "success": True,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "output_text": output_text}
    if stream_metrics:
        base.update(stream_metrics)
    return base


def _err(latency: float, exc: Exception) -> dict:
    return {"latency": latency, "success": False, "error": str(exc)}


def _log_trace(model_id: str, prompt: str, max_tokens: int,
               result: dict) -> None:
    """Loggt einen LLM-Call als MLflow Trace Span (erscheint im Traces Tab)."""
    try:
        import mlflow
        with mlflow.start_span(name=model_id, span_type="LLM") as span:
            span.set_inputs({
                "messages":   [{"role": "user", "content": prompt}],
                "model":      model_id,
                "max_tokens": max_tokens,
            })
            output_text = result.get("output_text", "")
            span.set_outputs({
                "message": {"role": "assistant",
                            "content": output_text[:500] if output_text else ""},
                "usage": {
                    "prompt_tokens":     result.get("tokens_in"),
                    "completion_tokens": result.get("tokens_out"),
                },
                "latency_s":  result.get("latency"),
                "ttft_ms":    round(result["ttft_s"] * 1000, 1) if result.get("ttft_s") else None,
                "tpot_ms":    round(result["tpot_s"] * 1000, 2) if result.get("tpot_s") else None,
            })
    except Exception:
        pass  # Tracing ist optional


# ── Anthropic ─────────────────────────────────────────────────────────────────

def _call_anthropic(model_def: ModelDef, prompt: str, max_tokens: int,
                    global_key: str, timeout_s: float) -> dict:
    try:
        import anthropic
    except ImportError:
        return _err(0.0, ImportError("pip install anthropic"))

    key = model_def.api_key or global_key
    if not key:
        return _err(0.0, ValueError("ANTHROPIC_API_KEY nicht gesetzt"))

    client = anthropic.Anthropic(api_key=key, timeout=timeout_s)
    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model_def.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        return _ok(time.perf_counter() - t0,
                   resp.usage.input_tokens, resp.usage.output_tokens, text)
    except Exception as exc:
        return _err(time.perf_counter() - t0, exc)


# ── OpenAI / OpenAI-kompatibler Server (Streaming) ───────────────────────────

def _call_openai_compatible(model_def: ModelDef, prompt: str, max_tokens: int,
                             global_key: str, timeout_s: float) -> dict:
    try:
        from openai import OpenAI
    except ImportError:
        return _err(0.0, ImportError("pip install openai"))

    key = model_def.api_key or global_key or "none"
    kwargs: dict[str, Any] = {"api_key": key, "timeout": timeout_s}
    if model_def.base_url:
        kwargs["base_url"] = model_def.base_url

    client = OpenAI(**kwargs)
    t0 = time.perf_counter()
    t_first_token: float | None = None
    t_last_token: float | None = None
    inter_token_times: list[float] = []
    try:
        stream = client.chat.completions.create(
            model=model_def.model_id,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            stream_options={"include_usage": True},
        )
        parts: list[str] = []
        tokens_in: int | None = None
        tokens_out: int | None = None
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                else:
                    inter_token_times.append(now - t_last_token)
                t_last_token = now
                parts.append(chunk.choices[0].delta.content)
            if chunk.usage:
                tokens_in  = chunk.usage.prompt_tokens
                tokens_out = chunk.usage.completion_tokens
        t_end = time.perf_counter()
        sm = _stream_metrics(inter_token_times, t0, t_first_token, t_end, tokens_out)
        result = _ok(t_end - t0, tokens_in, tokens_out, "".join(parts), sm)
        _log_trace(model_def.model_id, prompt, max_tokens, result)
        return result
    except Exception as exc:
        return _err(time.perf_counter() - t0, exc)


# ── Native Ollama-API (/api/chat, Streaming) ──────────────────────────────────

def _call_ollama(model_def: ModelDef, prompt: str, max_tokens: int,
                 timeout_s: float) -> dict:
    """
    Ruft die native Ollama-API per Streaming auf (POST /api/chat, stream=true).
    Jede Zeile ist ein JSON-Objekt; die letzte hat done=true mit Token-Counts.
    base_url zeigt auf den Ollama-Stammpfad, z.B. http://host/ollama
    """
    try:
        import requests
    except ImportError:
        return _err(0.0, ImportError("pip install requests"))

    if not model_def.base_url:
        return _err(0.0, ValueError("base_url fehlt für ollama-Provider"))

    base = model_def.base_url.rstrip("/")
    url  = f"{base}/api/chat"

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if model_def.api_key and model_def.api_key != "none":
        headers["Authorization"] = f"Bearer {model_def.api_key}"

    payload = {
        "model":    model_def.model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   True,
        "options":  {"num_predict": max_tokens},
    }

    t0 = time.perf_counter()
    t_first_token: float | None = None
    t_last_token: float | None = None
    inter_token_times: list[float] = []
    try:
        resp = requests.post(url, json=payload, headers=headers,
                             timeout=timeout_s, stream=True)
        resp.raise_for_status()

        parts: list[str] = []
        tokens_in: int | None  = None
        tokens_out: int | None = None
        final_data: dict = {}

        thinking_parts: list[str] = []

        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            data = json.loads(raw_line)
            msg = data.get("message", {})
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            # Thinking-Tokens count as first activity for TTFT
            if thinking or content:
                now = time.perf_counter()
                if t_first_token is None:
                    t_first_token = now
                else:
                    inter_token_times.append(now - t_last_token)
                t_last_token = now

            if thinking:
                thinking_parts.append(thinking)
            if content:
                parts.append(content)

            if data.get("done"):
                tokens_in  = data.get("prompt_eval_count")
                tokens_out = data.get("eval_count")
                final_data = data

        t_end       = time.perf_counter()
        latency     = t_end - t0
        # Preserve thinking content in output for transparency
        thinking_text = "".join(thinking_parts)
        content_text = "".join(parts)
        if thinking_text:
            output_text = f"<think>{thinking_text}</think>{content_text}"
        else:
            output_text = content_text
        sm = _stream_metrics(inter_token_times, t0, t_first_token, t_end, tokens_out)

        result = _ok(latency, tokens_in, tokens_out, output_text, sm)
        _log_trace(model_def.model_id, prompt, max_tokens, result)
        return result
    except Exception as exc:
        return _err(time.perf_counter() - t0, exc)


# ── Dispatcher ────────────────────────────────────────────────────────────────

def make_call(model_def: ModelDef, prompt: str, max_tokens: int,
              config: dict) -> dict:
    """Wählt den richtigen Client anhand von model_def.provider."""
    timeout = model_def.timeout_s or config.get("timeout_s", 30.0)

    if model_def.provider == "anthropic":
        return _call_anthropic(
            model_def, prompt, max_tokens,
            config.get("anthropic_api_key", ""), timeout
        )
    if model_def.provider in ("openai", "openai_compatible"):
        return _call_openai_compatible(
            model_def, prompt, max_tokens,
            config.get("openai_api_key", ""), timeout
        )
    if model_def.provider == "ollama":
        return _call_ollama(model_def, prompt, max_tokens, timeout)

    return _err(0.0, ValueError(f"Unbekannter Provider: '{model_def.provider}'"))
