"""Gemeinsame Datenstrukturen für den Benchmark."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelDef:
    """Definition eines zu testenden Modells."""
    provider: str          # "anthropic" | "openai" | "openai_compatible"
    model_id: str          # z.B. "claude-haiku-4-5" oder "llama-3.1-8b"
    label: str             # Anzeigename in Berichten
    base_url: Optional[str] = None   # für eigene Server (openai_compatible)
    api_key: Optional[str] = None    # überschreibt globalen Key
    timeout_s: Optional[float] = None  # überschreibt globales Timeout
    tags: Optional[dict] = None      # freie Metadaten (hardware, params_active, …)


@dataclass
class CallResult:
    """Ergebnis eines einzelnen API-Calls."""
    model: str
    provider: str
    prompt_idx: int
    mode: str              # "sequential" | "parallel"
    call_idx: int
    latency_s: float
    success: bool
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    error: Optional[str] = None
    output_text: Optional[str] = None
    # Streaming-Metriken (None wenn nicht streaming oder fehlgeschlagen)
    ttft_s: Optional[float] = None         # Time to First Token
    tpot_s: Optional[float] = None         # Time Per Output Token (Mittel über alle Inter-Token-Zeiten)
    itl_p50_s: Optional[float] = None      # Inter-Token-Latenz Median
    itl_p95_s: Optional[float] = None      # Inter-Token-Latenz P95
    gen_tok_s: Optional[float] = None      # Output-Tokens / (latency - ttft)


@dataclass
class ModelStats:
    """Aggregierte Statistiken für ein Modell+Modus-Paar."""
    model: str
    mode: str
    n: int                 # Gesamtzahl Calls
    mean_s: float
    median_s: float
    p95_s: float
    stdev_s: float
    success_rate: float
    tps: Optional[float]   # Output-Tokens pro Sekunde (None wenn unbekannt)
    wall_s: float = 0.0    # Gesamtlaufzeit des Blocks (Wall-Clock)
    # Streaming-Metriken (Mittelwerte über alle Calls dieses Modell+Modus)
    ttft_mean_s: Optional[float] = None    # Mittlere Time to First Token
    ttft_p95_s: Optional[float] = None     # P95 Time to First Token
    tpot_mean_s: Optional[float] = None    # Mittlere Time Per Output Token
    gen_tps_mean: Optional[float] = None   # Mittlere reine Generierungs-Tokens/s
