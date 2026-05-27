"""Lädt die Konfiguration aus der .env-Datei."""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .types import ModelDef

_ROOT = Path(__file__).parent.parent


def load_config() -> dict:
    """
    Liest .env aus dem Projektverzeichnis und gibt ein Config-Dict zurück.
    Bricht mit einer klaren Fehlermeldung ab, wenn Pflichtfelder fehlen.
    """
    env_path = _ROOT / ".env"
    if not env_path.exists():
        sys.exit(
            f"[Fehler] Keine .env gefunden unter {env_path}\n"
            f"         Kopiere .env.example → .env und passe die Werte an."
        )
    load_dotenv(env_path)

    # ── Modelle ──────────────────────────────────────────────────────────────
    models_raw = os.getenv("MODELS", "").strip()
    if not models_raw:
        sys.exit("[Fehler] MODELS ist in der .env nicht gesetzt.")

    try:
        models_data = json.loads(models_raw)
    except json.JSONDecodeError as exc:
        sys.exit(f"[Fehler] MODELS ist kein gültiges JSON: {exc}")

    models: list[ModelDef] = []
    allowed_keys = {f for f in ModelDef.__dataclass_fields__}
    for i, m in enumerate(models_data):
        required = {"provider", "model_id", "label"}
        missing = required - m.keys()
        if missing:
            sys.exit(f"[Fehler] Modell #{i} fehlen Pflichtfelder: {missing}")
        # Unbekannte Keys ignorieren statt zu crashen
        clean = {k: v for k, v in m.items() if k in allowed_keys}
        models.append(ModelDef(**clean))

    # ── Prompts ──────────────────────────────────────────────────────────────
    prompts_file = os.getenv("PROMPTS_FILE", "").strip()
    if prompts_file:
        prompts_path = Path(prompts_file)
        if not prompts_path.is_absolute():
            prompts_path = _ROOT / prompts_path
        if not prompts_path.exists():
            sys.exit(f"[Fehler] PROMPTS_FILE nicht gefunden: {prompts_path}")
        try:
            prompts: list[str] = json.loads(prompts_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            sys.exit(f"[Fehler] PROMPTS_FILE konnte nicht gelesen werden: {exc}")
        print(f"  Prompts geladen aus: {prompts_path}  ({len(prompts)} Einträge)")
    else:
        prompts_raw = os.getenv("PROMPTS", '["Hello, who are you?"]').strip()
        try:
            prompts = json.loads(prompts_raw)
        except json.JSONDecodeError as exc:
            sys.exit(f"[Fehler] PROMPTS ist kein gültiges JSON: {exc}")

    # ── Skalare Parameter ────────────────────────────────────────────────────
    def _int(key: str, default: int) -> int:
        try:
            return int(os.getenv(key, str(default)))
        except ValueError:
            sys.exit(f"[Fehler] {key} muss eine ganze Zahl sein.")

    def _float(key: str, default: float) -> float:
        try:
            return float(os.getenv(key, str(default)))
        except ValueError:
            sys.exit(f"[Fehler] {key} muss eine Zahl sein.")

    # ── CALL_COUNTS ──────────────────────────────────────────────────────────
    call_counts_raw = os.getenv("CALL_COUNTS", "").strip()
    if call_counts_raw:
        try:
            call_counts: list[int] = [int(x) for x in json.loads(call_counts_raw)]
        except (json.JSONDecodeError, ValueError) as exc:
            sys.exit(f"[Fehler] CALL_COUNTS ist kein gültiges JSON-Array mit Zahlen: {exc}")
    else:
        # Fallback: SEQUENTIAL_CALLS / PARALLEL_CALLS als einzelne Werte
        call_counts = [_int("SEQUENTIAL_CALLS", 3)]

    # ── MAX_TOKENS_LIST ───────────────────────────────────────────────────────
    max_tokens_raw = os.getenv("MAX_TOKENS_LIST", "").strip()
    if max_tokens_raw:
        try:
            max_tokens_list: list[int] = [int(x) for x in json.loads(max_tokens_raw)]
        except (json.JSONDecodeError, ValueError) as exc:
            sys.exit(f"[Fehler] MAX_TOKENS_LIST ist kein gültiges JSON-Array mit Zahlen: {exc}")
    else:
        max_tokens_list = [_int("MAX_TOKENS", 256)]

    # ── PROMPT_SELECTION ──────────────────────────────────────────────────────
    selection = os.getenv("PROMPT_SELECTION", "first").strip().lower()
    if selection not in ("first", "random"):
        sys.exit("[Fehler] PROMPT_SELECTION muss 'first' oder 'random' sein.")

    # Warnung wenn Pool zu klein
    max_count = max(call_counts) if call_counts else 0
    if max_count > len(prompts):
        print(
            f"[Warnung] CALL_COUNTS enthält {max_count}, aber nur {len(prompts)} "
            f"Prompts vorhanden. Größere Werte werden auf {len(prompts)} begrenzt."
        )

    # ── Warmup ────────────────────────────────────────────────────────────────
    warmup_env = os.getenv("WARMUP_ENABLED", "true").strip().lower()
    warmup_enabled = warmup_env not in ("false", "0", "no", "off")

    return {
        "models":               models,
        "prompts":              prompts,          # vollständiger Pool
        "call_counts":          call_counts,
        "max_tokens_list":      max_tokens_list,
        "prompt_selection":     selection,        # "first" | "random"
        "timeout_s":            _float("TIMEOUT_S", 30.0),
        "warmup_enabled":       warmup_enabled,
        "warmup_max_tokens":    _int("WARMUP_MAX_TOKENS", 16),
        "mlflow_experiment":    os.getenv("MLFLOW_EXPERIMENT", "llm-benchmark"),
        "mlflow_tracking_uri":  os.getenv("MLFLOW_TRACKING_URI", "").strip(),
        "anthropic_api_key":    os.getenv("ANTHROPIC_API_KEY", ""),
        "openai_api_key":       os.getenv("OPENAI_API_KEY", ""),
    }
