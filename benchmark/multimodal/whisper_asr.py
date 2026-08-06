"""Whisper ASR ingestion path.

Takes audio files (mp3/wav/m4a/flac/...) and returns one transcript chunk per
file in the framework's ``{context, metadata}`` corpus format. Output is
consumed downstream by the existing text chunk -> embed -> retrieve pipeline.

Two Whisper backends are supported:

- ``faster-whisper`` (default; recommended — CTranslate2, lower memory)
- ``openai-whisper`` (fallback)

Both are heavy deps (pull torch). They are imported lazily; missing deps raise
:class:`OptionalDependencyError`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from benchmark.multimodal.registry import OptionalDependencyError

logger = logging.getLogger(__name__)

_FEATURE = "audio_whisper"
_FASTER_WHISPER_PACKAGE = "faster-whisper"
_OPENAI_WHISPER_PACKAGE = "openai-whisper"

_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".webm", ".aac"}


def _require_faster_whisper() -> Any:
    try:
        from faster_whisper import WhisperModel  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE,
            package=_FASTER_WHISPER_PACKAGE,
            extra="faster-whisper",
        ) from exc
    return WhisperModel


def _require_openai_whisper() -> Any:
    try:
        import whisper  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OptionalDependencyError(
            feature=_FEATURE,
            package=_OPENAI_WHISPER_PACKAGE,
            extra="openai-whisper",
        ) from exc
    return whisper


def _transcribe_faster_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> str:
    WhisperModel = _require_faster_whisper()
    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=5,
    )
    # faster-whisper iterates lazily — force evaluation.
    return " ".join(segment.text.strip() for segment in segments).strip()


def _transcribe_openai_whisper(
    audio_path: Path,
    model_name: str,
    device: str,
    language: str | None,
) -> str:
    whisper = _require_openai_whisper()
    model = whisper.load_model(model_name, device=device)
    result = model.transcribe(str(audio_path), language=language)
    return str(result.get("text", "")).strip()


def ingest_audio_via_whisper(
    corpus_path: str | Path,
    *,
    model_name: str = "base",
    backend: str = "faster-whisper",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    **_: Any,
) -> list[dict]:
    """Walk ``corpus_path`` for audio files, return per-file transcript dicts.

    Parameters
    ----------
    corpus_path:
        Directory containing audio files (searched recursively).
    model_name:
        Whisper model size — ``tiny``, ``base``, ``small``, ``medium``,
        ``large-v3`` etc. Default ``base`` keeps the prototype installable on
        CPU; switch to ``large-v3`` for production quality.
    backend:
        ``"faster-whisper"`` (default) or ``"openai-whisper"``.
    device / compute_type:
        Forwarded to faster-whisper. Defaults (``cpu`` / ``int8``) work without
        a GPU. Use ``cuda`` / ``float16`` for GPU inference.
    language:
        Optional ISO-639-1 language code (e.g. ``"en"``). ``None`` lets
        Whisper auto-detect.
    """
    root = Path(corpus_path).resolve()
    if not root.is_dir():
        raise ValueError(f"corpus_path is not a directory: {root}")

    audio_paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _AUDIO_EXTS
    )
    if not audio_paths:
        raise ValueError(f"No audio documents found in {root}")

    corpus: list[dict] = []
    for audio_path in audio_paths:
        logger.info("Whisper: transcribing %s", audio_path.relative_to(root))
        if backend == "faster-whisper":
            transcript = _transcribe_faster_whisper(
                audio_path, model_name, device, compute_type, language
            )
        elif backend == "openai-whisper":
            transcript = _transcribe_openai_whisper(
                audio_path, model_name, device, language
            )
        else:
            raise ValueError(
                f"Unknown Whisper backend: {backend!r}. "
                "Use 'faster-whisper' or 'openai-whisper'."
            )

        if not transcript:
            logger.warning("Whisper produced empty transcript for %s", audio_path)
            continue

        relative = audio_path.relative_to(root)
        source_id = audio_path.stem
        corpus.append(
            {
                "context": transcript,
                "metadata": {
                    "doc_id": source_id,
                    "source_id": source_id,
                    "source_name": audio_path.name,
                    "source_path": str(relative),
                    "corpus_type": "audio",
                    "whisper_backend": backend,
                    "whisper_model": model_name,
                },
            }
        )
    logger.info(
        "Whisper: produced %d transcript chunks from %d audio files",
        len(corpus),
        len(audio_paths),
    )
    return corpus
