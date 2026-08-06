"""Multi-task RoBERTa classifier for RAGBench TRACe metrics.

Implements :class:`RobertaTraceClassifier`: a shared ``roberta-base``
encoder feeding four classification heads — one per TRACe metric
(Utilization, Relevance, Adherence, Completeness).

Two of the heads (Utilization, Relevance, Completeness) predict a
quantised version of the [0, 1] score into ``num_bins`` ordinal bins;
this matches the RAGBench paper's framing of the metrics as ordinal and
lets us report F1 / accuracy uniformly across all four heads.  The
Adherence head is binary.

Design notes
------------
* The model accepts the standard ``(input_ids, attention_mask)`` tensors
  produced by a Roberta tokenizer; the input text is formatted upstream
  (see :mod:`benchmark.roberta_evaluator.inference`) as
  ``[CLS] question [SEP] context [SEP] response [SEP]``.
* Heads are independent linear layers; the shared encoder is the only
  feature source.  This is the multi-task setup the RAGBench paper found
  beats single-task heads and LLM judges on RAG eval.
* ``transformers`` and ``torch`` are imported lazily so that this module
  can be imported on machines that don't have the HF stack installed.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Canonical TRACe metric ordering.  All four heads are always present; an
# individual example may mask any subset via the per-head label mask.
TRACE_METRICS: tuple[str, ...] = (
    "utilization",
    "relevance",
    "adherence",
    "completeness",
)

# Number of ordinal bins used to quantise the three [0,1] TRACe scores.
# 5 bins => {0.0, 0.25, 0.5, 0.75, 1.0}-aligned edges, which matches the
# bin granularity used in the RAGBench paper.
DEFAULT_NUM_BINS = 5

# Adherence is a boolean label.
ADHERENCE_NUM_CLASSES = 2

DEFAULT_MODEL_NAME = "roberta-base"


def quantize_score(value: float, num_bins: int = DEFAULT_NUM_BINS) -> int:
    """Map a [0, 1] score to an ordinal bin index in ``[0, num_bins)``.

    Values outside [0, 1] are clamped; NaN raises.  Bin edges are uniform
    so that bin ``i`` covers ``(i/N, (i+1)/N]`` with bin 0 also containing
    exactly 0.0.
    """
    if value != value:  # NaN
        raise ValueError("Cannot quantize NaN score")
    clamped = max(0.0, min(1.0, float(value)))
    if clamped >= 1.0:
        return num_bins - 1
    return int(clamped * num_bins)


def dequantize_bin(bin_idx: int, num_bins: int = DEFAULT_NUM_BINS) -> float:
    """Convert a predicted bin index back to a [0, 1] score (bin midpoint)."""
    return (bin_idx + 0.5) / num_bins


def _require_torch():
    try:
        import torch  # type: ignore[import-not-found]
        return torch
    except ImportError as exc:  # pragma: no cover - exercised via tests' lazy path
        raise ImportError(
            "torch is required for RobertaTraceClassifier. Install with "
            "`pip install torch` (or `torch>=2.0`)."
        ) from exc


def _require_transformers():
    try:
        import transformers  # type: ignore[import-not-found]
        return transformers
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "transformers is required for RobertaTraceClassifier. Install with "
            "`pip install transformers>=4.30`."
        ) from exc


class RobertaTraceClassifier:
    """Multi-task RoBERTa classifier with four TRACe heads.

    The class is a thin factory+wrapper around a HuggingFace
    ``PreTrainedModel``.  It does **not** subclass ``nn.Module`` so that
    importing this module never triggers a torch import; the underlying
    HF model is built on demand via :meth:`build`.

    Parameters
    ----------
    model_name
        HF model id or local path to the encoder backbone.  Defaults to
        ``roberta-base``.
    num_bins
        Number of ordinal bins for the three score heads.  The Adherence
        head always has 2 classes regardless of this value.
    dropout
        Dropout applied to the pooled encoder representation before each
        head's classifier.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_bins: int = DEFAULT_NUM_BINS,
        dropout: float = 0.1,
    ) -> None:
        self.model_name = model_name
        self.num_bins = num_bins
        self.dropout = dropout
        self._torch = None
        self._transformers = None
        self._model: Any = None  # built lazily
        self._config: Any = None

    # ── construction ─────────────────────────────────────────────────
    def build(self) -> Any:
        """Build and return the underlying multi-task HF model.

        Idempotent: subsequent calls return the cached model.
        """
        if self._model is not None:
            return self._model

        torch = _require_torch()
        transformers = _require_transformers()

        # Lazy import keeps torch/transformers optional for the package.
        from transformers import AutoConfig, AutoModel  # type: ignore[import-not-found]

        cfg = AutoConfig.from_pretrained(self.model_name)
        encoder = AutoModel.from_pretrained(self.model_name)

        hidden = cfg.hidden_size

        class _MultiTaskHeads(torch.nn.Module):
            """Four independent heads over a shared encoder."""

            def __init__(self, enc, hidden_size, num_bins, dropout_p) -> None:
                super().__init__()
                self.encoder = enc
                self.dropout = torch.nn.Dropout(dropout_p)
                d = hidden_size
                # Score heads: ordinal bins (>=2)
                self.utilization_head = torch.nn.Linear(d, num_bins)
                self.relevance_head = torch.nn.Linear(d, num_bins)
                self.completeness_head = torch.nn.Linear(d, num_bins)
                # Adherence head: binary
                self.adherence_head = torch.nn.Linear(d, ADHERENCE_NUM_CLASSES)

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                outputs = self.encoder(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                # Use [CLS] (<s>) pooled output.  Roberta doesn't ship a
                # pooling head by default; last_hidden_state[:, 0] is the
                # standard substitute.
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    pooled = outputs.pooler_output
                else:
                    pooled = outputs.last_hidden_state[:, 0, :]
                pooled = self.dropout(pooled)
                return {
                    "utilization": self.utilization_head(pooled),
                    "relevance": self.relevance_head(pooled),
                    "adherence": self.adherence_head(pooled),
                    "completeness": self.completeness_head(pooled),
                }

        self._torch = torch
        self._transformers = transformers
        self._config = cfg
        self._model = _MultiTaskHeads(encoder, hidden, self.num_bins, self.dropout)
        return self._model

    # ── save / load ──────────────────────────────────────────────────
    def save(self, output_dir: str) -> None:
        """Persist the encoder and heads to *output_dir*.

        The encoder + tokenizer are written via the standard HF API; the
        head state-dict is written alongside as ``trace_heads.pt`` and
        the bin layout as ``trace_config.json``.
        """
        import json
        import os

        model = self.build()
        torch = self._torch

        os.makedirs(output_dir, exist_ok=True)
        # Save the shared encoder with HF so it can be re-loaded with
        # AutoModel.from_pretrained(output_dir).
        model.encoder.save_pretrained(output_dir)
        # Heads
        heads_sd = {
            "utilization_head": model.utilization_head.state_dict(),
            "relevance_head": model.relevance_head.state_dict(),
            "adherence_head": model.adherence_head.state_dict(),
            "completeness_head": model.completeness_head.state_dict(),
        }
        torch.save(heads_sd, os.path.join(output_dir, "trace_heads.pt"))
        with open(os.path.join(output_dir, "trace_config.json"), "w") as fh:
            json.dump(
                {
                    "model_name": self.model_name,
                    "num_bins": self.num_bins,
                    "dropout": self.dropout,
                    "metrics": list(TRACE_METRICS),
                },
                fh,
                indent=2,
            )

    @classmethod
    def from_pretrained(cls, model_path: str) -> "RobertaTraceClassifier":
        """Load a classifier saved with :meth:`save`.

        ``model_path`` can be either a local directory or an HF Hub id
        pointing at a directory containing ``trace_config.json`` and
        ``trace_heads.pt`` next to the encoder weights.
        """
        import json
        import os

        from transformers import AutoModel  # type: ignore[import-not-found]

        config_path = os.path.join(model_path, "trace_config.json")
        heads_path = os.path.join(model_path, "trace_heads.pt")
        if not (os.path.exists(config_path) and os.path.exists(heads_path)):
            raise FileNotFoundError(
                f"{model_path} does not look like a trained RobertaTraceClassifier "
                f"checkpoint (missing trace_config.json or trace_heads.pt). Train one "
                f"with `python -m benchmark.roberta_evaluator.train --output_dir <dir>`."
            )

        with open(config_path) as fh:
            cfg_dict = json.load(fh)

        instance = cls(
            model_name=model_path,  # encoder weights live in the same dir
            num_bins=cfg_dict.get("num_bins", DEFAULT_NUM_BINS),
            dropout=cfg_dict.get("dropout", 0.1),
        )
        torch = _require_torch()
        encoder = AutoModel.from_pretrained(model_path)
        hidden = encoder.config.hidden_size

        # Reuse the internal _MultiTaskHeads class via build(); easier than
        # duplicating the class definition here.  After build() we splice
        # in the saved encoder + head weights.
        instance._model = None
        instance.model_name = cfg_dict["model_name"]
        model = instance.build()
        model.encoder = encoder
        heads_sd = torch.load(heads_path, map_location="cpu")
        model.utilization_head.load_state_dict(heads_sd["utilization_head"])
        model.relevance_head.load_state_dict(heads_sd["relevance_head"])
        model.adherence_head.load_state_dict(heads_sd["adherence_head"])
        model.completeness_head.load_state_dict(heads_sd["completeness_head"])
        return instance

    # ── inference helper ────────────────────────────────────────────
    def predict_logits(
        self,
        input_ids,
        attention_mask=None,
    ) -> dict[str, "torch.Tensor"]:
        """Run a forward pass and return per-head raw logits."""
        model = self.build()
        model.eval()
        return model(input_ids=input_ids, attention_mask=attention_mask)
