"""Dynamic ground-truth generation for RAG update operations.

Implements RAGPerf §3.2: given a target document chunk, mask a noun phrase
or numeric value, propose a replacement with DistilBERT, then generate a
question whose answer is the replacement using T5.  The resulting
``(updated_chunk, question, answer)`` triples feed the workload generator's
question pool so update operations can be verified end-to-end.

Submodules
----------
- :mod:`masker`                    – selects mask target inside a chunk
- :mod:`distilbert_generator`      – DistilBERT fill-mask proposals
- :mod:`t5_question_generator`     – T5 highlight-based question generation
- :mod:`synthesizer`               – top-level orchestrator
- :mod:`batch`                     – JSONL CLI / programmatic API

Heavy ML dependencies (``transformers``, ``torch``, ``spacy``) are imported
lazily so importing this package never triggers model downloads.
"""

from __future__ import annotations

from .synthesizer import (
    DynamicGroundTruthSynthesizer,
    UpdateWithGroundTruth,
    SynthesisSkipped,
)

__all__ = [
    "DynamicGroundTruthSynthesizer",
    "UpdateWithGroundTruth",
    "SynthesisSkipped",
]
