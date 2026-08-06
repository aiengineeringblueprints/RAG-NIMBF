# Dynamic Ground-Truth Generation (H) — Notes

Implementation of **RAGPerf §3.2 "Dynamic Ground Truth Generation for
Updates"**. Produces `(updated_chunk, question, ground_truth_answer)`
triples that let the workload generator verify update operations
end-to-end.

## Module layout

```
benchmark/dynamic_groundtruth/
├── __init__.py                  public re-exports
├── masker.py                    selects mask target (number > noun)
├── distilbert_generator.py      DistilBERT fill-mask
├── t5_question_generator.py     T5 highlight-based QG
├── synthesizer.py               orchestrator + dataclasses
└── batch.py                     JSONL CLI / programmatic API
```

Tests: `tests/test_dynamic_groundtruth.py`.

## Model choices

| Stage         | Model                            | Why                                       |
| ------------- | -------------------------------- | ----------------------------------------- |
| Mask fill     | `distilbert-base-uncased`        | Small (~268 MB), fast on CPU, BERT-style MLM. Spec-mandated. |
| QG            | `valhalla/t5-base-qg-hl`         | Pre-trained for highlight-based QG; produces single-hop questions whose answer is exactly the highlighted span. |
| Mask target   | spaCy `en_core_web_sm` (optional)| Clean POS tagging (NUM/PROPN/NOUN). Regex fallback when absent. |

All three heavy dependencies (`transformers`, `torch`, `spacy`) are
**lazy-imported**. Importing `benchmark.dynamic_groundtruth` does not
trigger model downloads or even import torch.

## Install requirements

The base repo `requirements.txt` already provides `sentence-transformers`
(which transitively pulls `transformers` + `torch`). To enable this
module end-to-end, additionally install:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

spaCy is **optional** — when absent the masker falls back to deterministic
regex + suffix heuristics. The regex path picks years / percentages /
decimals / thousands-separated numbers first, then a heuristic noun.

## Configuration via environment variables

| Variable                  | Purpose                                                |
| ------------------------- | ------------------------------------------------------ |
| `DYNAMIC_GTM_CACHE_DIR`   | HuggingFace cache directory for model downloads.       |
| `CUDA_VISIBLE_DEVICES`    | Standard CUDA env var; set empty to force CPU.         |
| `DYNAMIC_GTM_RUN_SLOW=1`  | Opt-in for the slow integration test (real models).    |

Forcing CPU from Python: `DynamicGroundTruthSynthesizer(device=-1)`.

## Output schema

JSONL, one record per synthesised update:

```json
{"chunk_id": "doc-007-chunk-3", "original": "...", "updated": "...",
 "masked_token_original": "1995", "masked_token_new": "2003",
 "question": "When was Acme Corporation founded?", "answer": "2003"}
```

## Quality safeguards (spec §3.2)

All implemented in `synthesizer.py`:

1. **Chunk length** — `< 50 chars` skipped with reason `chunk_too_short`.
2. **No mask target** — chunks without nouns/numbers skipped (`no_mask_target`).
3. **DistilBERT confidence** — candidates `< 0.05` filtered out.
4. **Punctuation / stopword filter** — DistilBERT proposals like `,`, `the`, `a` rejected.
5. **Distinct from original** — top candidate equal to the original token is skipped in favour of the next-ranked candidate.
6. **T5 question length** — questions `> 30 words` rejected (`t5_failed`).
7. **T5 question minimum length** — questions `< 3 words` rejected.
8. **Question dedup** — when `dedup_questions=True` (default), duplicate questions in the same batch are dropped with reason `duplicate_question`.

The synthesizer returns a `SynthesisReport` with both `.updates` and
`.skipped` (with reasons), so batch runs have full visibility into
rejection causes.

## Performance characteristics

All measurements taken on CPU (Intel i7-12700K, single chunk). Use as
ballpark only — actual numbers depend on hardware.

| Operation                    | Cold (model load) | Warm (cached)        |
| ---------------------------- | ----------------- | -------------------- |
| DistilBERT fill-mask (top-10)| ~3 s              | ~50 ms               |
| T5 QG (beam=4, max_len=64)   | ~6 s              | ~150 ms              |
| spaCy POS tagging (sm model) | ~2 s              | ~5 ms / 1k chars     |
| Regex fallback masker        | —                 | <1 ms / chunk        |

Models are loaded lazily on first call and cached at module level
(`_PIPELINE_CACHE` / `_MODEL_CACHE`). Subsequent calls within the same
process incur only inference cost.

For 1k chunks expect ~3 min warm on CPU; <30 s on a single mid-range
GPU.

## CPU fallback

The synthesizer defaults to `device=-1` (CPU). It works without any GPU
but is significantly slower. To use a GPU, pass `device=0` (or another
index). The synthesizer never explicitly requires CUDA — torch's
default device mapping is used.

## Integration recipe (workload generator, separate worktree)

The workload generator produces update / remove / insert ops on a
vector store but defers ground-truth synthesis to this module.

### Option A — programmatic

```python
from benchmark.dynamic_groundtruth import DynamicGroundTruthSynthesizer

synth = DynamicGroundTruthSynthesizer(device=-1)
report = synth.synthesize_batch(chunks)
# report.updates: list[UpdateWithGroundTruth]
# report.skipped: list[SynthesisSkipped] (diagnostics)

WorkloadGenerator.update_question_pool(report.updates)
```

### Option B — JSONL file (decouples worktrees)

Generator worktree writes chunks to `chunks.jsonl`; this module writes
`updates.jsonl`:

```bash
python -m benchmark.dynamic_groundtruth.batch \
    chunks.jsonl updates.jsonl --device -1
```

Then in the workload generator:

```python
WorkloadGenerator.update_question_pool(Path("updates.jsonl"))
# Accepts a path to JSONL or a list[UpdateWithGroundTruth].
```

The output JSONL schema is exactly the spec-mandated one (see above), so
the workload generator can read it without any custom adapter.

## Testing

```bash
# Fast unit tests (mocked models, no downloads)
pytest tests/test_dynamic_groundtruth.py

# Slow integration test — actually loads models (opt-in)
DYNAMIC_GTM_RUN_SLOW=1 pytest tests/test_dynamic_groundtruth.py -m slow
```

The fast suite mocks `transformers` pipeline/model calls so it runs in
under a second with no network access. The slow suite downloads
~700 MB of model weights on first run.

## Known limitations

- **Single mask per chunk.** Multi-mask generation would change the T5
  highlight format and is out of scope.
- **`valhalla/t5-base-qg-hl` is single-hop.** It cannot generate
  multi-hop questions; that is acceptable for RAG update verification
  but may be insufficient for complex reasoning benchmarks.
- **Regex fallback nouns are crude.** The heuristic correctly catches
  common nouns but misclassifies some verbs / adjectives. Use spaCy for
  production workloads.
- **No GPU batching.** Inference runs one chunk at a time. For >10k
  chunks, consider batched inference — straightforward to add by
  swapping `fill_mask` and `generate_question` internals.
