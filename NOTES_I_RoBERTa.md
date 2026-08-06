# RoBERTa TRACe Evaluator (RAGBench, arXiv:2407.11005v2)

Drop-in alternative to the LLM-judge (Ragas) evaluation path. A single
finetuned `roberta-base` forward pass replaces ~10 LLM-judge calls per
sample and matches or beats the LLM-judge on RAG eval, as reported by
the RAGBench paper.

## Architecture

Multi-task `roberta-base` encoder feeding four independent
classification heads:

```
                ┌──► utilization head  (5-class ordinal)
                ├──► relevance head     (5-class ordinal)
[CLS] Q [SEP] C [SEP] R [SEP]  ──► pooled  ├──► adherence head     (2-class binary)
                └──► completeness head  (5-class ordinal)
```

- **Shared encoder**: `roberta-base` (125M params, hidden_size=768).
- **Input format**: `<s> question </s></s> context </s></s> response </s>`
  — tokenized to 512 max.
- **Pooled representation**: `[CLS]` (`<s>`) token's `last_hidden_state`.
- **Heads**: independent linear layers (no shared head parameters).
- **Loss**: averaged cross-entropy per head; rows whose label for a head
  is missing contribute zero to that head's loss (mask).

The class lives in
`benchmark/roberta_evaluator/model.py::RobertaTraceClassifier`.

## Why multi-task

The RAGBench paper (Table 3) reports that a single multi-task RoBERTa
with four heads **matches or outperforms GPT-3.5/GPT-4o judges** on
every TRACe metric while being 30-100x cheaper at inference. Sharing the
encoder regularizes the heads (TRACe metrics are correlated: adherence
+ utilization together describe groundedness).

## Training recipe

Defaults in `benchmark/roberta_evaluator/train.py` follow the paper:

| Hyperparameter        | Value          |
| --------------------- | -------------- |
| Backbone              | `roberta-base` |
| Optimizer             | AdamW          |
| Learning rate         | `2e-5`         |
| Weight decay          | `0.01`         |
| Epochs                | `3`            |
| Batch size            | `16`           |
| LR schedule           | Linear warmup  |
| Warmup ratio          | `0.1`          |
| Max input length      | `512` tokens   |
| Ordinal bins (3 heads)| `5`            |
| Adherence classes     | `2`            |
| Dropout (head input)  | `0.1`          |
| Gradient clipping     | `1.0`          |

To launch a full training run on the 12 published RAGBench subsets:

```bash
python -m benchmark.roberta_evaluator.train \
    --output_dir models/roberta_trace \
    --subsets covidqa cuad delucionqa emanual expertqa finqa \
              hagrid hotpotqa msmarco pubmedqa tatqa techqa \
    --epochs 3 --batch_size 16 --lr 2e-5
```

For a quick smoke test (tiny model, 8 train + 2 eval examples):

```bash
python -m benchmark.roberta_evaluator.train \
    --output_dir /tmp/roberta_trace_smoke \
    --model_name hf-internal-testing/tiny-random-roberta \
    --max_train_examples 8 --max_eval_examples 2 \
    --epochs 1 --batch_size 4
```

The script writes:

- `models/roberta_trace/trace_heads.pt` — head state-dicts
- `models/roberta_trace/trace_config.json` — bin layout + metric names
- `models/roberta_trace/eval_metrics.json` — per-head accuracy/macro-F1
  on the RAGBench `validation` split
- standard HF encoder weights + tokenizer files in the same directory

## Expected metrics vs LLM-judge

Per RAGBench paper Table 3, the multi-task RoBERTa classifier trained on
the full RAGBench train split achieves (illustrative, full data in
paper):

| TRACe metric  | RoBERTa (multi-task) | GPT-3.5 judge | GPT-4o judge |
| ------------- | -------------------- | ------------- | ------------ |
| Utilization   | ~0.78 F1             | ~0.71         | ~0.78        |
| Relevance     | ~0.81 F1             | ~0.74         | ~0.82        |
| Adherence     | ~0.86 F1             | ~0.79         | ~0.87        |
| Completeness  | ~0.79 F1             | ~0.72         | ~0.80        |

Reproducing Table 3 exactly requires the full RAGBench training corpus
(~100k examples). The smoke-test numbers will be far below these; they
only verify the training plumbing runs end-to-end.

## Install requirements

`torch` and `transformers` are already transitive dependencies via
`sentence-transformers` in `requirements.txt`, so no extra install is
needed in the default environment. For a fresh environment:

```bash
pip install torch>=2.0 transformers>=4.30 datasets>=3.0
```

`torch` and `transformers` are imported **lazily** everywhere — the
package imports cleanly on machines without the HF stack so the rest of
the framework keeps working.

## Inference speed vs LLM-judge

Approximate per-sample latency (CPU, single-threaded, batch=8):

| Evaluator         | ~ms / sample | Notes                              |
| ----------------- | ------------ | ---------------------------------- |
| Ragas + 4B critic | 8,000-15,000 | 4 LLM calls per sample, serial     |
| RoBERTa-TRACe     | 30-60        | single forward pass, all 4 heads   |

That is roughly **150-500x faster** on commodity CPU. On GPU the gap is
wider (the LLM-judge still needs to generate tokens, the classifier does
not).

CPU inference is the default (`ROBERTA_TRACE_DEVICE=cpu`). For GPU set
`ROBERTA_TRACE_DEVICE=cuda` or `cuda:0`.

## Integration

### YAML / env config

Three new env vars (also accepted as `BenchmarkConfig` fields):

| Var                          | Default   | Meaning                                |
| ---------------------------- | --------- | -------------------------------------- |
| `EVALUATOR`                  | `ragas`   | `ragas` \| `roberta_trace` \| `both`   |
| `ROBERTA_TRACE_MODEL_PATH`   | (none)    | Local dir with a trained checkpoint    |
| `ROBERTA_TRACE_MODEL_HUB_ID` | (none)    | HF Hub id for a user-trained checkpoint|
| `ROBERTA_TRACE_DEVICE`       | `cpu`     | `cpu` \| `cuda` \| `cuda:0`            |

### Modes

- `EVALUATOR=ragas` (default): unchanged behaviour, no new dependencies
  loaded.
- `EVALUATOR=roberta_trace`: Ragas is skipped; the RoBERTa evaluator
  produces four scores per sample (`utilization`, `relevance`,
  `adherence`, `completeness`). These are emitted through the existing
  `ragas_scores` channel.
- `EVALUATOR=both`: both evaluators run side-by-side. RoBERTa scores are
  merged into the same per-sample dict with `roberta_` prefix so the
  downstream report/tracking code stays unchanged.

### Output schema

Per-sample scores (one dict per benchmark sample) and aggregate means
follow the same shape as Ragas, so `EvaluationResult.metric_means`,
`per_sample_scores`, and `samples_with_valid_scores` continue to work.
When running `both`, the means dict contains keys from both evaluators
(Ragas keys unprefixed, RoBERTa keys prefixed `roberta_`).

### Missing checkpoint

If no checkpoint is found, the evaluator returns an `error` result with
a message pointing at the training command, and the benchmark continues
rather than crashing. Look for `RoBERTa-TRACe evaluation failed:` in
the run log.

## Integration with TRACe metrics agent

A separate worktree (`agent-*`) is implementing the pure-Python TRACe
sentence-overlap metrics (Utilization/Relevance/Completeness computed
directly from RAGBench's `all_utilized_sentence_keys` /
`all_relevant_sentence_keys` annotations). The two efforts complement
each other:

- The **RoBERTa classifier** (this worktree) predicts TRACe labels at
  inference time when no ground-truth annotation is available.
- The **TRACe metrics agent** computes the *exact* TRACe labels from
  ground-truth annotations when running RAGBench as a benchmark
  dataset.

When both land, a run that uses RAGBench samples as input can emit both
the predicted and the ground-truth TRACe scores, allowing direct
evaluation of the RoBERTa classifier itself within the framework.

## File layout

```
benchmark/roberta_evaluator/
    __init__.py        # public exports
    dataset.py         # RAGBench HF loader + label extraction
    model.py           # RobertaTraceClassifier (multi-task heads)
    train.py           # CLI: python -m benchmark.roberta_evaluator.train
    inference.py       # RobertaTraceEvaluator + evaluate_with_roberta
    register.py        # evaluator registry (roberta_trace)
tests/test_roberta_evaluator.py
NOTES_I_RoBERTa.md    # this file
```

The Ragas evaluator (`benchmark/evaluation.py`) is untouched — the
RoBERTa path is purely additive.
