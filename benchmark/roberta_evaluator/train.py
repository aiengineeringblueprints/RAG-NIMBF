"""CLI training script for the multi-task RoBERTa TRACe classifier.

Example
-------
    python -m benchmark.roberta_evaluator.train \
        --output_dir models/roberta_trace \
        --subsets covidqa hotpotqa \
        --epochs 3 --batch_size 16 --lr 2e-5

Trains the four TRACe heads on the RAGBench train splits and reports
per-metric accuracy / F1 (macro) on the RAGBench dev (``validation``)
split.  See ``NOTES_I_RoBERTa.md`` for the recipe (LR / epochs / batch
size) taken from the RAGBench paper.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
from typing import Iterable, Sequence

from .dataset import DEFAULT_SUBSETS, TraceExample, load_ragbench_split
from .inference import MAX_INPUT_LENGTH, _flatten_contexts, _format_input
from .model import (
    DEFAULT_NUM_BINS,
    TRACE_METRICS,
    RobertaTraceClassifier,
    quantize_score,
)

logger = logging.getLogger(__name__)


def _build_argparser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="benchmark.roberta_evaluator.train",
        description="Train multi-task RoBERTa TRACe classifier on RAGBench.",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = _build_argparser()
    p.add_argument(
        "--output_dir",
        default="models/roberta_trace",
        help="Where to write the trained checkpoint.",
    )
    p.add_argument(
        "--model_name",
        default="roberta-base",
        help="HF backbone id (or local path). Defaults to roberta-base.",
    )
    p.add_argument(
        "--subsets",
        nargs="+",
        default=list(DEFAULT_SUBSETS),
        help="RAGBench subsets to train on (default: all 12).",
    )
    p.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Cap the number of training examples (debugging).",
    )
    p.add_argument(
        "--max_eval_examples",
        type=int,
        default=None,
        help="Cap the number of validation examples (debugging).",
    )
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_bins", type=int, default=DEFAULT_NUM_BINS)
    p.add_argument(
        "--max_input_length",
        type=int,
        default=MAX_INPUT_LENGTH,
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        default=None,
        help="cuda / cuda:0 / cpu. Auto-detect when omitted.",
    )
    p.add_argument(
        "--log_level", default="INFO", help="Python logging level (default INFO)."
    )
    return p.parse_args(argv)


def _build_dataset(
    classifier: RobertaTraceClassifier,
    tokenizer,
    examples: Iterable[TraceExample],
    *,
    max_input_length: int,
    torch,
):
    """Tokenise examples and pack per-head label tensors + masks.

    Returns a ``torch.utils.data.Dataset``-compatible object exposing
    ``__len__`` and ``__getitem__`` returning dicts with ``input_ids``,
    ``attention_mask`` and ``labels_per_head`` ({metric: (tensor, mask)}).
    """
    examples = list(examples)

    formatted = [
        _format_input(ex.question, ex.context, ex.response) for ex in examples
    ]
    encodings = tokenizer(
        formatted,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )

    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]

    # Pre-allocate per-head label + mask tensors.
    labels = {m: torch.full((len(examples),), -1, dtype=torch.long) for m in TRACE_METRICS}
    masks = {m: torch.zeros((len(examples),), dtype=torch.float) for m in TRACE_METRICS}

    for i, ex in enumerate(examples):
        for metric in TRACE_METRICS:
            v = ex.labels.get(metric)
            if v is None:
                continue  # mask stays 0, label stays -1
            if metric == "adherence":
                labels[metric][i] = int(bool(v))
            else:
                labels[metric][i] = quantize_score(float(v), classifier.num_bins)
            masks[metric][i] = 1.0

    class _DS:
        def __init__(self, input_ids, attention_mask, labels, masks):
            self.input_ids = input_ids
            self.attention_mask = attention_mask
            self.labels = labels
            self.masks = masks

        def __len__(self):
            return self.input_ids.shape[0]

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": {m: self.labels[m][idx] for m in TRACE_METRICS},
                "masks": {m: self.masks[m][idx] for m in TRACE_METRICS},
            }

    return _DS(input_ids, attention_mask, labels, masks)


def _masked_ce_loss(logits, labels, mask, torch):
    """Cross-entropy loss ignoring examples whose mask == 0."""
    if mask.sum() == 0:
        return logits.sum() * 0.0  # graph-preserving zero
    reduction_logits = logits[mask.bool()]
    reduction_labels = labels[mask.bool()].clamp(min=0)
    return torch.nn.functional.cross_entropy(reduction_logits, reduction_labels)


def _compute_metrics(logits_per_head, dataset, torch):
    """Per-head accuracy + macro-F1 on the eval split."""
    from collections import Counter

    metrics: dict[str, dict[str, float]] = {}
    for metric in TRACE_METRICS:
        mask = dataset.masks[metric]
        if mask.sum() == 0:
            metrics[metric] = {"accuracy": float("nan"), "macro_f1": float("nan"), "support": 0.0}
            continue
        idxs = mask.bool()
        preds = logits_per_head[metric].argmax(dim=-1)[idxs]
        labels = dataset.labels[metric][idxs]
        correct = (preds == labels).sum().item()
        total = len(labels)
        acc = correct / total if total else float("nan")

        # Macro-F1
        classes = sorted(set(preds.tolist()).union(set(labels.tolist())))
        f1s = []
        for c in classes:
            tp = ((preds == c) & (labels == c)).sum().item()
            fp = ((preds == c) & (labels != c)).sum().item()
            fn = ((preds != c) & (labels == c)).sum().item()
            prec = tp / (tp + fp) if tp + fp > 0 else 0.0
            rec = tp / (tp + fn) if tp + fn > 0 else 0.0
            f1 = (
                2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            )
            f1s.append(f1)
        macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
        metrics[metric] = {
            "accuracy": acc,
            "macro_f1": macro_f1,
            "support": float(total),
        }
    return metrics


def _eval_loop(model, dataloader, torch, device):
    model.eval()
    all_logits = {m: [] for m in TRACE_METRICS}
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            for m in TRACE_METRICS:
                all_logits[m].append(out[m].cpu())
    return {
        m: torch.cat(tensors, dim=0) if tensors else torch.empty(0)
        for m, tensors in all_logits.items()
    }


def _detect_device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch  # type: ignore[import-not-found]

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover - covered by build()
        return "cpu"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    random.seed(args.seed)

    try:
        import torch  # type: ignore[import-not-found]
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            f"Missing dependency for training: {exc}. Install torch and transformers."
        )

    device = _detect_device(args.device)
    logger.info("Training on device=%s", device)

    classifier = RobertaTraceClassifier(
        model_name=args.model_name,
        num_bins=args.num_bins,
    )
    classifier.build()
    model = classifier.build()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info("Loading RAGBench train split (subsets=%s)...", args.subsets)
    train_examples = list(load_ragbench_split(args.subsets, split="train"))
    if args.max_train_examples:
        random.shuffle(train_examples)
        train_examples = train_examples[: args.max_train_examples]
    logger.info("Loaded %d train examples", len(train_examples))

    logger.info("Loading RAGBench validation split...")
    eval_examples = list(load_ragbench_split(args.subsets, split="validation"))
    if args.max_eval_examples:
        random.shuffle(eval_examples)
        eval_examples = eval_examples[: args.max_eval_examples]
    logger.info("Loaded %d validation examples", len(eval_examples))

    train_ds = _build_dataset(
        classifier, tokenizer, train_examples,
        max_input_length=args.max_input_length, torch=torch,
    )
    eval_ds = _build_dataset(
        classifier, tokenizer, eval_examples,
        max_input_length=args.max_input_length, torch=torch,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=0
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = max(1, len(train_loader) * int(args.epochs))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    global_step = 0
    for epoch in range(int(args.epochs) + (1 if args.epochs % 1.0 else 0)):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = (
                _masked_ce_loss(out["utilization"], batch["labels"]["utilization"].to(device), batch["masks"]["utilization"].to(device), torch)
                + _masked_ce_loss(out["relevance"], batch["labels"]["relevance"].to(device), batch["masks"]["relevance"].to(device), torch)
                + _masked_ce_loss(out["adherence"], batch["labels"]["adherence"].to(device), batch["masks"]["adherence"].to(device), torch)
                + _masked_ce_loss(out["completeness"], batch["labels"]["completeness"].to(device), batch["masks"]["completeness"].to(device), torch)
            ) / 4.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            global_step += 1
        avg = running_loss / max(1, len(train_loader))
        logger.info("Epoch %d avg loss=%.4f", epoch + 1, avg)

    # Final eval
    logits_per_head = _eval_loop(model, eval_loader, torch, device)
    metrics = _compute_metrics(logits_per_head, eval_ds, torch)
    logger.info("Validation metrics:")
    print(json.dumps(metrics, indent=2))

    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer.save_pretrained(args.output_dir)
    classifier.save(args.output_dir)
    with open(os.path.join(args.output_dir, "eval_metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Saved trained model to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    # The slow test patches argv directly; main() is invoked via
    # `python -m benchmark.roberta_evaluator.train`.
    raise SystemExit(main())
