"""Standalone non-RAGAS evaluation metrics for RAG benchmarking.

Categories:
    IR metrics   – Hit@k, nDCG, Recall@k, Context Relevance
    NLG metrics  – ROUGE-L, BLEU, METEOR, BERTScore

All functions are pure and stateless.  Use ``compute_custom_metrics()``
to run every metric at once, or call individual functions directly.
"""
from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from benchmark.tracing import observe

logger = logging.getLogger(__name__)


# ── Tokenisation helpers ─────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lower-case whitespace tokenisation with punctuation stripping."""
    return [
        t.strip(".,;:!?\"'()[]{}")
        for t in text.lower().split()
        if t.strip(".,;:!?\"'()[]{}")
    ]


def _simple_stem(word: str) -> str:
    """Crude suffix-stripping stemmer used by :func:`meteor_score`."""
    w = word.lower()
    for suffix, replacement in sorted(
        [
            ("ational", "ate"),  ("tional", "tion"),
            ("enci", "ence"),    ("anci", "ance"),
            ("izer", "ize"),     ("isation", "ize"),
            ("ization", "ize"),  ("ation", "ate"),
            ("ness", ""),        ("ment", ""),
            ("able", ""),        ("ible", ""),
            ("ful", ""),         ("less", ""),
            ("ous", ""),         ("ive", ""),
            ("ing", ""),         ("ied", "y"),
            ("ies", "y"),        ("es", ""),
            ("ed", ""),          ("er", ""),
            ("ly", ""),          ("s", ""),
        ],
        key=lambda pair: -len(pair[0]),
    ):
        if w.endswith(suffix) and len(w) - len(suffix) + len(replacement) >= 2:
            return w[: -len(suffix)] + replacement
    return w


def is_refusal_answer(text: str) -> bool:
    """Return True if *text* is a refusal / non-answer from the LLM."""
    t = text.strip().lower()
    if not t:
        return True
    _REFUSAL_PATTERNS = (
        "i cannot answer",
        "i can't answer",
        "i cannot provide",
        "i can't provide",
        "i am unable to answer",
        "i'm unable to answer",
        "i am not able to answer",
        "not possible to answer",
        "cannot be answered",
        "not sufficient",
        "insufficient",
        "no information",
        "i don't have",
        "i do not have",
        "does not contain",
        "does not provide",
        "not contained",
        "not provided",
        "not available in",
        "not addressed",
        "context does not",
        "provided context",
        "given context",
    )
    return any(p in t for p in _REFUSAL_PATTERNS)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    """Length of the longest common subsequence."""
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if a[i] == b[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[n][m]


# ── Relevance determination ─────────────────────────────────────────

def determine_relevance(
    ground_truth: str,
    contexts: list[str],
    threshold: float = 0.5,
    embed_fn: Callable[[str], np.ndarray] | None = None,
) -> set[int]:
    """Return indices of contexts that are relevant to the ground truth.

    When *embed_fn* is provided, uses cosine similarity between embeddings
    (much more accurate).  Falls back to Jaccard token overlap otherwise.
    Only indices whose score >= *threshold* are considered relevant.
    """
    relevant: set[int] = set()
    if embed_fn is not None:
        gt_emb = embed_fn(ground_truth)
        for i, ctx in enumerate(contexts):
            try:
                ctx_emb = embed_fn(ctx)
            except Exception:
                # Context too long for embedding model — truncate to first 8000 chars
                ctx_emb = embed_fn(ctx[:8000])
            if _cosine_sim(gt_emb, ctx_emb) >= threshold:
                relevant.add(i)
    else:
        gt_tokens = set(_tokenize(ground_truth))
        for i, ctx in enumerate(contexts):
            if _jaccard(gt_tokens, set(_tokenize(ctx))) >= threshold:
                relevant.add(i)
    return relevant


# ════════════════════════════════════════════════════════════════════
# IR Metrics
# ════════════════════════════════════════════════════════════════════

def hit_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """1.0 if any relevant item appears in the top-*k* retrieved, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if relevant & set(retrieved[:k]) else 0.0


def ndcg_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at *k* (binary relevance)."""
    if not relevant or k <= 0:
        return 0.0

    def _dcg(indices: list[int]) -> float:
        return sum(
            1.0 / math.log2(i + 2)  # position = i + 1, so log2(pos + 1)
            for i, idx in enumerate(indices[:k])
            if idx in relevant
        )

    actual = _dcg(retrieved)
    # Ideal: all relevant items ranked first
    ideal = _dcg(list(range(len(retrieved))))
    # Cap ideal DCG at k relevant items
    ideal_k = min(k, len(relevant))
    ideal = sum(1.0 / math.log2(p + 2) for p in range(ideal_k))
    return actual / ideal if ideal > 0 else 0.0


def recall_at_k(relevant: set[int], retrieved: list[int], k: int) -> float:
    """Fraction of relevant items that appear in the top-*k* retrieved."""
    if not relevant or k <= 0:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def context_relevance(
    question: str,
    contexts: list[str],
    embed_fn: Callable[[str], np.ndarray],
    *,
    q_embedding: np.ndarray | None = None,
) -> float:
    """Average cosine similarity between the question and each context.

    *embed_fn* should accept a single string and return a 1-D numpy array.
    *q_embedding* can be provided to skip re-embedding the question.
    """
    if not contexts:
        return 0.0
    q_emb = q_embedding if q_embedding is not None else embed_fn(question)
    scores = []
    for ctx in contexts:
        try:
            ctx_emb = embed_fn(ctx)
        except Exception:
            ctx_emb = embed_fn(ctx[:8000])
        scores.append(_cosine_sim(q_emb, ctx_emb))
    return float(np.mean(scores))


# ════════════════════════════════════════════════════════════════════
# NLG Metrics
# ════════════════════════════════════════════════════════════════════

def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score based on Longest Common Subsequence."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def bleu(prediction: str, reference: str, max_n: int = 4) -> float:
    """Corpus-level BLEU score with brevity penalty.

    Computes modified n-gram precision for n = 1 … *max_n*, then takes
    the geometric mean weighted by uniform weights.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    if len(pred_tokens) == 0:
        return 0.0
    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / len(pred_tokens)))

    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams: Counter[tuple[str, ...]] = Counter(
            tuple(pred_tokens[i : i + n])
            for i in range(len(pred_tokens) - n + 1)
        )
        ref_ngrams: Counter[tuple[str, ...]] = Counter(
            tuple(ref_tokens[i : i + n])
            for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(
            min(count, ref_ngrams[gram]) for gram, count in pred_ngrams.items()
        )
        total = max(sum(pred_ngrams.values()), 1)
        precision_n = clipped / total
        if precision_n == 0.0:
            return 0.0
        log_avg += math.log(precision_n)

    log_avg /= max_n
    return bp * math.exp(log_avg)


def meteor_score(prediction: str, reference: str) -> float:
    """Simplified METEOR score with stemming-based alignment.

    Uses exact word match and simple suffix-stripping stems (no WordNet).
    Applies recall-weighted F-score with a fragmentation penalty.
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return 0.0

    pred_stems = [_simple_stem(t) for t in pred_tokens]
    ref_stems = [_simple_stem(t) for t in ref_tokens]

    ref_matched = [False] * len(ref_tokens)
    pred_matched = [False] * len(pred_tokens)

    # Pass 1: exact word matches
    for i, pw in enumerate(pred_tokens):
        for j, rw in enumerate(ref_tokens):
            if not ref_matched[j] and pw == rw:
                pred_matched[i] = True
                ref_matched[j] = True
                break

    # Pass 2: stem matches for remaining unmatched tokens
    for i, ps in enumerate(pred_stems):
        if pred_matched[i]:
            continue
        for j, rs in enumerate(ref_stems):
            if not ref_matched[j] and ps == rs:
                pred_matched[i] = True
                ref_matched[j] = True
                break

    matches = sum(pred_matched)
    if matches == 0:
        return 0.0

    precision = matches / len(pred_tokens)
    recall = matches / len(ref_tokens)

    # Recall-weighted F (METEOR weights recall 9× more)
    denom = recall + 9.0 * precision
    f_mean = (10.0 * recall * precision) / denom if denom > 0 else 0.0

    # Fragmentation penalty: count contiguous matched segments
    chunks = 0
    in_chunk = False
    for matched in pred_matched:
        if matched and not in_chunk:
            chunks += 1
            in_chunk = True
        elif not matched:
            in_chunk = False

    penalty = 0.5 * (chunks / matches) ** 3
    return f_mean * (1.0 - penalty)


def bert_score(
    prediction: str,
    reference: str,
    model: object,
) -> tuple[float, float, float]:
    """Compute BERTScore (precision, recall, F1) using token-level embeddings.

    *model* should be a ``SentenceTransformer`` instance.  The underlying
    transformer and tokenizer are extracted automatically.

    Requires ``torch`` (transitive dependency of sentence-transformers).
    """
    import torch

    if not prediction.strip() or not reference.strip():
        return 0.0, 0.0, 0.0

    # Extract underlying transformer + tokenizer from SentenceTransformer
    try:
        transformer = model[0].auto_model  # type: ignore[index]
        tokenizer = model.tokenizer  # type: ignore[attr-defined]
    except (AttributeError, IndexError) as exc:
        raise ValueError(
            "Could not extract transformer/tokenizer from model. "
            "Pass a SentenceTransformer instance."
        ) from exc

    def _token_embeddings(text: str) -> torch.Tensor:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(transformer.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = transformer(**inputs)
        # Drop [CLS] and [SEP] → shape (seq_len, hidden_dim)
        return outputs.last_hidden_state.squeeze(0)[1:-1]

    pred_emb = _token_embeddings(prediction)  # (P, D)
    ref_emb = _token_embeddings(reference)  # (R, D)

    # Normalise for cosine similarity
    pred_norm = pred_emb / pred_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ref_norm = ref_emb / ref_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    sim = pred_norm @ ref_norm.T  # (P, R)

    precision = sim.max(dim=1).values.mean().item()
    recall = sim.max(dim=0).values.mean().item()
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return precision, recall, f1


# ════════════════════════════════════════════════════════════════════
# Aggregate
# ════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CustomMetricsResult:
    """Aggregated custom metrics across all samples.

    Mirrors :class:`benchmark.evaluation.EvaluationResult` so the two
    can coexist in the pipeline later.
    """

    metric_means: dict[str, float]
    per_sample: list[dict[str, float | None]]
    samples_with_valid_scores: dict[str, int]
    error: str | None = None


@observe(name="custom_metrics")
def compute_custom_metrics(
    questions: list[str],
    ground_truths: list[str],
    answers: list[str],
    contexts: list[list[str]],
    *,
    ir_k_values: list[int] | None = None,
    relevance_threshold: float = 0.5,
    bleu_max_n: int = 4,
    bert_model: object | None = None,
    embed_fn: Callable[[str], np.ndarray] | None = None,
) -> CustomMetricsResult:
    """Compute all custom metrics for a set of benchmark samples.

    Parameters
    ----------
    questions, ground_truths, answers, contexts
        Parallel lists of benchmark data (same length).
    ir_k_values
        *k* values for Hit@k, nDCG@k, Recall@k.  Defaults to ``[1, 3, 5]``.
    relevance_threshold
        Jaccard threshold for auto-determining context relevance (0–1).
        Default 0.5 — lower values classify too many unrelated chunks as
        relevant, making Hit/nDCG trivially 1.0.
    bleu_max_n
        Maximum n-gram order for BLEU (default 4).
    bert_model
        Optional ``SentenceTransformer`` instance for BERTScore.
        Skipped when ``None``.
    embed_fn
        Optional callable ``str → np.ndarray`` for Context Relevance.
        Skipped when ``None``.
    """
    if not questions:
        return CustomMetricsResult(
            metric_means={},
            per_sample=[],
            samples_with_valid_scores={},
            error="No questions provided.",
        )

    k_values = ir_k_values or [1, 3, 5]
    per_sample: list[dict[str, float | None]] = []
    accum: dict[str, list[float]] = {}

    for idx, (q, gt, ans, ctx) in enumerate(
        zip(questions, ground_truths, answers, contexts)
    ):
        scores: dict[str, float | None] = {}

        # ── IR metrics ──────────────────────────────────────────
        relevant = determine_relevance(gt, ctx, threshold=relevance_threshold, embed_fn=embed_fn)
        retrieved = list(range(len(ctx)))

        for k in k_values:
            h = hit_at_k(relevant, retrieved, k)
            n = ndcg_at_k(relevant, retrieved, k)
            r = recall_at_k(relevant, retrieved, k)
            for name, val in [(f"hit@{k}", h), (f"ndcg@{k}", n), (f"recall@{k}", r)]:
                scores[name] = val
                accum.setdefault(name, []).append(val)

        # Context Relevance + Vector distance metrics (requires embedding function)
        if embed_fn is not None:
            q_emb = embed_fn(q)
            cr = context_relevance(q, ctx, embed_fn, q_embedding=q_emb)
            scores["context_relevance"] = cr
            accum.setdefault("context_relevance", []).append(cr)

            # Vector space distances: question ↔ ground_truth vs question ↔ answer
            gt_emb = embed_fn(gt)
            ans_emb = embed_fn(ans)
            scores["vec_dist_q_gt"] = 1.0 - _cosine_sim(q_emb, gt_emb)
            scores["vec_dist_q_answer"] = 1.0 - _cosine_sim(q_emb, ans_emb)
            accum.setdefault("vec_dist_q_gt", []).append(scores["vec_dist_q_gt"])
            accum.setdefault("vec_dist_q_answer", []).append(scores["vec_dist_q_answer"])
        else:
            scores["context_relevance"] = None
            scores["vec_dist_q_gt"] = None
            scores["vec_dist_q_answer"] = None

        # ── NLG metrics (skip for refusal answers) ──────────────
        if is_refusal_answer(ans):
            for name in ("rouge_l", "bleu", "meteor",
                         "bert_score_precision", "bert_score_recall", "bert_score_f1"):
                scores[name] = 0.0
                accum.setdefault(name, []).append(0.0)
        else:
            for name, val in [
                ("rouge_l", rouge_l(ans, gt)),
                ("bleu", bleu(ans, gt, max_n=bleu_max_n)),
                ("meteor", meteor_score(ans, gt)),
            ]:
                scores[name] = val
                accum.setdefault(name, []).append(val)

            # ── BERTScore (requires model) ──────────────────────
            if bert_model is not None:
                try:
                    p, r, f1 = bert_score(ans, gt, bert_model)
                    for name, val in [
                        ("bert_score_precision", p),
                        ("bert_score_recall", r),
                        ("bert_score_f1", f1),
                    ]:
                        scores[name] = val
                        accum.setdefault(name, []).append(val)
                except Exception as exc:
                    logger.warning("BERTScore failed for sample %d: %s", idx, exc)
                    scores["bert_score_precision"] = 0.0
                    scores["bert_score_recall"] = 0.0
                    scores["bert_score_f1"] = 0.0
                    for n in ("bert_score_precision", "bert_score_recall", "bert_score_f1"):
                        accum.setdefault(n, []).append(0.0)
            else:
                scores["bert_score_precision"] = None
                scores["bert_score_recall"] = None
                scores["bert_score_f1"] = None

        per_sample.append(scores)

    means = {k: sum(v) / len(v) for k, v in accum.items() if v}
    valid_counts = {k: len(v) for k, v in accum.items() if v}

    return CustomMetricsResult(
        metric_means=means,
        per_sample=per_sample,
        samples_with_valid_scores=valid_counts,
    )
