"""Gold-document retrieval metrics for datasets with evidence IDs.

These metrics evaluate whether retrieved chunks came from the known gold
document(s) for a question. They are intentionally separate from the existing
heuristic context-overlap metrics so both modes can be compared.

Supports both single-gold datasets (SQuAD-style, one gold doc per question)
and multi-gold datasets (HotpotQA / MuSiQue / 2WikiMultiHopQA, where
``supporting_facts`` references several gold paragraphs). For multi-hop data
the gold set is auto-derived from ``supporting_facts`` / ``supporting_contexts``
in the sample metadata when ``gold_doc_id`` is absent.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class GoldRetrievalMetricsResult:
    metric_means: dict[str, float]
    per_sample: list[dict[str, float | None]]
    samples_with_valid_scores: dict[str, int]
    skipped_samples: int = 0


# ── Identifier extraction ────────────────────────────────────────────


def _extract_doc_id(metadata: dict[str, Any] | None) -> str | None:
    """Pull a comparable doc identifier out of a retrieved-chunk metadata dict.

    Multi-hop corpora expose gold evidence as paragraph titles (HotpotQA,
    2WikiMultiHopQA) or paragraph ids (MuSiQue). Retrieved chunks may therefore
    carry the title in ``title`` / ``paragraph_id`` rather than ``doc_id``.
    """
    if not metadata:
        return None
    for key in ("doc_id", "title", "paragraph_id", "id"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _gold_titles_from_supporting_facts(raw: Any) -> set[str]:
    """Extract unique titles from ``supporting_facts`` / ``supporting_contexts``.

    Both fields follow HotpotQA's shape: ``list[[title, sent_idx], ...]``.
    Some datasets use ``list[{"title": ...}]`` — handled too.
    """
    titles: set[str] = set()
    if not raw:
        return titles
    if isinstance(raw, dict):
        raw = raw.get("title") or raw.get("titles") or []
    for item in raw:
        if isinstance(item, str):
            titles.add(item)
        elif isinstance(item, (list, tuple)) and item:
            titles.add(str(item[0]))
        elif isinstance(item, dict):
            title = item.get("title") or item.get("paragraph_id")
            if title:
                titles.add(str(title))
    return titles


def _resolve_gold_set(
    gold: str | Iterable[str] | None,
    sample_metadata: dict[str, Any] | None,
) -> set[str]:
    """Normalise the per-sample gold argument into a set of comparable ids.

    Order of precedence:
      1. explicit ``gold`` argument (str or iterable of str)
      2. ``metadata["gold_doc_id"]``
      3. ``metadata["gold_doc_ids"]``
      4. titles derived from ``metadata["supporting_facts"]``
      5. titles derived from ``metadata["supporting_contexts"]``
    """
    if gold is not None:
        if isinstance(gold, str):
            return {gold} if gold else set()
        return {str(g) for g in gold if g}

    if sample_metadata:
        single = sample_metadata.get("gold_doc_id")
        if single:
            return {str(single)}
        multi = sample_metadata.get("gold_doc_ids")
        if multi:
            return {str(g) for g in multi if g}
        supporting = _gold_titles_from_supporting_facts(
            sample_metadata.get("supporting_facts")
        )
        if supporting:
            return supporting
        supporting = _gold_titles_from_supporting_facts(
            sample_metadata.get("supporting_contexts")
        )
        if supporting:
            return supporting
    return set()


# ── Metric primitives ────────────────────────────────────────────────


def _hit_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """1.0 if ANY gold id appears in the top-k retrieved, else 0.0."""
    if not gold or k <= 0:
        return 0.0
    return 1.0 if gold & set(retrieved[:k]) else 0.0


def _ndcg_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """NDCG@k with binary multi-relevance.

    IDCG ranks every gold item in the first positions, capped at k. Duplicate
    retrieved ids keep their original positions: a gold document is credited
    only once (at its first rank) instead of being moved up after renumbering
    — deduplicating the list first inflated nDCG (run7 regression). This also
    prevents multiple chunks of the same document from pushing DCG above 1.
    """
    if not gold or k <= 0:
        return 0.0
    credited: set[str] = set()
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k]):
        if doc_id in gold and doc_id not in credited:
            dcg += 1.0 / math.log2(rank + 2)
            credited.add(doc_id)
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(gold: set[str], retrieved: list[str], k: int) -> float:
    """Fraction of gold ids retrieved in top-k."""
    if not gold or k <= 0:
        return 0.0
    hits = len(gold & set(retrieved[:k]))
    return hits / len(gold)


def _span_relevance(
    evidence: list[dict[str, Any]],
    retrieved_ids: list[str | None],
    contexts: list[str],
) -> list[set[int]]:
    """Map provider chunks to gold evidence spans using source + text overlap."""
    relevance: list[set[int]] = []
    for rank, context in enumerate(contexts):
        context_tokens = set(re.findall(r"[a-z0-9]+", context.lower()))
        document_id = retrieved_ids[rank] if rank < len(retrieved_ids) else None
        matched: set[int] = set()
        for evidence_index, item in enumerate(evidence):
            source_id = item.get("source_id")
            if source_id and document_id and str(source_id) != document_id:
                continue
            quote_tokens = set(
                re.findall(r"[a-z0-9]+", str(item.get("quote") or "").lower())
            )
            if not quote_tokens:
                continue
            overlap = len(quote_tokens & context_tokens) / len(quote_tokens)
            if overlap >= 0.6:
                matched.add(evidence_index)
        relevance.append(matched)
    return relevance


def _append_span_metrics(
    scores: dict[str, float | None],
    accum: dict[str, list[float]],
    evidence: list[dict[str, Any]],
    relevance: list[set[int]],
    ks: list[int],
) -> None:
    if not evidence:
        return
    for k in ks:
        top = relevance[:k]
        covered = set().union(*top) if top else set()
        first_rank = next((rank for rank, hits in enumerate(top, 1) if hits), None)
        gains: list[int] = []
        seen: set[int] = set()
        for hits in top:
            new_hits = hits - seen
            gains.append(len(new_hits))
            seen.update(hits)
        dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))
        ideal = min(len(evidence), k)
        idcg = sum(1.0 / math.log2(rank + 2) for rank in range(ideal))
        values = {
            f"span_hit@{k}": 1.0 if covered else 0.0,
            f"span_mrr@{k}": 1.0 / first_rank if first_rank else 0.0,
            f"span_ndcg@{k}": dcg / idcg if idcg else 0.0,
            f"span_recall@{k}": len(covered) / len(evidence),
        }
        for name, value in values.items():
            scores[name] = value
            accum.setdefault(name, []).append(value)


# ── RAGBench sentence-level recall ───────────────────────────────────


@dataclass(frozen=True)
class RelevantSentenceRecallResult:
    metric_means: dict[str, float]
    per_sample: list[float | None]
    samples_with_valid_scores: int


def resolve_relevant_sentences(sample_metadata: dict[str, Any] | None) -> list[str]:
    """Resolve RAGBench sentence labels into the actual relevant sentence texts.

    RAGBench stores ``documents_sentences`` as a list of documents, each a
    list of ``[sentence_key, sentence_text]`` pairs, and
    ``all_relevant_sentence_keys`` lists the keys of the relevant sentences
    (e.g. ``"3a"`` = document 3, sentence ``a``; sometimes with a trailing
    period). A ``"doc.sent"`` numeric fallback is supported for datasets with
    a different layout. Returns the sentence texts; empty when either field
    is missing.
    """
    if not isinstance(sample_metadata, dict):
        return []
    docs = sample_metadata.get("documents_sentences")
    keys = sample_metadata.get("all_relevant_sentence_keys")
    if not isinstance(docs, list) or not isinstance(keys, list):
        return []

    by_key: dict[str, str] = {}
    for doc in docs:
        if not isinstance(doc, list):
            continue
        if doc and isinstance(doc[0], list):
            entries = doc  # document already a flat list of pairs
        else:
            entries = [doc]
        for entry in entries:
            if (
                isinstance(entry, list)
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], str)
                and entry[1].strip()
            ):
                by_key[entry[0]] = entry[1]

    sentences: list[str] = []
    for key in keys:
        if not isinstance(key, str):
            continue
        # Exact key match, then without/with trailing period.
        sentence = by_key.get(key)
        if sentence is None:
            stripped = key.rstrip(".")
            sentence = by_key.get(stripped)
        if sentence is None and "." in key:
            doc_part, _, sent_part = key.partition(".")
            try:
                doc_idx, sent_idx = int(doc_part), int(sent_part)
            except ValueError:
                sent_idx = None
            if sent_idx is not None and 0 <= doc_idx < len(docs):
                doc = docs[doc_idx]
                if isinstance(doc, str):
                    sentence = doc
                elif isinstance(doc, list) and 0 <= sent_idx < len(doc):
                    if isinstance(doc[sent_idx], str):
                        sentence = doc[sent_idx]
        if sentence is not None and sentence.strip():
            sentences.append(sentence)
    return sentences


def _normalised_tokens(text: str) -> list[str]:
    """Lowercase Unicode word tokens; whitespace, punctuation and underscores
    dropped."""
    return re.findall(r"[^\W_]+", text.lower(), re.UNICODE)


def _normalised_words(text: str) -> str:
    """Lowercased text with whitespace collapsed to single spaces.

    Word boundaries are preserved so materially different sentences ("the
    rapist …" vs "therapist …") never match; only case and whitespace are
    normalised away.
    """
    return " ".join(_normalised_tokens(text))


def _compact(text: str) -> str:
    """Normalised text with all separators removed.

    Used only to repair sentences split across a chunk boundary — even
    mid-word — by matching against the concatenation of adjacent chunks.
    """
    return "".join(_normalised_tokens(text))


def _longest_suffix_prefix_overlap(left: str, right: str) -> int:
    """Length of the longest suffix of ``left`` that is a prefix of ``right``.

    KMP prefix function, O(len(left) + len(right)) — no quadratic slicing.
    """
    if not left or not right:
        return 0
    s = right + "\x00" + left
    pi = [0] * len(s)
    for i in range(1, len(s)):
        j = pi[i - 1]
        while j and s[i] != s[j]:
            j = pi[j - 1]
        if s[i] == s[j]:
            j += 1
        pi[i] = j
    return pi[-1]


def _split_repair_joins(chunks: list[str]) -> list[tuple[str, int]]:
    """Compact concatenations of each chunk pair for split repair.

    All pairs are considered — not just rank-adjacent chunks — so a sentence
    split across chunks retrieved at arbitrary ranks still matches and the
    metric does not shift with small embedding or reranker changes. Returns
    ``(joined, boundary)`` tuples where ``boundary`` is the character
    position of the chunk join inside ``joined``. For overlapping chunks the
    longest overlap is trimmed before joining (linear time), so a repeated
    fragment does not break the match. A plain concatenation is included as
    well so a split that cuts mid-word still matches.
    """
    compacts = [_compact(chunk) for chunk in chunks]
    joins: list[tuple[str, int]] = []
    for i in range(len(compacts) - 1):
        left = compacts[i]
        if not left:
            continue
        for j in range(i + 1, len(compacts)):
            right = compacts[j]
            if not right:
                continue
            # Both rank orders: fragments may be retrieved in reverse.
            joins.append((left + right, len(left)))
            joins.append((right + left, len(right)))
            overlap = _longest_suffix_prefix_overlap(left, right)
            if overlap:
                joins.append((left[:-overlap] + right, len(left) - overlap))
                joins.append((right[:-overlap] + left, len(right) - overlap))
    return joins


def _spans_join(compact: str, join: str, boundary: int) -> bool:
    """True if ``compact`` occurs in ``join`` crossing the chunk ``boundary``.

    Requiring the match to span the join point keeps word-boundary semantics:
    a sentence contained entirely within one chunk is only credited by the
    word-boundary path, never by the compact repair path.
    """
    if not compact:
        return False
    pos = join.find(compact)
    while pos != -1:
        if pos < boundary < pos + len(compact):
            return True
        pos = join.find(compact, pos + 1)
    return False


def compute_relevant_sentence_recall(
    retrieved_contexts: list[list[str]],
    sample_metadata: list[dict[str, Any]],
) -> RelevantSentenceRecallResult:
    """Compute per-sample ``relevant_sentence_recall`` from RAGBench labels.

    A sample's score is the share of its relevant sentences (per
    ``all_relevant_sentence_keys``) that appear — after case/whitespace
    normalisation, tolerating chunk-boundary splits — in the retrieved
    contexts. Samples without any relevant sentence (unanswerable per
    RAGBench) or without the label fields at all get ``None`` and are
    excluded from the mean.
    """
    per_sample: list[float | None] = []
    scores: list[float] = []

    for i, meta in enumerate(sample_metadata):
        # Deduplicate resolved sentences: duplicate keys ("3a", "3a.") or the
        # same key listed twice must not inflate the denominator.
        seen_words: set[str] = set()
        unique: list[str] = []
        for sentence in resolve_relevant_sentences(meta):
            words = _normalised_words(sentence)
            if not words or words in seen_words:
                continue  # empty/label-only sentences never match
            seen_words.add(words)
            unique.append(sentence)
        if not unique:
            per_sample.append(None)
            continue
        contexts = retrieved_contexts[i] if i < len(retrieved_contexts) else []
        chunks = [str(c) for c in contexts]
        # Primary match: boundary-padded word match across the space-joined
        # chunks — robust to case and whitespace differences without false
        # positives at word edges ("Yes." never matches "yesterday").
        joined_words = f" {_normalised_words(' '.join(chunks))} "
        # Split repair: compact (boundary-free) concatenations of adjacent
        # chunk pairs, so a sentence cut by a chunk boundary — even mid-word
        # or across overlapping chunks — still matches. The match must span
        # the join point, so compact matching never credits a sentence that
        # sits entirely inside one chunk.
        split_joins = _split_repair_joins(chunks)
        retrieved = 0
        for sentence in unique:
            padded_words = f" {_normalised_words(sentence)} "
            compact = _compact(sentence)
            if padded_words in joined_words or any(
                _spans_join(compact, join, boundary)
                for join, boundary in split_joins
            ):
                retrieved += 1
        value = retrieved / len(unique)
        per_sample.append(value)
        scores.append(value)

    mean = sum(scores) / len(scores) if scores else 0.0
    return RelevantSentenceRecallResult(
        metric_means={"relevant_sentence_recall": mean} if scores else {},
        per_sample=per_sample,
        samples_with_valid_scores=len(scores),
    )


# ── Public entry point ───────────────────────────────────────────────


def compute_gold_doc_retrieval_metrics(
    gold_doc_ids: list[str | None | Iterable[str]],
    retrieved_metadata: list[list[dict[str, Any]]],
    *,
    sample_metadata: list[dict[str, Any]] | None = None,
    retrieved_contexts: list[list[str]] | None = None,
    k_values: list[int] | None = None,
) -> GoldRetrievalMetricsResult:
    """Compute document-level retrieval metrics from gold and retrieved IDs.

    ``gold_doc_ids`` and ``retrieved_metadata`` must be parallel lists. Each
    entry of ``gold_doc_ids`` may be a single id, ``None``, or an iterable of
    ids (for multi-hop datasets with several gold paragraphs). A sample without
    any resolvable gold id is skipped.

    When ``sample_metadata`` is provided, the resolver falls back to
    ``supporting_facts`` / ``gold_doc_ids`` entries so callers that only pass
    multi-hop metadata do not need to pre-flatten gold titles.
    """
    ks = k_values or [1, 3, 5]
    per_sample: list[dict[str, float | None]] = []
    accum: dict[str, list[float]] = {}
    skipped = 0

    for i, (gold, metadata_list) in enumerate(zip(gold_doc_ids, retrieved_metadata)):
        meta = (
            sample_metadata[i] if sample_metadata and i < len(sample_metadata) else None
        )
        gold_set = _resolve_gold_set(gold, meta)
        scores: dict[str, float | None] = {}

        if not gold_set:
            skipped += 1
            for k in ks:
                scores[f"hit@{k}"] = None
                scores[f"ndcg@{k}"] = None
                scores[f"recall@{k}"] = None
                scores[f"supporting_fact_coverage@{k}"] = None
            per_sample.append(scores)
            continue

        # Keep ``None`` entries in place: chunks without a provider document
        # ID must not shift gold documents to better ranks, and the top-k
        # window must cover exactly k retrieved chunks. ``None`` never
        # matches a gold id, so hit/nDCG/recall need no special handling.
        ranked_doc_ids = [_extract_doc_id(metadata) for metadata in metadata_list]

        for k in ks:
            hit = _hit_at_k(gold_set, ranked_doc_ids, k)
            ndcg = _ndcg_at_k(gold_set, ranked_doc_ids, k)
            recall = _recall_at_k(gold_set, ranked_doc_ids, k)
            # supporting_fact_coverage@k is the title-level fraction of gold
            # supporting facts found in the top-k. For multi-hop datasets this
            # is exactly multi-relevant recall@k; for single-gold data it
            # collapses to hit@k. Exposed under its own name so downstream
            # reporting can pick the multi-hop interpretation explicitly.
            coverage = recall
            for name, value in (
                (f"hit@{k}", hit),
                (f"ndcg@{k}", ndcg),
                (f"recall@{k}", recall),
                (f"supporting_fact_coverage@{k}", coverage),
            ):
                scores[name] = value
                accum.setdefault(name, []).append(value)

        evidence = meta.get("evidence", []) if isinstance(meta, dict) else []
        contexts = (
            retrieved_contexts[i]
            if retrieved_contexts and i < len(retrieved_contexts)
            else []
        )
        if isinstance(evidence, list) and contexts:
            relevance = _span_relevance(evidence, ranked_doc_ids, contexts)
            _append_span_metrics(scores, accum, evidence, relevance, ks)

        per_sample.append(scores)

    means = {key: sum(values) / len(values) for key, values in accum.items() if values}
    valid_counts = {key: len(values) for key, values in accum.items() if values}

    return GoldRetrievalMetricsResult(
        metric_means=means,
        per_sample=per_sample,
        samples_with_valid_scores=valid_counts,
        skipped_samples=skipped,
    )
