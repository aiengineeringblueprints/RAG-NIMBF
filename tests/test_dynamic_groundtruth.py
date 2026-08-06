"""Tests for benchmark.dynamic_groundtruth.

Unit tests fully mock the transformers models so they don't trigger
heavy downloads.  The slow integration test (marked ``@pytest.mark.slow``)
loads the real models and is skipped unless ``DYNAMIC_GTM_RUN_SLOW=1`` is
set in the environment.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from benchmark.dynamic_groundtruth import (
    DynamicGroundTruthSynthesizer,
    UpdateWithGroundTruth,
)
from benchmark.dynamic_groundtruth import masker as masker_mod
from benchmark.dynamic_groundtruth.masker import (
    MASK_TOKEN,
    MaskError,
    select_mask,
)
from benchmark.dynamic_groundtruth import distilbert_generator as dgb_mod
from benchmark.dynamic_groundtruth.distilbert_generator import (
    Candidate,
    DistilBERTError,
    FillMaskResult,
)
from benchmark.dynamic_groundtruth import t5_question_generator as t5_mod
from benchmark.dynamic_groundtruth.t5_question_generator import (
    QuestionResult,
    T5QuestionError,
)
from benchmark.dynamic_groundtruth.synthesizer import (
    SynthesisReport,
    _coerce_chunk,
    update_from_json_dict,
    update_to_json_dict,
)
from benchmark.dynamic_groundtruth.batch import synthesize_jsonl, _iter_jsonl


# ───────────────────────────────────────────────────────────────────────
# Masker — pure regex / heuristic, fast tests
# ───────────────────────────────────────────────────────────────────────

class TestMaskerFallback:
    """Exercises the regex/heuristic fallback (spaCy disabled)."""

    def test_picks_year_as_mask(self):
        chunk = "The company was founded in 1995 and grew rapidly."
        res = select_mask(chunk, prefer_spacy=False)
        assert res.original_token == "1995"
        assert MASK_TOKEN in res.masked_chunk
        assert res.mask_kind == "number"
        assert res.spacy_used is False

    def test_picks_percentage_as_mask(self):
        chunk = "Revenue grew by 42% in the third quarter alone."
        res = select_mask(chunk, prefer_spacy=False)
        assert res.original_token == "42%"
        assert res.mask_kind == "number"

    def test_picks_decimal_number(self):
        chunk = "The reactor reached 3.14 gigawatts of peak power output."
        res = select_mask(chunk, prefer_spacy=False)
        # 3.14 should be matched before any noun
        assert res.original_token == "3.14"
        assert res.mask_kind == "number"

    def test_falls_back_to_noun_when_no_number(self):
        chunk = (
            "The laboratory developed a new compound using "
            "sophisticated equipment and many reagents."
        )
        res = select_mask(chunk, prefer_spacy=False)
        assert res.mask_kind == "noun"
        assert res.original_token  # non-empty
        assert MASK_TOKEN in res.masked_chunk

    def test_mask_offsets_are_correct(self):
        chunk = "Founded in 1995, the firm expanded."
        res = select_mask(chunk, prefer_spacy=False)
        substr = res.masked_chunk[res.mask_start:res.mask_end]
        assert substr == MASK_TOKEN

    def test_rejects_empty_chunk(self):
        with pytest.raises(MaskError):
            select_mask("", prefer_spacy=False)

    def test_rejects_whitespace_chunk(self):
        with pytest.raises(MaskError):
            select_mask("   \n\t  ", prefer_spacy=False)

    def test_rejects_chunk_with_no_noun_or_number(self):
        # Just prepositions, articles, pronouns
        with pytest.raises(MaskError):
            select_mask("of the by and with", prefer_spacy=False)

    def test_thousands_separated_number(self):
        chunk = "They processed 1,500,000 records during the migration."
        res = select_mask(chunk, prefer_spacy=False)
        assert res.original_token == "1,500,000"
        assert res.mask_kind == "number"

    def test_replacement_is_reversible(self):
        """The mask token sits exactly where the original was."""
        chunk = "Born in 1879, Einstein published revolutionary papers."
        res = select_mask(chunk, prefer_spacy=False)
        # Round-trip: replace [MASK] back with original → original chunk
        rebuilt = res.masked_chunk.replace(MASK_TOKEN, res.original_token, 1)
        assert rebuilt == chunk


@pytest.mark.skipif(
    masker_mod._get_spacy_nlp() is None,
    reason="spaCy en_core_web_sm not installed",
)
class TestMaskerSpacy:
    """Only runs if spaCy + en_core_web_sm are available."""

    def test_spacy_picks_number_first(self):
        chunk = "The firm was established in 2003 and went public later."
        res = select_mask(chunk, prefer_spacy=True)
        assert res.mask_kind == "number"
        assert res.original_token == "2003"
        assert res.spacy_used is True


# ───────────────────────────────────────────────────────────────────────
# DistilBERT generator — mocked pipeline
# ───────────────────────────────────────────────────────────────────────

class TestDistilBERTGenerator:
    """Mock the fill-mask pipeline; never load real models."""

    def _mock_preds(self, *toks_scores: tuple[str, float]) -> list[dict]:
        return [
            {
                "score": score,
                "token_str": tok,
                "sequence": f"prefix {tok} suffix",
            }
            for tok, score in toks_scores
        ]

    def test_picks_top_candidate_different_from_original(self):
        preds = self._mock_preds(
            ("2003", 0.5),  # equals original — must be skipped
            ("2005", 0.3),  # this one should be chosen
            ("2007", 0.1),
        )
        with patch.object(dgb_mod, "_ensure_pipeline") as mock_pipe, \
             patch.object(dgb_mod, "_PIPELINE_CACHE", {}):
            mock_pipe.return_value = lambda *a, **k: preds
            res = dgb_mod.fill_mask(
                "Founded in [MASK].",
                original_token="2003",
                min_confidence=0.05,
            )
        assert res.chosen.token == "2005"
        assert res.chosen.score == pytest.approx(0.3)

    def test_filters_low_confidence(self):
        preds = self._mock_preds(
            ("good", 0.5),
            ("bad", 0.01),  # below threshold
        )
        with patch.object(dgb_mod, "_ensure_pipeline") as mock_pipe, \
             patch.object(dgb_mod, "_PIPELINE_CACHE", {}):
            mock_pipe.return_value = lambda *a, **k: preds
            res = dgb_mod.fill_mask("The [MASK] idea.", min_confidence=0.05)
        assert [c.token for c in res.alternatives] == ["good"]

    def test_filters_punctuation_and_stopwords(self):
        preds = self._mock_preds(
            (",", 0.4),     # punctuation
            ("the", 0.3),   # stopword
            ("rivers", 0.2),
        )
        with patch.object(dgb_mod, "_ensure_pipeline") as mock_pipe, \
             patch.object(dgb_mod, "_PIPELINE_CACHE", {}):
            mock_pipe.return_value = lambda *a, **k: preds
            res = dgb_mod.fill_mask("Many [MASK] flow east.")
        assert res.chosen.token == "rivers"

    def test_raises_when_no_candidate_passes_filters(self):
        preds = self._mock_preds(("the", 0.3), ("a", 0.2))
        with patch.object(dgb_mod, "_ensure_pipeline") as mock_pipe, \
             patch.object(dgb_mod, "_PIPELINE_CACHE", {}):
            mock_pipe.return_value = lambda *a, **k: preds
            with pytest.raises(DistilBERTError):
                dgb_mod.fill_mask("The [MASK] thing.")

    def test_raises_when_no_mask_token(self):
        with pytest.raises(DistilBERTError):
            dgb_mod.fill_mask("no mask here")

    def test_raises_when_multiple_masks(self):
        with pytest.raises(DistilBERTError):
            dgb_mod.fill_mask(f"{MASK_TOKEN} and {MASK_TOKEN}")


# ───────────────────────────────────────────────────────────────────────
# T5 question generator — mocked model
# ───────────────────────────────────────────────────────────────────────

class _FakeTensor:
    """Tiny stand-in for a torch.Tensor — supports ``.to(device)``."""

    def __init__(self, data):
        self.data = data

    def to(self, *_a, **_k):
        return self


class _FakeTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": _FakeTensor([[1, 2, 3]]),
            "attention_mask": _FakeTensor([[1, 1, 1]]),
        }

    def decode(self, ids, **kwargs):
        # Caller-controlled fake output via an attribute
        return getattr(self, "_next_output", "When was this founded?")


class _FakeModel:
    device = "cpu"

    def to(self, *_a, **_k):
        return self

    def generate(self, **kwargs):
        # Token ids corresponding to whatever the tokenizer decodes to.
        return [[101, 102]]


class TestT5Generator:
    """Mock the T5 tokenizer/model — never load real models."""

    def test_generates_question(self):
        tok = _FakeTokenizer()
        tok._next_output = "When was the company founded?"
        text = "Founded in 2003, the company grew."
        # Find offsets of "2003" so the assertion is robust to typos.
        ans = "2003"
        a_start = text.index(ans)
        a_end = a_start + len(ans)
        with patch.object(t5_mod, "_ensure_model") as mock_m, \
             patch.object(t5_mod, "_MODEL_CACHE", {}):
            mock_m.return_value = (tok, _FakeModel())
            res = t5_mod.generate_question(
                text,
                answer_start=a_start,
                answer_end=a_end,
            )
        assert "2003" in res.answer
        assert res.question.endswith("?")

    def test_rejects_too_long_question(self):
        long_q = " ".join(["word"] * 50) + "?"
        tok = _FakeTokenizer()
        tok._next_output = long_q
        with patch.object(t5_mod, "_ensure_model") as mock_m, \
             patch.object(t5_mod, "_MODEL_CACHE", {}):
            mock_m.return_value = (tok, _FakeModel())
            with pytest.raises(T5QuestionError, match="too long"):
                t5_mod.generate_question(
                    "Some chunk with answer here",
                    answer_start=5,
                    answer_end=11,
                )

    def test_rejects_empty_question(self):
        tok = _FakeTokenizer()
        tok._next_output = ""
        with patch.object(t5_mod, "_ensure_model") as mock_m, \
             patch.object(t5_mod, "_MODEL_CACHE", {}):
            mock_m.return_value = (tok, _FakeModel())
            with pytest.raises(T5QuestionError, match="empty"):
                t5_mod.generate_question(
                    "Some chunk with answer here",
                    answer_start=5,
                    answer_end=11,
                )

    def test_invalid_span_raises(self):
        with pytest.raises(T5QuestionError, match="Invalid answer span"):
            t5_mod.generate_question(
                "short",
                answer_start=0,
                answer_end=100,
            )


# ───────────────────────────────────────────────────────────────────────
# Synthesizer — orchestrator, all stages mocked
# ───────────────────────────────────────────────────────────────────────

_CHUNK_TEXT = (
    "The Acme Corporation was founded in 1995 in Palo Alto, California, "
    "and quickly became a leader in consumer electronics manufacturing "
    "throughout the western United States during the late nineties boom."
)


def _patch_stages(
    monkeypatch,
    *,
    new_token: str = "2003",
    question: str = "When was Acme Corporation founded?",
    confidence: float = 0.42,
):
    """Patch the three model-touching stages with deterministic stubs."""
    def fake_select(chunk, *, prefer_spacy=True, spacy_model="en_core_web_sm"):
        # Find a 4-digit year so the offsets are real.
        import re
        m = re.search(r"\d{4}", chunk)
        if not m:
            raise MaskError("no year in test chunk")
        start, end = m.span()
        return masker_mod.MaskResult(
            masked_chunk=chunk[:start] + MASK_TOKEN + chunk[end:],
            original_token=m.group(0),
            mask_start=start,
            mask_end=start + len(MASK_TOKEN),
            mask_kind="number",
            spacy_used=False,
        )
    monkeypatch.setattr("benchmark.dynamic_groundtruth.synthesizer.select_mask", fake_select)

    monkeypatch.setattr(
        "benchmark.dynamic_groundtruth.distilbert_generator.fill_mask",
        lambda masked, *, original_token=None, model_name=None, device=-1,
               top_k=10, min_confidence=0.05: FillMaskResult(
            chosen=Candidate(token=new_token, score=confidence, raw_sequence=masked.replace(MASK_TOKEN, new_token)),
            alternatives=[Candidate(token=new_token, score=confidence, raw_sequence="")],
        ),
    )

    def fake_gen(chunk_with_answer, *, answer_start, answer_end, **_):
        return QuestionResult(
            question=question,
            answer=chunk_with_answer[answer_start:answer_end],
            raw_output=question,
        )
    monkeypatch.setattr(
        "benchmark.dynamic_groundtruth.t5_question_generator.generate_question",
        fake_gen,
    )


class TestSynthesizer:
    def test_synthesize_one_success(self, monkeypatch):
        _patch_stages(monkeypatch)
        synth = DynamicGroundTruthSynthesizer(device=-1, dedup_questions=False)
        upd = synth.synthesize_one("c1", _CHUNK_TEXT)
        assert isinstance(upd, UpdateWithGroundTruth)
        assert upd.chunk_id == "c1"
        assert upd.masked_token_original == "1995"
        assert upd.masked_token_new == "2003"
        assert upd.question == "When was Acme Corporation founded?"
        assert upd.answer == "2003"
        assert "2003" in upd.updated
        assert "1995" not in upd.updated

    def test_synthesize_batch_with_dict_chunks(self, monkeypatch):
        _patch_stages(monkeypatch)
        # Disable dedup: both chunks produce the same question otherwise.
        synth = DynamicGroundTruthSynthesizer(device=-1, dedup_questions=False)
        chunks = [
            {"chunk_id": "a", "text": _CHUNK_TEXT},
            {"chunk_id": "b", "content": _CHUNK_TEXT},
        ]
        report = synth.synthesize_batch(chunks)
        assert isinstance(report, SynthesisReport)
        assert len(report.updates) == 2
        assert {u.chunk_id for u in report.updates} == {"a", "b"}

    def test_synthesize_batch_skips_short_chunk(self, monkeypatch):
        _patch_stages(monkeypatch)
        synth = DynamicGroundTruthSynthesizer(device=-1, min_chunk_chars=50)
        report = synth.synthesize_batch([
            {"chunk_id": "long", "text": _CHUNK_TEXT},
            {"chunk_id": "short", "text": "tiny."},
        ])
        assert len(report.updates) == 1
        assert report.updates[0].chunk_id == "long"
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "chunk_too_short"

    def test_synthesize_batch_dedups_questions(self, monkeypatch):
        _patch_stages(monkeypatch, question="When was Acme Corporation founded?")
        synth = DynamicGroundTruthSynthesizer(device=-1, dedup_questions=True)
        report = synth.synthesize_batch([
            {"chunk_id": "a", "text": _CHUNK_TEXT},
            {"chunk_id": "b", "text": _CHUNK_TEXT},
        ])
        # Second is deduped
        assert len(report.updates) == 1
        assert len(report.skipped) == 1
        assert report.skipped[0].reason == "duplicate_question"

    def test_synthesize_batch_handles_str_and_tuple_inputs(self, monkeypatch):
        _patch_stages(monkeypatch)
        # Disable dedup so the second (identical) chunk isn't dropped.
        synth = DynamicGroundTruthSynthesizer(device=-1, dedup_questions=False)
        report = synth.synthesize_batch([
            _CHUNK_TEXT,
            ("named", _CHUNK_TEXT),
        ])
        assert len(report.updates) == 2
        assert report.updates[0].chunk_id == "chunk-0000"
        assert report.updates[1].chunk_id == "named"

    def test_synthesize_batch_skips_when_no_mask_target(self, monkeypatch):
        _patch_stages(monkeypatch)
        synth = DynamicGroundTruthSynthesizer(device=-1, min_chunk_chars=10)
        # All stop-words / prepositions — no noun or number
        report = synth.synthesize_batch([{"chunk_id": "x", "text": "of the by and with from into"}])
        assert len(report.updates) == 0
        assert report.skipped[0].reason == "no_mask_target"

    def test_synthesize_batch_skips_on_distilbert_failure(self, monkeypatch):
        _patch_stages(monkeypatch)
        monkeypatch.setattr(
            "benchmark.dynamic_groundtruth.distilbert_generator.fill_mask",
            lambda *a, **k: (_ for _ in ()).throw(DistilBERTError("no good candidate")),
        )
        synth = DynamicGroundTruthSynthesizer(device=-1)
        report = synth.synthesize_batch([{"chunk_id": "x", "text": _CHUNK_TEXT}])
        assert report.skipped[0].reason == "distilbert_failed"

    def test_synthesize_batch_skips_on_t5_failure(self, monkeypatch):
        _patch_stages(monkeypatch)
        monkeypatch.setattr(
            "benchmark.dynamic_groundtruth.t5_question_generator.generate_question",
            lambda *a, **k: (_ for _ in ()).throw(T5QuestionError("bad question")),
        )
        synth = DynamicGroundTruthSynthesizer(device=-1)
        report = synth.synthesize_batch([{"chunk_id": "x", "text": _CHUNK_TEXT}])
        assert report.skipped[0].reason == "t5_failed"

    def test_json_roundtrip(self, monkeypatch):
        _patch_stages(monkeypatch)
        synth = DynamicGroundTruthSynthesizer(device=-1)
        upd = synth.synthesize_one("c1", _CHUNK_TEXT)
        d = update_to_json_dict(upd)
        # Verify spec schema (mask_kind/confidence are diagnostic only)
        assert set(d.keys()) == {
            "chunk_id", "original", "updated",
            "masked_token_original", "masked_token_new",
            "question", "answer",
        }
        # Deserialisation restores the spec fields; diagnostic fields
        # default because they aren't part of the JSON schema.
        upd2 = update_from_json_dict(d)
        for field_name in (
            "chunk_id", "original", "updated",
            "masked_token_original", "masked_token_new",
            "question", "answer",
        ):
            assert getattr(upd2, field_name) == getattr(upd, field_name)

    def test_coerce_chunk_variants(self):
        cid, text = _coerce_chunk({"chunk_id": "x", "text": "hello"}, 0)
        assert cid == "x" and text == "hello"
        cid, text = _coerce_chunk({"id": "y", "content": "world"}, 1)
        assert cid == "y" and text == "world"
        cid, text = _coerce_chunk("rawtext", 2)
        assert cid == "chunk-0002" and text == "rawtext"
        cid, text = _coerce_chunk(("abc", "def"), 3)
        assert cid == "abc" and text == "def"

    def test_coerce_chunk_unsupported_type_raises(self):
        with pytest.raises(TypeError):
            _coerce_chunk(42, 0)


# ───────────────────────────────────────────────────────────────────────
# Batch JSONL I/O
# ───────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_iter_jsonl_handles_dicts_and_skips_malformed(self, tmp_path):
        p = tmp_path / "in.jsonl"
        p.write_text(
            json.dumps({"chunk_id": "a", "text": "first"}) + "\n"
            + json.dumps({"chunk_id": "b", "text": "second"}) + "\n"
            + "\n"  # blank line — skipped
            + json.dumps("bare string is valid JSON") + "\n"  # valid JSON string
            + "not-json-at-all\n",  # malformed — logged + skipped
            encoding="utf-8",
        )
        items = list(_iter_jsonl(p))
        # 2 dict records + 1 valid JSON-string record
        assert len(items) == 3
        assert items[0]["chunk_id"] == "a"
        assert items[2] == "bare string is valid JSON"

    def test_synthesize_jsonl_writes_output(self, tmp_path, monkeypatch):
        _patch_stages(monkeypatch)
        inp = tmp_path / "in.jsonl"
        out = tmp_path / "out.jsonl"
        inp.write_text(
            json.dumps({"chunk_id": "a", "text": _CHUNK_TEXT}) + "\n"
            + json.dumps({"chunk_id": "b", "text": _CHUNK_TEXT}) + "\n",
            encoding="utf-8",
        )
        summary = synthesize_jsonl(inp, out, device=-1)
        assert summary["input_count"] == 2
        # dedup is on by default → only 1 written
        assert summary["output_count"] == 1
        assert summary["skipped_count"] == 1
        assert out.exists()
        # Verify each line is valid JSON with the right schema
        lines = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1
        assert lines[0]["masked_token_original"] == "1995"
        assert lines[0]["masked_token_new"] == "2003"


# ───────────────────────────────────────────────────────────────────────
# Slow integration test — actually loads models.  Skipped by default.
# ───────────────────────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(
    not pytest.importorskip("transformers", reason="transformers not installed"),
    reason="transformers not installed",
)
class TestSlowIntegration:
    """Loads real DistilBERT + T5 models.  Slow and network-bound."""

    @pytest.fixture(autouse=True)
    def _require_explicit_opt_in(self):
        import os
        if os.getenv("DYNAMIC_GTM_RUN_SLOW") != "1":
            pytest.skip("Set DYNAMIC_GTM_RUN_SLOW=1 to run slow integration tests")

    def test_end_to_end_one_chunk(self):
        synth = DynamicGroundTruthSynthesizer(device=-1, dedup_questions=False)
        upd = synth.synthesize_one("it-1", _CHUNK_TEXT)
        assert upd.masked_token_original != upd.masked_token_new
        assert upd.question
        assert upd.answer
