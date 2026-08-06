"""Tests for benchmark.roberta_evaluator — RoBERTa TRACe evaluator.

Heavy model loading is mocked so the unit tests run in seconds without
GPU/network access.  A ``@pytest.mark.slow`` integration test exercises
the real training loop with a tiny model + 10 examples.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable when run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.roberta_evaluator import (
    TRACE_METRICS,
    RobertaTraceClassifier,
    RobertaTraceEvalResult,
    RobertaTraceEvaluator,
    get_evaluator,
    register,
)
from benchmark.roberta_evaluator.dataset import (
    RAGBENCH_HUB_ID,
    RAGBENCH_MIRROR_HUB_ID,
    _documents_to_context,
    _coerce_float,
    row_to_example,
    to_trace_tuples,
)
from benchmark.roberta_evaluator.inference import (
    _flatten_contexts,
    _format_input,
    evaluate_with_roberta,
)
from benchmark.roberta_evaluator.model import (
    DEFAULT_NUM_BINS,
    dequantize_bin,
    quantize_score,
)


# ─── dataset helpers ────────────────────────────────────────────────


class TestDatasetHelpers:
    def test_documents_to_context_flat_pairs(self):
        docs = [["0a", "Title: foo"], ["0b", "Body text here"]]
        assert _documents_to_context(docs) == "Title: foo\nBody text here"

    def test_documents_to_context_handles_plain_strings(self):
        assert _documents_to_context(["alpha", "beta"]) == "alpha\nbeta"

    def test_documents_to_context_empty(self):
        assert _documents_to_context(None) == ""
        assert _documents_to_context([]) == ""

    def test_coerce_float_handles_bool(self):
        # RAGBench stores adherence as bool/int
        assert _coerce_float(True) == 1.0
        assert _coerce_float(1) == 1.0
        assert _coerce_float(0.5) == 0.5

    def test_coerce_float_handles_none_and_garbage(self):
        assert _coerce_float(None) is None
        assert _coerce_float("not-a-number") is None

    def test_row_to_example_extracts_all_four_labels(self):
        row = {
            "question": "  What is X? ",
            "response": "  X is a thing. ",
            "documents": [["0a", "Doc text"]],
            "utilization_score": 0.5,
            "relevance_score": 0.75,
            "completeness_score": 1.0,
            "adherence_score": True,
        }
        ex = row_to_example(row)
        assert ex.question == "What is X?"
        assert ex.response == "X is a thing."
        assert ex.context == "Doc text"
        assert ex.labels == {
            "utilization": 0.5,
            "relevance": 0.75,
            "completeness": 1.0,
            "adherence": 1.0,
        }

    def test_row_to_example_supports_answer_alias(self):
        # Older RAGBench revisions use "answer" instead of "response".
        ex = row_to_example({"question": "q", "answer": "a", "documents": []})
        assert ex.response == "a"

    def test_row_to_example_masks_missing_labels(self):
        ex = row_to_example({"question": "q", "response": "a", "documents": []})
        for m in TRACE_METRICS:
            assert ex.labels[m] is None

    def test_to_trace_tuples_roundtrip(self):
        from benchmark.roberta_evaluator.dataset import TraceExample

        ex = TraceExample(
            question="q",
            context="c",
            response="r",
            labels={"utilization": 0.5, "relevance": 0.5, "adherence": 1.0, "completeness": 0.0},
        )
        out = to_trace_tuples([ex])
        assert out[0]["question"] == "q"
        assert out[0]["labels"]["utilization"] == 0.5

    def test_hub_ids_present(self):
        assert "rungalileo" in RAGBENCH_HUB_ID
        assert "galileo" in RAGBENCH_MIRROR_HUB_ID


# ─── quantisation ───────────────────────────────────────────────────


class TestQuantization:
    def test_quantize_score_zero(self):
        assert quantize_score(0.0, 5) == 0

    def test_quantize_score_one(self):
        assert quantize_score(1.0, 5) == 4

    def test_quantize_score_midpoint(self):
        # 0.3 with 5 bins → bin 1 (edge at 0.2)
        assert quantize_score(0.3, 5) == 1

    def test_quantize_score_clamps(self):
        assert quantize_score(-0.5, 5) == 0
        assert quantize_score(2.0, 5) == 4

    def test_quantize_score_nan_raises(self):
        with pytest.raises(ValueError):
            quantize_score(float("nan"), 5)

    def test_dequantize_bin_midpoint(self):
        # Bin 0 of 5 → 0.1; bin 4 of 5 → 0.9
        assert dequantize_bin(0, 5) == pytest.approx(0.1)
        assert dequantize_bin(4, 5) == pytest.approx(0.9)


# ─── inference helpers ──────────────────────────────────────────────


class TestInferenceHelpers:
    def test_format_input_joins_three_fields(self):
        s = _format_input("Q", "C", "R")
        # Roberta separator </s></s> between segments
        assert "Q" in s and "C" in s and "R" in s
        assert "Q</s></s>C</s></s>R" == s

    def test_format_input_handles_empty_fields(self):
        assert _format_input("", "", "") == "</s></s></s></s>"

    def test_flatten_contexts_list_of_strings(self):
        assert _flatten_contexts(["a", "b"]) == "a\nb"

    def test_flatten_contexts_list_of_lists(self):
        assert _flatten_contexts([["a", "b"], ["c"]]) == "a\nb\nc"

    def test_flatten_contexts_handles_none_and_empty(self):
        assert _flatten_contexts(None) == ""
        assert _flatten_contexts([]) == ""
        assert _flatten_contexts(["", None]) == ""


# ─── evaluator with mocked model ────────────────────────────────────


def _make_fake_torch_logits(per_sample_scores: list[dict[str, float]]):
    """Build a fake torch-like logits dict the evaluator will return."""
    fake_logits = {}
    for metric in TRACE_METRICS:
        rows = []
        for s in per_sample_scores:
            v = s.get(metric, 0.5)
            # Build a 2-or-5 class logit tensor where argmax == expected bin.
            if metric == "adherence":
                rows.append([0.0, 5.0] if v >= 0.5 else [5.0, 0.0])
            else:
                bins = DEFAULT_NUM_BINS
                target = quantize_score(v, bins)
                row = [-5.0] * bins
                row[target] = 5.0
                rows.append(row)
        fake_logits[metric] = rows
    return fake_logits


class _FakeTensor:
    """Tiny stand-in for a torch tensor supporting .item() / indexing.

    Internal shape is always 2-D (rows × cols).  ``__getitem__`` with an
    int returns a *1-D row* tensor (single list, not wrapped in a list of
    lists); indexing a 1-D row with another int returns a 0-D scalar.
    """

    def __init__(self, data):
        # ``data`` is either a list[float] (1-D row) or list[list[float]] (2-D).
        if data and isinstance(data[0], list):
            self._rows = data  # 2-D
            self._is_row = False
        else:
            self._rows = [data]  # treat 1-D as single row
            self._is_row = True

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self._is_row:
                # Indexing a 1-D row at an int → scalar
                return _FakeScalar(self._rows[0][idx])
            # Indexing 2-D at row idx → 1-D row tensor
            return _FakeTensor(self._rows[idx])
        return self

    def cpu(self):
        return self

    def item(self):
        # Return the single scalar when 1-D/0-D, else first element.
        if self._is_row:
            return self._rows[0][0] if self._rows[0] else 0.0
        return self._rows[0][0] if self._rows and self._rows[0] else 0.0

    def argmax(self, dim=-1):
        if self._is_row:
            return _FakeScalar(max(range(len(self._rows[0])), key=lambda i: self._rows[0][i]))
        return _IdxTensor([max(range(len(r)), key=lambda i: r[i]) for r in self._rows])

    def softmax(self, dim=-1):
        import math

        def _softmax_row(r):
            m = max(r)
            exps = [math.exp(v - m) for v in r]
            s = sum(exps)
            return [e / s for e in exps]

        if self._is_row:
            return _FakeTensor(_softmax_row(self._rows[0]))
        return _FakeTensor([_softmax_row(r) for r in self._rows])

    def __iter__(self):
        if self._is_row:
            yield _FakeScalar(self._rows[0][0]) if self._rows[0] else _FakeScalar(0.0)
            return
        for r in self._rows:
            yield _FakeTensor(r)


class _FakeScalar:
    """Stand-in for a 0-D torch scalar tensor."""

    def __init__(self, v):
        self._v = float(v)

    def item(self):
        return self._v

    def cpu(self):
        return self


class _IdxTensor:
    def __init__(self, idxs):
        self._idxs = idxs

    def __getitem__(self, i):
        return self if i is None else _IdxScalar(self._idxs[i])

    def item(self):
        return self._idxs[0] if self._idxs else 0


class _IdxScalar:
    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v


def _patch_torch_and_transformers(monkeypatch, fake_logits_rows):
    """Inject fake torch/transformers into the inference module."""
    fake_torch = types.SimpleNamespace(
        no_grad=lambda: MagicMock(__enter__=lambda self: None, __exit__=lambda *a: None),
        float32="float32",
        long="long",
        Tensor=_FakeTensor,
        # ``torch.argmax`` / ``torch.softmax`` are module-level functions.
        # They delegate to the corresponding methods on our _FakeTensor.
        argmax=lambda t, dim=-1: t.argmax(dim=dim),
        softmax=lambda t, dim=-1: t.softmax(dim=dim),
    )

    # Build a tokenizer mock whose __call__ returns a real dict of
    # tensor-like objects (anything with a ``.to`` method works).
    class _FakeTokenTensor:
        def __init__(self, val):
            self.val = val

        def to(self, device):
            return self

    fake_tokenizer = MagicMock()
    fake_tokenizer.return_value = {
        "input_ids": _FakeTokenTensor([[1, 2, 3]]),
        "attention_mask": _FakeTokenTensor([[1, 1, 1]]),
    }
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=MagicMock(from_pretrained=MagicMock(return_value=fake_tokenizer)),
    )

    # Patch the lazy imports inside inference._ensure_loaded and model.build.
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    # Patch classifier so build() returns a mock with predict_logits.
    fake_classifier = MagicMock()
    fake_classifier.num_bins = DEFAULT_NUM_BINS
    fake_classifier.model_name = "roberta-base"

    def _predict(input_ids=None, attention_mask=None):
        return {
            m: _FakeTensor(rows) for m, rows in fake_logits_rows.items()
        }

    fake_classifier.predict_logits = _predict
    fake_classifier.build = MagicMock(return_value=fake_classifier)
    monkeypatch.setattr(
        "benchmark.roberta_evaluator.inference.RobertaTraceClassifier.from_pretrained",
        MagicMock(return_value=fake_classifier),
    )
    return fake_torch, fake_transformers


@pytest.fixture
def checkpoint_dir(tmp_path):
    """Create a fake checkpoint directory with the expected files."""
    import json

    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "trace_heads.pt").write_bytes(b"")
    (d / "trace_config.json").write_text(
        json.dumps(
            {
                "model_name": "roberta-base",
                "num_bins": DEFAULT_NUM_BINS,
                "dropout": 0.1,
                "metrics": list(TRACE_METRICS),
            }
        )
    )
    return str(d)


class TestRobertaTraceEvaluator:
    def test_evaluate_empty_questions_returns_error(self):
        evaluator = RobertaTraceEvaluator(model_path="nope")
        result = evaluator.evaluate([], [], [], [])
        assert isinstance(result, RobertaTraceEvalResult)
        assert result.error is not None
        assert result.per_sample_scores == []

    def test_missing_checkpoint_returns_actionable_error(self, tmp_path):
        evaluator = RobertaTraceEvaluator(model_path=str(tmp_path / "does-not-exist"))
        result = evaluator.evaluate(["q"], ["gt"], ["a"], [["c"]])
        assert result.error is not None
        assert "train one" in result.error.lower() or "checkpoint" in result.error.lower()
        assert len(result.per_sample_scores) == 1

    def test_evaluate_with_mocked_model(self, monkeypatch, checkpoint_dir):
        # Three samples with predictable scores.
        samples = [
            {"utilization": 0.1, "relevance": 0.1, "adherence": 0.0, "completeness": 0.1},
            {"utilization": 0.9, "relevance": 0.9, "adherence": 1.0, "completeness": 0.9},
            {"utilization": 0.5, "relevance": 0.5, "adherence": 1.0, "completeness": 0.5},
        ]
        fake_logits = _make_fake_torch_logits(samples)
        _patch_torch_and_transformers(monkeypatch, fake_logits)

        evaluator = RobertaTraceEvaluator(model_path=checkpoint_dir, batch_size=2)
        result = evaluator.evaluate(
            questions=["q1", "q2", "q3"],
            ground_truths=["gt1", "gt2", "gt3"],
            answers=["a1", "a2", "a3"],
            contexts=[["c1"], ["c2"], ["c3"]],
        )
        assert result.error is None
        assert len(result.per_sample_scores) == 3
        for sample_scores in result.per_sample_scores:
            assert set(sample_scores.keys()) == set(TRACE_METRICS)
            for v in sample_scores.values():
                assert v is not None and 0.0 <= v <= 1.0

        # Metric means present for all four heads.
        for m in TRACE_METRICS:
            assert m in result.metric_means
            assert m in result.samples_with_valid_scores
            assert result.samples_with_valid_scores[m] == 3

    def test_score_batch_length_mismatch_raises(self, checkpoint_dir):
        evaluator = RobertaTraceEvaluator(model_path=checkpoint_dir)
        with pytest.raises(ValueError):
            evaluator.score_batch(["q1"], [["c1"], ["c2"]], ["a1"])

    def test_functional_entry_point_env_var(self, monkeypatch, checkpoint_dir):
        monkeypatch.setenv("ROBERTA_TRACE_MODEL_PATH", checkpoint_dir)
        samples = [{"utilization": 0.5, "relevance": 0.5, "adherence": 1.0, "completeness": 0.5}]
        fake_logits = _make_fake_torch_logits(samples)
        _patch_torch_and_transformers(monkeypatch, fake_logits)

        result = evaluate_with_roberta(["q"], ["gt"], ["a"], [["c"]])
        assert result.error is None
        assert "utilization" in result.metric_means


# ─── model construction (mocked torch / transformers) ───────────────


class TestRobertaTraceClassifier:
    def test_cannot_load_without_checkpoint_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            RobertaTraceClassifier.from_pretrained(str(tmp_path))

    def test_save_writes_config_and_heads_files(self, tmp_path, monkeypatch):
        """Verify save() emits trace_config.json and trace_heads.pt.

        We bypass the heavy torch.nn model construction by stubbing the
        internal ``build`` to return a fake model object that exposes the
        four head state-dicts and an encoder with ``save_pretrained``.
        """
        import json

        # Build a fake torch that actually writes a file when ``save`` runs
        # so we can assert on the resulting artefact.
        class _FakeTorch:
            @staticmethod
            def save(sd, path):
                with open(path, "wb") as fh:
                    fh.write(b"fake-heads")

        # Construct classifier without calling build(); splice in fake state.
        clf = RobertaTraceClassifier(model_name="roberta-base")

        fake_head = MagicMock()
        fake_head.state_dict.return_value = {"w": "fake"}
        fake_encoder = MagicMock()

        class _FakeModel:
            encoder = fake_encoder
            utilization_head = fake_head
            relevance_head = fake_head
            adherence_head = fake_head
            completeness_head = fake_head

        fake_model = _FakeModel()
        clf._model = fake_model  # type: ignore[attr-defined]
        clf._torch = _FakeTorch

        out_dir = tmp_path / "out"
        clf.save(str(out_dir))

        config_path = out_dir / "trace_config.json"
        heads_path = out_dir / "trace_heads.pt"
        assert config_path.exists()
        assert heads_path.exists()
        cfg = json.loads(config_path.read_text())
        assert cfg["num_bins"] == DEFAULT_NUM_BINS
        assert set(cfg["metrics"]) == set(TRACE_METRICS)
        fake_encoder.save_pretrained.assert_called_once_with(str(out_dir))


class _FakeLinear:
    def state_dict(self):
        return {}

    def load_state_dict(self, sd):
        return MagicMock()

    def __call__(self, *a, **k):
        return MagicMock()


# ─── registry ───────────────────────────────────────────────────────


class TestRegistry:
    def test_roberta_trace_registered_by_default(self):
        from benchmark.roberta_evaluator.register import available_evaluators

        # Importing the package triggers register("roberta_trace", ...).
        assert "roberta_trace" in available_evaluators()

    def test_get_evaluator_unknown_raises(self):
        with pytest.raises(KeyError):
            get_evaluator("does_not_exist")

    def test_register_and_retrieve(self):
        factory = MagicMock()
        register("custom_test_evaluator", factory)
        get_evaluator("custom_test_evaluator", foo="bar")
        factory.assert_called_once_with(foo="bar")


# ─── slow integration: tiny training run ────────────────────────────


@pytest.mark.slow
def test_training_smoke(tmp_path, monkeypatch):
    """Train the real (tiny) model on 10 fake examples.

    Skipped unless ``transformers`` and ``torch`` are installed and the
    user explicitly runs slow tests via ``pytest -m slow``.
    """
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from benchmark.roberta_evaluator.dataset import TraceExample
    from benchmark.roberta_evaluator.train import main as train_main

    # Stub load_ragbench_split to return 10 deterministic examples so we
    # don't hit the network.
    def _fake_examples(n, split):
        for i in range(n):
            yield TraceExample(
                question=f"Question {i}?",
                context=f"Context number {i} is here.",
                response=f"Answer to {i}.",
                labels={
                    "utilization": (i % 5) / 4.0,
                    "relevance": ((i + 1) % 5) / 4.0,
                    "adherence": float(i % 2),
                    "completeness": ((i + 2) % 5) / 4.0,
                },
            )

    import benchmark.roberta_evaluator.train as train_mod

    monkeypatch.setattr(
        train_mod,
        "load_ragbench_split",
        lambda subsets, split: _fake_examples(10, split),
    )

    out_dir = tmp_path / "trained"
    rc = train_mod.main(
        [
            "--output_dir", str(out_dir),
            "--model_name", "hf-internal-testing/tiny-random-roberta",
            "--epochs", "1",
            "--batch_size", "4",
            "--eval_batch_size", "4",
            "--max_train_examples", "8",
            "--max_eval_examples", "2",
            "--device", "cpu",
            "--log_level", "WARNING",
        ]
    )
    assert rc == 0
    assert (out_dir / "trace_heads.pt").exists()
    assert (out_dir / "trace_config.json").exists()
    assert (out_dir / "eval_metrics.json").exists()
