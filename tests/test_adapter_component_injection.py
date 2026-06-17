"""Integration-level test: main.py wires build_components -> set_components."""
from unittest.mock import MagicMock, patch

import pytest

from benchmark.adapters import ComponentBundle
from benchmark.adapters.base import RagSystemOutput

# main.py imports mlflow at module load; skip this whole module if the
# optional dep is missing in the test environment.
pytest.importorskip("mlflow")


def _make_legacy_adapter():
    class LegacyAdapter:
        name = "legacy"
        def __init__(self, cfg): pass
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="ok")
    return LegacyAdapter(MagicMock())


def _make_full_adapter(captured):
    class FullAdapter:
        name = "fake"
        def __init__(self, cfg): self.cfg = cfg
        def supports_components(self):
            return {"llm": True, "chunker": True}
        def set_components(self, bundle: ComponentBundle):
            captured["bundle"] = bundle
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="ok")
    return FullAdapter(MagicMock())


def _base_cfg(**overrides):
    cfg = MagicMock()
    cfg.benchmark_stage = "all"
    cfg.name = "test"
    cfg.retrieval_mode = "retrieval"
    cfg.rag_adapter_accepts = ""
    cfg.rag_system_adapter = "fake"
    cfg.ragas_enabled = False
    cfg.custom_metrics_enabled = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def test_inject_components_called_when_adapter_supports():
    captured: dict = {}
    fake_bundle = ComponentBundle(llm="FAKE_LLM", chunker="FAKE_CHUNKER")

    with patch("main.get_rag_adapter", return_value=_make_full_adapter(captured)), \
         patch("main.build_components", return_value=fake_bundle) as bc_mock:
        import main
        cfg = _base_cfg(rag_adapter_accepts="llm,chunker")
        data = [{"question": "q", "ground_truth": "g", "context": "c"}]
        main.run_single_benchmark(cfg, data)

    assert "bundle" in captured
    assert captured["bundle"].llm == "FAKE_LLM"
    bc_mock.assert_called_once()
    assert bc_mock.call_args.args[0].rag_adapter_accepts == "chunker,llm"
    assert cfg.rag_adapter_accepts == "llm,chunker"


def test_set_components_not_called_when_adapter_lacks_method():
    with patch("main.get_rag_adapter", return_value=_make_legacy_adapter()), \
         patch("main.build_components", return_value=ComponentBundle()) as bc_mock:
        import main
        cfg = _base_cfg(rag_adapter_accepts="")
        data = [{"question": "q", "ground_truth": "g", "context": "c"}]
        main.run_single_benchmark(cfg, data)

    bc_mock.assert_not_called()
