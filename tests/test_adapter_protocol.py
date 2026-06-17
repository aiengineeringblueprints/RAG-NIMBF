"""Protocol is structural — verify optional methods are truly optional."""

from benchmark.adapters import (
    ComponentBundle,
    RagSystemOutput,
    build_components,
)
from benchmark.adapters.base import RagSystemAdapter


def test_protocol_optional_methods_absent_on_legacy_adapter():
    class LegacyAdapter:
        name = "legacy"
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="x")

    inst = LegacyAdapter()
    assert not hasattr(inst, "supports_components")
    assert not hasattr(inst, "set_components")


def test_protocol_accepts_full_implementation():
    class FullAdapter:
        name = "full"
        def supports_components(self):
            return {"llm": True, "chunker": False}
        def set_components(self, bundle: ComponentBundle):
            self.bundle = bundle
        def prepare(self, config, data, corpus=None): pass
        def answer(self, sample, config):
            return RagSystemOutput(answer="x")

    inst = FullAdapter()
    inst.set_components(ComponentBundle(llm="FAKE"))
    assert inst.bundle.llm == "FAKE"
    assert inst.supports_components() == {"llm": True, "chunker": False}


def test_re_exports():
    assert ComponentBundle is not None
    assert callable(build_components)
    # RagSystemAdapter remains exported from base for type hints
    assert RagSystemAdapter is not None
