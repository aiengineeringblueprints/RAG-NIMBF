"""Tests for benchmark.prompt_templates — template registry and lookup."""

import pytest

from benchmark.prompt_templates import get_template, BUILTIN_TEMPLATES


class TestGetTemplate:
    def test_concise_exists(self):
        t = get_template("concise")
        assert t.name == "concise"
        assert t.system_prompt
        assert "{context}" in t.human_template
        assert "{question}" in t.human_template

    def test_detailed_exists(self):
        t = get_template("detailed")
        assert t.name == "detailed"
        assert t.system_prompt
        assert "{context}" in t.human_template
        assert "{question}" in t.human_template

    def test_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown prompt template"):
            get_template("nonexistent")

    def test_registry_has_both(self):
        assert set(BUILTIN_TEMPLATES.keys()) == {"concise", "detailed"}
