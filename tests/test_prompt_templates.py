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

    def test_finqa_exists(self):
        t = get_template("finqa")
        assert t.name == "finqa"
        assert t.system_prompt
        assert "{context}" in t.human_template
        assert "{question}" in t.human_template

    def test_finqa_has_final_instruction(self):
        t = get_template("finqa")
        assert "FINAL:" in t.system_prompt

    def test_registry_has_all(self):
        assert set(BUILTIN_TEMPLATES.keys()) == {
            "concise", "detailed", "finqa",
            "minimal", "no_context", "cot", "cited",
            "abstain_strict", "question_first", "chain_of_note",
        }

    @pytest.mark.parametrize("name", [
        "minimal", "cot", "cited", "abstain_strict", "question_first", "chain_of_note",
    ])
    def test_rag_templates_include_context_and_question(self, name):
        t = get_template(name)
        assert t.name == name
        assert t.system_prompt
        assert "{context}" in t.human_template
        assert "{question}" in t.human_template

    def test_no_context_omits_context(self):
        t = get_template("no_context")
        assert "{context}" not in t.human_template
        assert "{question}" in t.human_template

    def test_question_first_puts_question_before_and_after_context(self):
        human = get_template("question_first").human_template
        first_q = human.index("{question}")
        ctx = human.index("{context}")
        last_q = human.rindex("{question}")
        assert first_q < ctx < last_q

    @pytest.mark.parametrize("name", ["finqa", "cot", "chain_of_note"])
    def test_final_line_templates(self, name):
        t = get_template(name)
        assert t.final_answer_line is True
        assert "FINAL:" in t.system_prompt

    def test_cited_numbers_contexts(self):
        t = get_template("cited")
        assert t.number_contexts is True
        assert t.strip_citations is True

    def test_plain_templates_have_no_flags(self):
        t = get_template("concise")
        assert not (t.final_answer_line or t.number_contexts or t.strip_citations)
