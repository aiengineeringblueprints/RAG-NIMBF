# Prompt Templates

Sources:

- [benchmark/prompt_templates/__init__.py](../benchmark/prompt_templates/__init__.py)
- [benchmark/prompt_templates/concise.py](../benchmark/prompt_templates/concise.py)
- [benchmark/prompt_templates/detailed.py](../benchmark/prompt_templates/detailed.py)
- [benchmark/prompt_templates/finqa.py](../benchmark/prompt_templates/finqa.py)

Built-in templates:

- `concise`: short answer behavior.
- `detailed`: fuller answer behavior.
- `finqa`: finance-oriented numeric answer behavior.

`get_template(name)` validates template names and returns a `PromptTemplate` with:

- `name`
- `system_prompt`
- `human_template`

Prompt templates are part of the benchmark grid through `PROMPT_TEMPLATES`.

When adding a template:

1. Add a module or definition under `benchmark/prompt_templates/`.
2. Register it in `BUILTIN_TEMPLATES`.
3. Add tests in `tests/test_prompt_templates.py`.
4. Update [[Configuration Reference]] and this note.

Related notes:

- [[Generation Layer]]
- [[Benchmark Pipeline]]

