# OCR-04: TEDS table metric with markdown→HTML conversion

**What to build:** Table-parsing quality scoring via TEDS, vendoring the Apache-2.0 PubTabNet/OmniDocBench implementation, plus the markdown-table → HTML conversion so Markdown-only parsers score on tables. Scored on aligned table regions where GT provides table structure; reported alongside text metrics.

**Blocked by:** OCR-02 (text metrics — shared metric infrastructure), OCR-03 (alignment — tables are matched regions).

**Status:** ready-for-agent

- [ ] TEDS + TEDS-S vendored with license attribution; reference fixtures score identically to the upstream implementation
- [ ] Markdown tables (pipe/HTML) convert to comparable HTML before TEDS
- [ ] Documents without tables score without error; per-table scores aggregate to a document table score
