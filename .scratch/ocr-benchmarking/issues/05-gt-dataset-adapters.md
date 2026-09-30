# OCR-05: Document ground-truth dataset adapters (OmniDocBench, DP-Bench, olmOCR-Bench)

**What to build:** Ground-truth loading for the three v1 document benchmarks through the existing dataset layer: OmniDocBench v1.6 (per-page JSON with per-element GT; research-only license flagged in config), DP-Bench (MIT; HTML + Markdown GT), olmOCR-Bench (Apache-2.0; unit-test specs reduced to GT text where applicable or explicitly marked unsupported for GT scoring). License flags are part of the loaded dataset metadata and surface in run metadata.

**Blocked by:** campaign #4 (dataset layer settles with the worker-only loop). Otherwise none.

**Status:** ready-for-agent

- [ ] OmniDocBench pages load into the normalized sample/document format with text and table GT
- [ ] DP-Bench loads; olmOCR-Bench loads with an explicit capability statement (what can and cannot be GT-scored)
- [ ] License metadata present on every loaded dataset and propagated to run metadata; research-only flag testable
- [ ] Tiny local fixtures stand in for downloads in tests (no network in CI)
