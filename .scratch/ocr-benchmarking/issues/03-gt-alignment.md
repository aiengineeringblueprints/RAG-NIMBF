# OCR-03: GT↔prediction alignment (vendored OmniDocBench quick_match)

**What to build:** Ground-truth ↔ prediction alignment for page-level scoring, vendoring OmniDocBench's quick_match semantics (Apache-2.0): paragraph segmentation of predictions, truncation/merging via adjacency matching, so text metrics compare aligned paragraph pairs rather than raw whole-page strings. The matching algorithm and its version are recorded into run metadata.

**Blocked by:** OCR-02 (text metrics — alignment feeds their inputs).

**Status:** ready-for-agent

- [ ] quick_match semantics vendored with license attribution; scores on fixture docs match OmniDocBench behavior on equivalent input
- [ ] Truncation/merge cases covered by alignment tests (split paragraphs, merged paragraphs, missing predictions)
- [ ] Matching algorithm name + version emitted into run metadata; surfaced in reports
