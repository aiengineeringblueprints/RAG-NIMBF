# OCR-02: Parsing metrics — CER/WER and normalized edit distance

**What to build:** The text-metric core of the parsing metric family, in the new parsing module group: CER, WER (jiwer) and normalized edit distance (RapidFuzz), computed per page and per document from a ParseResult against ground truth, with OmniDocBench-style normalization (case, whitespace, unicode folding) applied before distance.

**Blocked by:** OCR-01 (Parser adapter protocol — needs the ParseResult types), campaign #4.

**Status:** ready-for-agent

- [ ] CER/WER + normalized edit distance implemented as pure functions over prediction/GT text
- [ ] Normalization edge cases covered: case, whitespace, unicode, blank pages
- [ ] Per-page and per-document aggregation with the same mean/stats pattern as existing metrics
- [ ] Fixture tests with hand-computed expected scores; jiwer as reference oracle where applicable
