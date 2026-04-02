# RAG Answer Quality Analysis

**Date:** 2026-04-02
**Config:** recursive chunking (1000/200), nomic-embed-text, gemma3:4b
**Dataset:** FinQA subset, 50 samples
**Result file:** `results/run1/benchmark_20260402_104806.json`

---

## RAGAS Scores Overview

| Metric              | Score |
|---------------------|-------|
| Faithfulness        | 0.598 |
| Answer Relevancy    | 0.708 |
| Context Precision   | 0.426 |
| Context Recall      | 0.544 |

---

## Problem 1: Cross-Company Retrieval Contamination (Highest Impact)

**Symptom:** Many retrieved chunks come from the wrong company entirely.

**Examples:**
- Q about AES Corporation debt → retrieves Goldman Sachs, PPG, Motorola chunks
- Q about Lockheed Martin expenses → retrieves only Goldman Sachs data
- Q about Dollar General LIFO cost → retrieves Motorola, PNC, JPMorgan chunks
- Q about Under Armour working capital → retrieves Goldman Sachs data
- Q about Marathon Oil development costs → retrieves Motorola and other unrelated chunks

**Root Cause:** The embedding model (`nomic-embed-text`) does not distinguish between similar financial document structures. Many 10-K filings share boilerplate language, so semantically similar chunks from different companies get mixed together. With 320 chunks from potentially many documents, cross-contamination is severe.

**Possible Fixes:**
- Add company name as metadata to each chunk and filter retrieval by company
- Use metadata-enriched chunk prefixes (e.g., prepend "Company: Goldman Sachs. " to each chunk)
- Use a stronger embedding model with better financial domain understanding
- Implement a re-ranking step that scores chunks for company relevance

---

## Problem 2: LLM Cannot Perform Numerical Calculations

**Symptom:** Gemma 3 4B identifies the right numbers but refuses or fails to compute the final answer.

**Examples:**

| Question asks for | Model gives | Ground truth |
|---|---|---|
| Percentage of Euro impact | "$35 million" | 26.9% |
| Avg total assets | States both values separately | (911507+938555)/2 = 925031 |
| Shareholders' equity change | "$75716 to $78467" | 2751 |
| Avg residential mortgage loans | "$9557M and $6337M" | 7947.0 |
| Devon Energy debt ratio | "does not contain" | 3189/500 = 6.378 |
| Percentage of lease payments | "$136,462" | 11.5% |
| Ratio of pre-tax gain to carrying | "$6M to $4.7B" | 1.489 |
| Republic Services restricted cash % | "$78.6M of $108.1M" | 72.7% |

**Root Cause:** Gemma 3 4B is too small for the multi-step numerical reasoning FinQA requires. The model can extract facts but cannot reliably perform division, subtraction, averaging, or percentage calculations.

**Possible Fixes:**
- Use a larger generation model (7B+ parameters) with better reasoning
- Add a post-processing calculator step: extract numbers from LLM output, compute with Python
- Add a tool-use / function-calling pattern where the LLM can invoke arithmetic
- Chain-of-thought prompting to force step-by-step calculation

---

## Problem 3: Hallucination with Wrong Company Data

**Symptom:** When retrieval mixes in chunks from other companies, the model uses the wrong numbers in its answer.

**Examples:**
- Q about American Water Works liabilities → Uses Goldman Sachs total liabilities ($833.04B)
- Q about Aon share purchase price → Uses share repurchase data from a different company ($58.15)
- Q about SL Green restricted stock → Uses tax position data instead ($118314/$158578)
- Q about Goldman Sachs 2016 net revenues → Reports 2017 figure ($6.22B vs actual $5.81B)

**Root Cause:** Direct consequence of Problem 1. When wrong-company chunks are retrieved, the model cannot distinguish which numbers belong to the target company.

**Possible Fixes:**
- Fix retrieval (see Problem 1) — this is a downstream symptom
- Add prompt instruction: "Only use information from the company mentioned in the question"
- Post-check answers against known company identifiers

---

## Problem 4: Table Chunking Destroys Structure

**Symptom:** Tables split across chunks lose their headers or column context.

**Root Cause:** With chunk_size=1000 and overlap=200, financial tables get split at arbitrary boundaries. A table header may be in chunk N while the data rows are in chunk N+1.

**Possible Fixes:**
- Implement table-aware chunking that keeps tables intact within a single chunk
- Increase overlap specifically for table regions
- Convert tables to natural language summaries before chunking
- Merge adjacent chunks that contain partial tables

---

## Problem 5: Prompt Does Not Instruct Calculation

**Symptom:** Model gives up or just restates values without computing.

**Possible Fixes:**
- Add explicit instruction: "You MUST calculate the final numerical answer. Do not just restate the values from the text."
- Provide few-shot examples showing step-by-step calculation
- Separate "find the relevant numbers" from "compute the answer" into two LLM calls

---

## Recommended Priority

| Priority | Fix | Expected Impact |
|---|---|---|
| **1** | Metadata filtering on company name | Eliminates cross-company contamination, largest quality gain |
| **2** | Upgrade generation model (7B+) or add calculator post-processing | Enables numerical reasoning |
| **3** | Table-aware chunking | Preserves table structure |
| **4** | Prompt engineering for calculation | Low effort, moderate gain |
| **5** | Increase top-k and add re-ranking | Improves recall of relevant chunks |
