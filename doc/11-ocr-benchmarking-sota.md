# 11 — OCR & Document-Parsing Evaluation: State of the Art (2024–2026)

*Research note, 2026-09-16. Primary sources only (arXiv pages, official GitHub repos, HuggingFace dataset cards). Goal: ground the integration of OCR/document-parsing benchmarking into this framework (adapters in `benchmark/adapters/`, resumable worker, `results/runN` + MLflow reporting).*

---

## 1. Metrics: what is standard, what is specialized, and how each is computed

### 1.1 Text quality (table-stakes)

| Metric | Definition / computation | Reference implementation |
|---|---|---|
| **CER / WER** | Levenshtein edit distance between reference and hypothesis, normalized by reference length (char or word level). Standard in ASR and OCR since Tesseract-era evaluations. | [`jiwer`](https://github.com/jitsi/jiwer) (Apache-2.0): implements WER, CER, MER, WIL, WIP; edit distance computed via [RapidFuzz](https://github.com/maxbachmann/RapidFuzz) (C++). As of v4.0, empty-reference behavior is defined (`wer('','')==0`), which matters for blank-page tests. |
| **Normalized Edit Distance (1−ED)** | Character edit distance after normalization (case, whitespace, unicode folding); OmniDocBench's primary text metric, reported as `TextEdit↓` (lower = better). OmniDocBench normalizes GT and predictions before distance computation (see changelog entry 2025/01/16 on normalized table GT/preds). | [OmniDocBench repo](https://github.com/opendatalab/OmniDocBench) (`Edit_dist` metric, Apache-2.0). |
| **ANLS** | Average Normalized Levenshtein Similarity: `1 − NL(pred, gt)` averaged over OCR'd ground truths, with a 0.5 score floor; defined in the [DocVQA paper (arXiv:2007.00398, WACV 2021)](https://arxiv.org/abs/2007.00398) and used by the official [DocVQA leaderboard](https://docvqa.org) (RRC portal scoring). | Official scoring on the RRC server; `lmms-eval` ships DocVQA tasks ([utils.py](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/docvqa/utils.py)) that emit RRC submission files for the test split. |

**Table-stakes verdict:** CER/WER-style normalized edit distance is the universal floor for text. Every 2024–2026 document-parsing benchmark reports it in some normalized form.

### 1.2 Tables

- **TEDS / TEDS-S** — Tree-Edit-Distance-based Similarity between predicted and ground-truth HTML table trees; TEDS-S is structure-only. Introduced with **PubTabNet** ([arXiv:1911.10683](https://arxiv.org/abs/1911.10683), ECCV 2020; 568k table images with HTML ground truth). Reference implementation: [ibm-aur-nlp/PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet). OmniDocBench reuses this implementation (explicitly credited in its [README acknowledgement](https://github.com/opendatalab/OmniDocBench#acknowledgement)) and converts predictions to comparable HTML via [LaTeXML](https://github.com/brucemiller/LaTeXML) and a markdown→HTML table converter ([intsig-textin/markdown_tester](https://github.com/intsig-textin/markdown_tester)).
- **GriTS** (grid table similarity) — evaluates the predicted table *as a matrix* (2D most-similar-substructure), unifying topology/location/content subtasks ([arXiv:2203.12555](https://arxiv.org/abs/2203.12555), Microsoft). Implementation: [microsoft/table-transformer](https://github.com/microsoft/table-transformer). Used mainly in the table-structure-recognition literature; TEDS remains the de-facto headline metric in document-parsing benchmarks.

### 1.3 Layout detection

- **COCO-style mAP/mAR at IoU thresholds** is standard. [DocLayNet](https://arxiv.org/abs/2206.01062) (KDD 2022) established the protocol for general documents: 80,863 manually annotated pages, 11 classes, COCO format; best models land ~10 mAP points below human inter-annotator agreement. OmniDocBench's `COCODet` metric consumes a flat JSON of `{image_name, bbox, category_id, score}` and maps both GT (28 block + 4 span categories) and prediction categories into a shared eval taxonomy via configurable `gt_cat_mapping`/`pred_cat_mapping` tables ([README, Layout Detection](https://github.com/opendatalab/OmniDocBench#layout-detection)).

### 1.4 Reading order

- **Reading-order edit distance** — OmniDocBench annotates an `order` integer per block and evaluates predicted sequence against it with edit distance (`reading_order: metric: [Edit_dist]` in `end2end.yaml`). This is currently the only widely-cited reading-order metric in doc-parsing benchmarks ([README](https://github.com/opendatalab/OmniDocBench#end-to-end-evaluation)). Older layout work used BLEU over order sequences (e.g., LayoutReader lineage) but it has not carried into the 2024–2026 doc-parsing harnesses.

### 1.5 Formulas

- **Legacy:** BLEU + normalized edit distance over LaTeX strings (token/char level). Both are known to be unfair: the same formula has many equivalent LaTeX renderings.
- **CDM (Character Detection Matching)** — CVPR 2025 ([arXiv:2409.03643](https://arxiv.org/abs/2409.03643)). Renders *both* predicted and ground-truth LaTeX to images, then does spatially-aware character-level visual matching; reports `CDM` and `ExpRate@CDM`. Code lives in [opendatalab/UniMERNet/tree/main/cdm](https://github.com/opendatalab/UniMERNet/tree/main/cdm). OmniDocBench integrated CDM directly into its metric suite in **v1.5** (Sep 2025) and in **v1.6** (Apr 2026) replaced the Node.js/KaTeX dependency with a pure-Python re-implementation (~3× faster). CDM still requires TeX Live 2025, ImageMagick ≥7.1.1-47 and Ghostscript 9.55 for rendering; OmniDocBench ships a Docker image pinning this stack ([README, Evaluation setup](https://github.com/opendatalab/OmniDocBench#environment-setup-and-running)).
- The OmniDocBench v1.5+ **Overall** score is `((1 − text Edit distance) × 100 + table TEDS + formula CDM) / 3` ([README, Updates](https://github.com/opendatalab/OmniDocBench#updates)) — i.e., text, tables and formulas weigh equally.

### 1.6 Charts

- ⚠ **Unsettled.** No doc-parsing benchmark has a chart-parsing track with a converged metric (OmniDocBench v1.7 does not evaluate charts). The nearest standardized evaluation is chart→code/image similarity in [ChartMimic](https://arxiv.org/abs/2406.09961) (ICLR 2025; 4,800 chart–instruction–code triplets, multi-level metrics on code and rendered output). Treat chart parsing as specialized/out-of-scope for a first integration.

### 1.7 Unit-test-style scoring (new 2025 pattern)

- **olmOCR-Bench** ([arXiv:2502.18443](https://arxiv.org/abs/2502.18443); benchmark in [allenai/olmocr/olmocr/bench](https://github.com/allenai/olmocr/tree/main/olmocr/bench)) abandons full-text alignment: 1,400 curated PDFs, **7,000+ binary unit tests** (anchor-text presence, table constraints, reading-order anchors, header/footer removal) executed against the emitted Markdown; score = % of tests passed. Category view: ArXiv math, old scans with math, tables, old scans, headers/footers, multi-column, long tiny text, base. The 2026 French PDF→Markdown benchmark ([arXiv:2602.11960](https://arxiv.org/abs/2602.11960)) adopted the same pattern explicitly to stop penalizing presentation-only variance.

> ⚠ **Fast-moving/unsettled — matching & aggregation.** *How* a markdown prediction is aligned to ground truth before computing edit distance is still evolving: OmniDocBench changed its matching algorithm twice (hybrid text↔formula matching in v1.5; **MGAM** — multi-granularity adaptive matching, searching segmentation granularity only on the prediction side — in v1.6) and scores are **not comparable across benchmark versions** (v1.0 → v1.5 → v1.6/v1.7 each changed the dataset and/or scoring). Always pin the version in experiment metadata.

---

## 2. Benchmark datasets

| Benchmark | Task type | Size | License | HuggingFace | Ground-truth format |
|---|---|---|---|---|---|
| **OmniDocBench** ([repo](https://github.com/opendatalab/OmniDocBench), [arXiv:2412.07626](https://arxiv.org/abs/2412.07626), CVPR 2025) | PDF → Markdown end-to-end + module evals (text OCR, layout, tables, formulas, reading order), EN+ZH | v1.0: 981 pages → v1.5: +374 → v1.6: +296 = **1,651 pages**, 9–10 document sources, 28 block + 4 span categories | Code Apache-2.0; **data "research purposes only, not commercial"** per repo copyright statement | [opendatalab/OmniDocBench](https://huggingface.co/datasets/opendatalab/OmniDocBench) | JSON per page: `layout_dets[]` with `poly` coords, `order` (reading order), per-element `text` / `latex` / `html` (tables in LaTeX *and* HTML), attribute tags, parent/truncation relations |
| **olmOCR-Bench** ([in olmOCR repo](https://github.com/allenai/olmocr/tree/main/olmocr/bench), [arXiv:2502.18443](https://arxiv.org/abs/2502.18443), [arXiv:2510.19817](https://arxiv.org/abs/2510.19817)) | PDF → Markdown, English, unit-test scoring | 1,400 PDFs, 7,000+ test cases, 8 categories | Apache-2.0 (model/code/bench "permissive open licenses") | distributed inside the repo (`pip install olmocr[bench]`) | PDFs + per-page unit-test specs (anchors, structural checks) — **no plain-text GT** |
| **OCRBench v1** ([arXiv:2305.07895](https://arxiv.org/abs/2305.07895), Sci China Inf Sci) | 5 LMM task families: text recognition, scene-text VQA, doc VQA, KIE, HMER; aggregate "Total Points" score | 29 source datasets | — | [echo840/OCRBench](https://huggingface.co/datasets/echo840/OCRBench) | QA pairs (text answers; custom per-question scoring) |
| **OCRBench v2** ([arXiv:2501.00321](https://arxiv.org/abs/2501.00321)) | Text-centric LMM benchmark incl. text localization & reasoning, EN+ZH, 31 scenarios | **10,000** human-verified QA pairs + private 1,500-image test set; most LMMs score <50/100 | MIT (HF mirror) | official code [Yuliang-Liu/MultimodalOCR](https://github.com/Yuliang-Liu/MultimodalOCR); mirrors: [ling99/OCRBench_v2](https://huggingface.co/datasets/ling99/OCRBench_v2) (MIT), [lmms-lab/OCRBench-v2](https://huggingface.co/datasets/lmms-lab/OCRBench-v2) | QA pairs; localization tasks with polygons |
| **CC-OCR** ([arXiv:2412.02210](https://arxiv.org/abs/2412.02210), Alibaba/Qwen) | 4 tracks: multi-scene reading, multilingual reading, **document parsing**, KIE | 39 subsets, **7,058** annotated images, 41% from real applications | MIT | [wulipc/CC-OCR](https://huggingface.co/datasets/wulipc/CC-OCR) (TSV for VLMEvalKit) | Per-task text/markdown GT |
| **CC-OCR V2** ([HF card](https://huggingface.co/datasets/Eioss/CC-OCR-V2), 2026) | 5 OCR-centric tracks, enterprise-document focus | 7,093 high-difficulty samples | MIT | [Eioss/CC-OCR-V2](https://huggingface.co/datasets/Eioss/CC-OCR-V2) | QA/text GT (details only via card; see §6 verification notes) |
| **DocVQA** ([arXiv:2007.00398](https://arxiv.org/abs/2007.00398), WACV 2021; [docvqa.org](https://docvqa.org), [RRC portal](https://rrc.cvc.uab.es/?ch=17)) | Document VQA (downstream proxy for parsing quality) | 50,000 questions / 12,000+ pages | research (UC authority) | official via RRC portal | QA pairs; scored with ANLS |
| **Fox** ([arXiv:2405.14295](https://arxiv.org/abs/2405.14295), [repo](https://github.com/ucaslcl/Fox)) | Dense-page bilingual OCR + region OCR/translation/summary + multi-page & cross-page VQA; 9 subtasks | page/region/line/multi-page subsets (EN+ZH) | Code Apache-2.0; **data CC-BY-NC 4.0** | [ucaslcl/Fox_benchmark_data](https://huggingface.co/datasets/ucaslcl/Fox_benchmark_data) | JSON conversation format (`image`, prompt, GT label); metrics: BLEU, METEOR, F1/P/R, edit distance, ROUGE, accuracy |
| **DP-Bench** (Upstage, [HF card](https://huggingface.co/datasets/upstage/dp-bench), Oct 2024) | PDF/image → HTML + Markdown parsing; explicit "preprocessor for RAG" framing | small (document-level set) | MIT | [upstage/dp-bench](https://huggingface.co/datasets/upstage/dp-bench) | HTML + Markdown; scored with TEDS-family metrics (and NID, per third-party reports, e.g. [NovaLAD, arXiv:2603.00122](https://arxiv.org/abs/2603.00122)) |
| **DocLayNet** ([arXiv:2206.01062](https://arxiv.org/abs/2206.01062), KDD 2022) | Layout detection/segmentation | **80,863** pages, 11 classes, 6 document categories | "other" on HF (CDLA-Permissive family, IBM) | [docling-project/DocLayNet](https://huggingface.co/datasets/docling-project/DocLayNet), v1.2 with embedded PDFs (2025): [docling-project/DocLayNet-v1.2](https://huggingface.co/datasets/docling-project/DocLayNet-v1.2) | COCO bboxes |
| **PubTabNet** ([arXiv:1911.10683](https://arxiv.org/abs/1911.10683), ECCV 2020) | Table image → HTML (TEDS evaluation) | **568k** table images | Apache-2.0 ([repo](https://github.com/ibm-aur-nlp/PubTabNet)) | (community copies; official = GitHub) | HTML table trees |
| **M3DocVQA** (with [M3DocRAG, arXiv:2411.04952](https://arxiv.org/abs/2411.04952)) | Open-domain multimodal DocVQA (RAG-style) | 3,000+ PDFs / 40,000+ pages | — | [project page](https://m3docrag.github.io) | QA pairs with evidence pages |

Notes:
- "DocTabNet" does not exist as a separate benchmark — PubTabNet is the IBM table dataset commonly meant.
- License traps for production use: OmniDocBench data (research-only) and Fox data (CC-BY-NC) are **not** clean for commercial/corporate benchmarking; olmOCR-Bench, CC-OCR, OCRBench v2 mirrors, DP-Bench and DocLayNet are permissive.

---

## 3. What is being compared, and the de-facto system interface

Three comparison classes coexist in 2024–2026 literature:

1. **Raw OCR models (image → text/regions).** Compared at region/page/line level, e.g. OmniDocBench's text-OCR module evaluation (PaddleOCR, Tesseract, EasyOCR vs VLMs, with per-attribute breakdowns), CC-OCR's multi-scene/multilingual reading tracks, Fox's line/region/page OCR subtasks.
2. **End-to-end document-parsing systems (PDF → Markdown/structured).** The dominant setting of 2025–2026: OmniDocBench end2end, olmOCR-Bench, DP-Bench. OmniDocBench's leaderboard explicitly groups **Pipeline Tools** (Marker, MinerU-Pipeline), **Specialized VLM parsers** and **General VLMs**.
3. **VLM-based parsers.** The clear trend: small specialized VLMs (0.8–4B) now top the leaderboards. On OmniDocBench v1.6_full (README as of 2026-09): TeleOCR 1.2B **96.91** Overall, OvisOCR2 0.8B 96.47, PaddleOCR-VL-1.6 0.9B 96.34, MinerU2.5 95.75 — ahead of Gemini 3 Pro (92.91), GPT-5.2 (86.59) and pipeline tool Marker (78.44). olmOCR-Bench shows the same ordering (PaddleOCR-VL 80.0, olmOCR v0.4.0 82.4, Chandra 83.1, vs Mistral OCR API 72.0, Marker 1.10.1 76.1, MinerU 2.5.4 75.2). ⚠ Leaderboard ordering is volatile — new parsers appear monthly (the OmniDocBench README added 10+ systems between Mar and Sep 2026).

**De-facto system interface.** Converged and simple:

- **Input:** a PDF or page image(s).
- **Output:** one **Markdown** file per page, optionally plus a **structured JSON** with per-block `bbox`, category, text, and table HTML / formula LaTeX.
  - OmniDocBench's prediction contract is literally "a folder of per-page `.md` files named after the page images" ([README](https://github.com/opendatalab/OmniDocBench#end-to-end-evaluation)), with a separate COCO-ish JSON for layout predictions.
  - olmOCR CLI: `olmocr workspace --markdown --pdfs *.pdf` → Markdown files ([README](https://github.com/allenai/olmocr)).
  - Docling converts PDFs into a structured `DoclingDocument` (Markdown/JSON views) using DocLayNet-based layout + TableFormer ([arXiv:2408.09869](https://arxiv.org/abs/2408.09869), MIT).
  - GOT-OCR 2.0 frames the output space as "plain or formatted results (markdown/tikz/smiles/kern)" ([arXiv:2409.01704](https://arxiv.org/abs/2409.01704)).
- **Service interface:** OpenAI-compatible HTTP endpoints are becoming the standard serving layer — olmOCR supports `--server <openai-compatible-url>`, and the community EvalScope integration runs OmniDocBench against "OpenAI-compatible model endpoints with standardized predictions, metrics, and reports" ([OmniDocBench README, 2026/07/27 update](https://github.com/opendatalab/OmniDocBench#updates)).

For this framework that means: an OCR/parser system is just another HTTP (or in-process Python) service with a thin request/response contract — the same adapter shape as the existing RAG system adapters.

---

## 4. Integration patterns of existing harnesses

### OmniDocBench harness (the reference implementation)

- **Config-driven evaluation**: a YAML selects metrics per category (`text_block: [Edit_dist, BLEU, METEOR]`, `display_formula: [Edit_dist, CDM]`, `table: [TEDS, Edit_dist]`, `reading_order: [Edit_dist]`), the matching method, and page-attribute filters ([end2end.yaml](https://github.com/opendatalab/OmniDocBench/blob/main/configs/end2end.yaml)).
- **Ground-truth alignment** is the hard, bespoke part:
  - three matching modes — `no_split` (whole-page string), `simple_match` (split on double newlines, 1:1 match), `quick_match` (paragraph segmentation + truncation/merging via "Adjacency Search Match") — recommended default `quick_match`;
  - since v1.5, **hybrid matching** lets text and formula blocks match each other (mitigates unicode-vs-LaTeX formula outputs); since v1.6, **MGAM** adaptively adjusts segmentation granularity on the prediction side only;
  - special relations (`truncated` paragraph concatenation, caption↔figure parent/child) and `ignore` flags are honored from the GT JSON;
  - three parallel worker pools (page matching, CDM rendering, TEDS) with tuning knobs (`match_workers`, `cdm_workers`, `teds_workers`) ([README, Evaluation](https://github.com/opendatalab/OmniDocBench#evaluation)).
- **Deployment reality**: Python 3.10 + TeX Live 2025 + ImageMagick 7.1.1-47 + Ghostscript 9.55 for CDM; official Docker image `ghcr.io/zeng-weijun/omnidocbench-eval:repro-ubuntu2204`.

### lmms-eval ([arXiv:2407.12772](https://arxiv.org/abs/2407.12772), NAACL 2025 Findings; [repo](https://github.com/EvolvingLMMs-Lab/lmms-eval))

- 100+ multimodal tasks; a task = a YAML (dataset path, splits, prompt functions) + a `utils.py` with `process_results` + `metric_list` ([task guide](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/guides/task_guide.md)).
- OCR-relevant tasks (OCRBench, OCRBench v2, DocVQA, Fox-style QA) are evaluated as per-sample image+question → answer with custom scoring functions; for DocVQA-test it generates an RRC submission file instead of scoring locally (ANLS is applied server-side) ([docvqa utils.py](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/tasks/docvqa/utils.py)).
- Designed for **uniform model serving** (chat protocol, vLLM/SGLang/OpenAI-compatible backends), not for markdown↔GT alignment.

### olmOCR-Bench

- No GT text alignment at all: binary unit tests over the markdown output (anchor strings present, table shape constraints, order checks). Cheap to run, robust to formatting variance — the pattern was adopted by the French PDF→Markdown benchmark ([arXiv:2602.11960](https://arxiv.org/abs/2602.11960)).

### Reuse vs. reimplement

| Reuse (Apache-2.0/MIT, vendorable) | Reimplement/wire ourselves |
|---|---|
| OmniDocBench metric code: normalized edit distance, TEDS wrapper (PubTabNet-derived), COCODet wrapper, reading-order ED, matching algorithms (`quick_match`, hybrid matching, MGAM) | The **adapter**: PDF/page-image in → Markdown/JSON out, per our `benchmark/adapters/` protocol |
| `jiwer` for CER/WER; RapidFuzz for fast Levenshtein | Dataset loading into our JSONL/dataset adapter layer |
| PubTabNet TEDS; GriTS (table-transformer) if matrix-level table scoring is wanted | MLflow logging of per-category/per-attribute scores |
| CDM *as a container* (official Docker image or the v1.6 pure-Python CDM inside OmniDocBench) | Unit-test-style checks (olmOCR pattern) tailored to our own document types |
| lmms-eval as a **baseline runner** for VLM parsers (it already serves Qwen-VL-class models uniformly) | — |

---

## 5. Multimodal RAG connection: does parsing quality predict RAG quality?

**Framing is universal; quantified links are scarce.**

- The two central benchmarks explicitly justify themselves via RAG: OmniDocBench — document extraction "underpin[s] the data needs of LLMs and RAG systems" ([arXiv:2412.07626](https://arxiv.org/abs/2412.07626)); the survey *Document Parsing Unveiled* ([arXiv:2410.21169](https://arxiv.org/abs/2410.21169), v5 2026) lists RAG/knowledge-base construction as the downstream application and standardizes the metrics/benchmarks taxonomy.
- **Image-RAG vs parse-then-RAG comparisons** (the strongest quantified evidence, but about *avoiding* parsing, not predicting RAG from parsing metrics):
  - VisRAG: image-based RAG yields **25–40% end-to-end gains** over parse-then-text-RAG, attributing the gap to information loss during parsing ([arXiv:2410.10594](https://arxiv.org/abs/2410.10594)).
  - ColPali/ViDoRe: text-extraction pipelines are "lengthy and brittle"; embedding page images directly outperforms them on visually-rich retrieval ([arXiv:2407.01449](https://arxiv.org/abs/2407.01449), ICLR 2025).
  - M3DocRAG: OCR-based text RAG misses evidence in figures/charts; multimodal retrieval over page images addresses this ([arXiv:2411.04952](https://arxiv.org/abs/2411.04952)).
  - VDocRAG/OpenDocVQA: unified image-format RAG "to prevent missing information that occurs by parsing" ([arXiv:2504.09795](https://arxiv.org/abs/2504.09795), CVPR 2025).
- **Direct parsing-quality → downstream studies (2026, early):**
  - *Comparative Evaluation of Digitization Pipelines for Historiographical Sources* ([arXiv:2608.24976](https://arxiv.org/abs/2608.24976)): 13 PDF→text pipelines; OCR errors "propagate through RAG pipelines, compromising factual accuracy"; Marker best (98.70% CER / 97.71% WER accuracy); **LLM post-correction of OCR does not systematically help and often hurts**. Closest existing evidence, single domain, small scale.
  - French PDF→Markdown benchmark ([arXiv:2602.11960](https://arxiv.org/abs/2602.11960)): motivates its unit-test-style metrics by "transcription and layout errors propagat[ing] to downstream retrieval and grounding" — design-for-downstream, but does not measure RAG outcomes directly.
  - HiPerRAG ([arXiv:2505.04846](https://arxiv.org/abs/2505.04846)): parsing throughput/quality (Oreo model) as the gating stage of million-article RAG; no metric-correlation analysis.

> ⚠ **Fast-moving/unsettled.** As of this writing there is **no established, peer-reviewed quantitative result** that a parsing metric (CER/TEDS/CDM) predicts downstream RAG metrics (retrieval recall / answer quality). The image-RAG papers demonstrate that *bad parsing loses information*; the 2026 case studies demonstrate error propagation anecdotally in specific domains. A systematic "parsing metrics vs RAG outcomes" correlation study is an open gap — and a publishable opportunity for this framework, which already computes both sides (gold-doc retrieval metrics + RAGAS/answer metrics).

---

## 6. Recommendations for our framework

**Adapter-shaped integration (matches `benchmark/adapters/` conventions)**

1. Add a **document-parser adapter family** beside the RAG adapters, registered via a `register_parser_adapter`-style entry point: contract = `PDF/page images → ParseResult(page_markdown, blocks[{bbox, category, text, table_html?, formula_latex?}])`. HTTP adapter covers Mistral-OCR-style APIs and OpenAI-compatible olmOCR/vLLM servers (the emerging service standard); in-process Python plugin adapter covers Docling, MinerU, Marker locally.
2. Make the parser a **swappable ingestion stage** in existing experiments: corpus = concatenated per-page markdown, then the existing retrieval (`gold_doc_id`) and RAGAS/answer metrics run unchanged. Log parser identity + version in the run manifest (the resumable worker already keys results by config).
3. Run the **correlation experiment nobody has published**: same corpus, N parsers, report parsing metrics (CER/TEDS/CDM) vs retrieval recall and answer quality, per document category. This directly fills the §5 gap.

**Metric library choices**

4. Table stakes now: `jiwer` (CER/WER) + RapidFuzz-based normalized edit distance. Vendor OmniDocBench's `src` metric code (Apache-2.0) for TEDS, COCODet (mAP) and reading-order edit distance instead of reimplementing.
5. Treat **GT↔prediction matching** as first-class: adopt OmniDocBench's `quick_match` semantics; version-pin the matching algorithm in run metadata (v1.5 hybrid matching ≠ v1.6 MGAM — scores are not cross-version comparable).
6. **CDM is optional and containerized**: defer until formula-heavy corpora matter; when needed, run OmniDocBench's official Docker eval image rather than assembling TeX Live/ImageMagick/Ghostscript by hand. Skip charts entirely for v1 (no standard metric exists).
7. Add olmOCR-Bench-style **unit tests** (anchor presence, table constraints, header/footer removal) as cheap CI-grade regression checks for our internal parsing pipeline — no GT alignment required.

**Dataset shortlist**

8. Primary: **OmniDocBench v1.5/v1.6** (most complete annotation schema; EN+ZH). Cheap regression: **olmOCR-Bench** (Apache-2.0). Tables: **PubTabNet** (TEDS). Layout: **DocLayNet** (COCO mAP). Downstream proxy: **DocVQA** (ANLS). Dense-page stress: **Fox** (⚠ CC-BY-NC data). Lightweight commercial-safe parsing set: **DP-Bench** (MIT).
9. License hygiene: OmniDocBench data (research-only) and Fox data (NC) must not back production/vendor-facing claims; keep them in `experiments/` configs flagged accordingly.

---

## 7. Verification notes (primary-source status)

Verified against primary sources (paper page / official repo / HF dataset API): OmniDocBench (README + arXiv + version history + leaderboard), CDM (arXiv + UniMERNet path), PubTabNet (arXiv + repo), GriTS (arXiv), DocLayNet (arXiv + HF), olmOCR/olmOCR-Bench (repo README + both arXiv papers), OCRBench v1/v2 (arXiv abstracts + HF API), CC-OCR (arXiv + HF), Fox (repo README + HF card), DocVQA (arXiv + lmms-eval utils), DP-Bench (HF card), Docling (arXiv), GOT-OCR 2.0 (arXiv), ColPali (arXiv), VisRAG (arXiv), M3DocRAG (arXiv), VDocRAG (arXiv), lmms-eval (arXiv + README + task utils), jiwer (README), ChartMimic (arXiv), Document Parsing Unveiled (arXiv), digitization-pipelines study and French PDF→MD benchmark (arXiv abstracts).

Not fully verified from a primary source (flagged above where used):
- Fox page counts (112 EN / 100 ZH) — only seen in a secondary wiki; omitted from §2.
- OCRBench v1 total question count — abstract states 29 datasets; no size stated here.
- CC-OCR V2 details beyond its HF card (paper arXiv:2605.03903 not read).
- DocLayNet's exact license string (HF card says `license: other`; commonly documented as CDLA-Permissive).
- DP-Bench's "NID" metric definition — taken from a third-party paper's abstract (NovaLAD), not from Upstage's own documentation.
- MinerU2.5 / PaddleOCR-VL model details — cited only via the OmniDocBench leaderboard, not their own papers.
