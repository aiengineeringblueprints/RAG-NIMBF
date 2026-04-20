# Vector Distance Evaluation — Design Spec

## Context

Currently the benchmarking framework evaluates LLM answers using RAGAS metrics and custom NLG/IR metrics (ROUGE-L, BLEU, METEOR, BERTScore, etc.). These metrics measure textual similarity but not semantic distance in embedding space. Adding cosine-distance comparisons between question↔ground_truth and question↔answer provides a geometric interpretation of answer quality — how close is the LLM answer to the ground truth relative to the original question in vector space.

## Scope

- Compute `vec_dist_q_gt` and `vec_dist_q_answer` per sample in existing `compute_custom_metrics()`
- Generate a scatter plot comparing these two distances across all samples
- Integrate into the existing reporting pipeline (automatic after each benchmark run)

Out of scope: new CLI commands, new embedding models, changes to config or data models.

## Metrics

Two new per-sample metrics added to `compute_custom_metrics()`:

| Metric | Formula | Meaning |
|--------|---------|---------|
| `vec_dist_q_gt` | `1 - cosine_similarity(embed(question), embed(ground_truth))` | Semantic distance between question and ground truth |
| `vec_dist_q_answer` | `1 - cosine_similarity(embed(question), embed(answer))` | Semantic distance between question and LLM answer |

Range: [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite.

Both metrics require `embed_fn` (already passed to `compute_custom_metrics()`). When `embed_fn` is None, both metrics are set to None.

## Visualization

New scatter plot in `visualization.py`:

- **X-axis**: `vec_dist_q_gt` — how semantically far the ground truth is from the question
- **Y-axis**: `vec_dist_q_answer` — how semantically far the LLM answer is from the question
- **Diagonal reference line** (dashed): points below = answer is closer to question than GT
- **Multiple configurations**: different colors per config, using the existing `_PALETTE`
- **Saved as**: `results_plots/vector_distance_scatter.png`

### Interpretation

- Points near origin (0,0): question and GT/answer are semantically very similar (easy questions)
- Points below diagonal: answer is closer to the question than GT (possible oversimplification)
- Points above diagonal: answer is farther from question than GT (potential hallucination or drift)
- Cluster positions reveal dataset difficulty distribution

## Files to Modify

### 1. `benchmark/custom_metrics.py`

In `compute_custom_metrics()`, after the existing IR metrics block, add:

```python
# Vector distance metrics
if embed_fn is not None:
    q_emb = embed_fn(q)
    gt_emb = embed_fn(gt)
    ans_emb = embed_fn(ans)
    vec_dist_q_gt = 1.0 - _cosine_sim(q_emb, gt_emb)
    vec_dist_q_answer = 1.0 - _cosine_sim(q_emb, ans_emb)
    scores["vec_dist_q_gt"] = vec_dist_q_gt
    scores["vec_dist_q_answer"] = vec_dist_q_answer
    accum.setdefault("vec_dist_q_gt", []).append(vec_dist_q_gt)
    accum.setdefault("vec_dist_q_answer", []).append(vec_dist_q_answer)
else:
    scores["vec_dist_q_gt"] = None
    scores["vec_dist_q_answer"] = None
```

Embedding the question is already done once per sample for `context_relevance`. To avoid redundant embedding, reuse the question embedding from that computation if it precedes this block. The `_cosine_sim()` function already exists in this file.

### 2. `benchmark/reporting/visualization.py`

Add a new function `_plot_vector_distance_scatter()`:

- Extract `vec_dist_q_gt` and `vec_dist_q_answer` from each result's `per_sample.custom_scores`
- For single config: single scatter with one color
- For multiple configs: overlay with different colors and legend
- Plot diagonal line `y = x` as dashed reference
- Labels, title, grid following existing plot style

Integrate into `generate_plots()` by appending the call alongside existing plot functions.

### No changes needed in:
- `main.py` — already passes `embed_fn` to `compute_custom_metrics()`
- `benchmark/reporting/models.py` — `PerSampleResult.custom_scores` dict already holds arbitrary metrics
- `benchmark/reporting/__init__.py` — `generate_plots()` is already called
- Configuration — no new config parameters needed

## Verification

1. Run a benchmark with at least 3-5 samples
2. Check that `vec_dist_q_gt` and `vec_dist_q_answer` appear in per-sample CSV export
3. Verify `vector_distance_scatter.png` is generated in `results_plots/`
4. Confirm scatter plot shows expected pattern: answers generally cluster near GT distances
5. Run with `embed_fn=None` and confirm both metrics are None (graceful degradation)
