"""Comparison scoring: AUROC, Pearson/Spearman correlation, RMSE, Mann-Whitney."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ComparisonMetrics:
    """Metrics comparing one detector against the RAGAS faithfulness baseline."""

    detector_name: str
    dataset_name: str
    n_samples: int
    mean_score: float
    std_score: float
    auroc: float | None
    auroc_threshold: float
    pearson_r: float | None
    spearman_r: float | None
    rmse_vs_ragas: float | None
    hallucinated_pct: float
    mann_whitney_p: float | None


def compute_comparison(
    detector_scores: list[float],
    ragas_scores: list[float],
    detector_name: str = "",
    dataset_name: str = "",
    threshold: float = 0.5,
) -> ComparisonMetrics:
    """Compare a detector's scores against RAGAS faithfulness."""
    pairs = [
        (d, r)
        for d, r in zip(detector_scores, ragas_scores)
        if not (math.isnan(d) or math.isnan(r))
    ]
    if len(pairs) < 5:
        return ComparisonMetrics(
            detector_name=detector_name,
            dataset_name=dataset_name,
            n_samples=0,
            mean_score=0.0,
            std_score=0.0,
            auroc=None,
            auroc_threshold=threshold,
            pearson_r=None,
            spearman_r=None,
            rmse_vs_ragas=None,
            hallucinated_pct=0.0,
            mann_whitney_p=None,
        )

    d_arr = np.array([p[0] for p in pairs])
    r_arr = np.array([p[1] for p in pairs])
    binary = (r_arr < threshold).astype(float)

    auroc = None
    if binary.sum() >= 2 and (1 - binary).sum() >= 2:
        try:
            from sklearn.metrics import roc_auc_score
            auroc = roc_auc_score(binary, 1.0 - d_arr)
        except Exception:
            pass

    pearson_r = None
    spearman_r = None
    try:
        from scipy.stats import pearsonr, spearmanr
        pearson_r, _ = pearsonr(d_arr, r_arr)
        spearman_r, _ = spearmanr(d_arr, r_arr)
    except Exception:
        pass

    rmse = float(np.sqrt(np.mean((r_arr - d_arr) ** 2)))

    mw_p = None
    if binary.sum() >= 2 and (1 - binary).sum() >= 2:
        try:
            from scipy.stats import mannwhitneyu
            _, mw_p = mannwhitneyu(
                d_arr[binary == 1], d_arr[binary == 0], alternative="two-sided"
            )
        except Exception:
            pass

    return ComparisonMetrics(
        detector_name=detector_name,
        dataset_name=dataset_name,
        n_samples=len(pairs),
        mean_score=float(d_arr.mean()),
        std_score=float(d_arr.std()),
        auroc=auroc,
        auroc_threshold=threshold,
        pearson_r=pearson_r,
        spearman_r=spearman_r,
        rmse_vs_ragas=rmse,
        hallucinated_pct=float(binary.mean() * 100),
        mann_whitney_p=mw_p,
    )


def compute_threshold_sensitivity(
    detector_scores: list[float],
    ragas_scores: list[float],
    thresholds: list[float] | None = None,
) -> list[tuple[float, float | None]]:
    """Compute AUROC at multiple RAGAS faithfulness thresholds."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []
    for t in thresholds:
        pairs = [
            (d, r)
            for d, r in zip(detector_scores, ragas_scores)
            if not (math.isnan(d) or math.isnan(r))
        ]
        if len(pairs) < 10:
            results.append((t, None))
            continue

        d_arr = np.array([p[0] for p in pairs])
        r_arr = np.array([p[1] for p in pairs])
        binary = (r_arr < t).astype(float)

        auroc = None
        if binary.sum() >= 2 and (1 - binary).sum() >= 2:
            try:
                from sklearn.metrics import roc_auc_score
                auroc = roc_auc_score(binary, 1.0 - d_arr)
            except Exception:
                pass
        results.append((t, auroc))

    return results
