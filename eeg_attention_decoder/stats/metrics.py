"""
stats/metrics.py
================
Publication-grade performance metrics for EEG spatial attention decoding.

Metrics computed
----------------
* Accuracy (window-level)
* Balanced accuracy (macro-averaged recall; robust to class imbalance)
* ROC-AUC (binary or multi-class OvR)
* Block-level majority-vote accuracy
* 95 % bootstrap confidence intervals (BCa method via resample)

IMPORTANT: Window-level accuracy must NEVER be reported alone in a
publication.  Always accompany it with block-level aggregation and
bootstrap CI.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from stats.cross_validation import CVFoldResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-level majority vote
# ---------------------------------------------------------------------------

def block_majority_vote(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_ids: np.ndarray,
) -> Dict:
    """
    Aggregate window predictions to block-level via majority vote.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_windows,)
    y_pred : np.ndarray, shape (n_windows,)
    block_ids : np.ndarray, shape (n_windows,)

    Returns
    -------
    dict with keys:
        block_accuracy, n_blocks,
        block_true_labels, block_predicted_labels
    """
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)
    block_true: List[int] = []
    block_pred: List[int] = []

    for bid in unique_blocks:
        mask = block_ids == bid
        # True label: majority vote of window labels in this block
        vals, counts = np.unique(y_true[mask], return_counts=True)
        true_lbl = int(vals[np.argmax(counts)])
        # Predicted label: majority vote of window predictions
        pvals, pcounts = np.unique(y_pred[mask], return_counts=True)
        pred_lbl = int(pvals[np.argmax(pcounts)])
        block_true.append(true_lbl)
        block_pred.append(pred_lbl)

    block_true_arr = np.array(block_true)
    block_pred_arr = np.array(block_pred)
    block_acc = float(accuracy_score(block_true_arr, block_pred_arr))

    return {
        "block_accuracy": round(block_acc, 4),
        "n_blocks": n_blocks,
        "block_true_labels": block_true_arr.tolist(),
        "block_predicted_labels": block_pred_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
    **metric_kwargs,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap percentile confidence interval for a metric.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    metric_fn : callable (y_true, y_pred, **kwargs) -> float
    n_bootstrap : int
    ci_level : float  (e.g. 0.95 for 95 % CI)
    random_seed : int
    **metric_kwargs : passed to ``metric_fn``.

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(random_seed)
    n = len(y_true)
    boot_scores: List[float] = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            score = float(metric_fn(y_true[idx], y_pred[idx], **metric_kwargs))
        except Exception:
            continue
        boot_scores.append(score)

    boot_arr = np.array(boot_scores)
    alpha = 1.0 - ci_level
    lo = float(np.percentile(boot_arr, 100 * alpha / 2))
    hi = float(np.percentile(boot_arr, 100 * (1 - alpha / 2)))
    point = float(metric_fn(y_true, y_pred, **metric_kwargs))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Full metric suite
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    block_ids: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
    multiclass: bool = False,
) -> Dict:
    """
    Compute the complete publication-grade metric suite.

    Parameters
    ----------
    y_true : np.ndarray, shape (n_windows,)
    y_pred : np.ndarray, shape (n_windows,)
    y_proba : np.ndarray, shape (n_windows, n_classes)
    block_ids : np.ndarray, shape (n_windows,)
    n_bootstrap : int
    ci_level : float
    random_seed : int
    multiclass : bool
        If True, ROC-AUC uses OvR averaging.

    Returns
    -------
    dict with all metrics and CIs.
    """
    # ── Window-level ────────────────────────────────────────────────────────
    acc, acc_lo,  acc_hi  = bootstrap_ci(
        y_true, y_pred, accuracy_score,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_seed=random_seed,
    )
    bacc, bacc_lo, bacc_hi = bootstrap_ci(
        y_true, y_pred, balanced_accuracy_score,
        n_bootstrap=n_bootstrap, ci_level=ci_level, random_seed=random_seed,
    )

    # ── ROC-AUC ─────────────────────────────────────────────────────────────
    n_classes = y_proba.shape[1] if y_proba.ndim == 2 else 2
    try:
        if n_classes == 2:
            auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        else:
            auc = float(roc_auc_score(
                y_true, y_proba, multi_class="ovr", average="macro"
            ))
    except Exception as exc:
        logger.warning("ROC-AUC computation failed: %s", exc)
        auc = float("nan")

    # ── Block-level ─────────────────────────────────────────────────────────
    block_metrics = block_majority_vote(y_true, y_pred, block_ids)

    metrics = {
        "window_level": {
            "accuracy":             round(acc, 4),
            "accuracy_ci_lo":       round(acc_lo, 4),
            "accuracy_ci_hi":       round(acc_hi, 4),
            "balanced_accuracy":    round(bacc, 4),
            "bacc_ci_lo":           round(bacc_lo, 4),
            "bacc_ci_hi":           round(bacc_hi, 4),
            "roc_auc":              round(auc, 4),
            "n_windows":            int(len(y_true)),
        },
        "block_level": block_metrics,
        "bootstrap": {
            "n_bootstrap": n_bootstrap,
            "ci_level":    ci_level,
        },
    }

    logger.info(
        "Metrics: acc=%.3f [%.3f–%.3f] | bacc=%.3f [%.3f–%.3f] | "
        "AUC=%.3f | block_acc=%.3f",
        acc, acc_lo, acc_hi,
        bacc, bacc_lo, bacc_hi,
        auc,
        block_metrics["block_accuracy"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Aggregate across CV folds
# ---------------------------------------------------------------------------

def aggregate_cv_metrics(
    cv_results: List[CVFoldResult],
    block_ids_full: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    random_seed: int = 42,
) -> Dict:
    """
    Pool all CV predictions and compute metrics on the full held-out set.

    Parameters
    ----------
    cv_results : list of CVFoldResult
    block_ids_full : np.ndarray, shape (n_total_windows,)
        Block IDs for ALL windows (before CV split); indexed by the
        test_idx stored in each fold result.
    n_bootstrap, ci_level, random_seed : as in ``compute_metrics``.

    Returns
    -------
    dict (same structure as ``compute_metrics``).
    """
    # Sort folds by fold_idx so concatenation is deterministic
    cv_results = sorted(cv_results, key=lambda r: r.fold_idx)

    all_y_true  = np.concatenate([r.y_test  for r in cv_results])
    all_y_pred  = np.concatenate([r.y_pred  for r in cv_results])
    all_y_proba = np.concatenate([r.y_proba for r in cv_results], axis=0)
    all_test_idx = np.concatenate([r.test_idx for r in cv_results])
    all_block_ids = block_ids_full[all_test_idx]

    multiclass = len(np.unique(all_y_true)) > 2

    return compute_metrics(
        all_y_true, all_y_pred, all_y_proba, all_block_ids,
        n_bootstrap=n_bootstrap, ci_level=ci_level,
        random_seed=random_seed, multiclass=multiclass,
    )
