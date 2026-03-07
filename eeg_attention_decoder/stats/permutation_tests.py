"""
stats/permutation_tests.py
===========================
Block-structure-preserving permutation test for EEG attention decoding.

Method
------
1. Shuffle labels at the *block* level (not the window level) to preserve
   temporal autocorrelation within blocks.
2. Re-run the full LOBO or LOSO cross-validation on each shuffle.
3. Build the null distribution of balanced accuracy.
4. Compute empirical p-value: fraction of null samples >= observed.
5. Reject the null at ``alpha_level`` (default 0.05).

This approach satisfies the minimum requirement for publication-level
statistical inference on a single-subject dataset.
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt

from stats.cross_validation import run_cv, summarise_cv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Block-level label shuffler
# ---------------------------------------------------------------------------

def shuffle_labels_block_structure(
    y: np.ndarray,
    block_ids: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permute labels by shuffling entire blocks, preserving within-block
    temporal structure.

    Parameters
    ----------
    y : np.ndarray, shape (n_windows,)
        Class labels.
    block_ids : np.ndarray, shape (n_windows,)
        Block ID for each window.
    rng : np.random.Generator

    Returns
    -------
    y_shuffled : np.ndarray, shape (n_windows,)
        Labels with block-level permutation applied; an entire block's
        windows receive another block's majority label.
    """
    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    # Determine the majority label per block
    block_labels: List[int] = []
    for bid in unique_blocks:
        mask = block_ids == bid
        vals, counts = np.unique(y[mask], return_counts=True)
        block_labels.append(int(vals[np.argmax(counts)]))

    # Permute the block-label mapping
    perm = rng.permutation(n_blocks)
    shuffled_block_labels = [block_labels[p] for p in perm]

    y_shuffled = y.copy()
    for orig_bid, new_label in zip(unique_blocks, shuffled_block_labels):
        y_shuffled[block_ids == orig_bid] = new_label

    return y_shuffled


# ---------------------------------------------------------------------------
# Main permutation test
# ---------------------------------------------------------------------------

def run_permutation_test(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    window_timestamps: np.ndarray,
    window_size_sec: float,
    model_factory: Callable,
    n_permutations: int = 1000,
    alpha_level: float = 0.05,
    cv_strategy: str = "leave_one_block_out",
    session_ids: Optional[np.ndarray] = None,
    random_seed: int = 42,
    plot: bool = True,
    plot_path: Optional[str] = None,
) -> Dict:
    """
    Run block-preserving permutation test.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_features)
    y : np.ndarray, shape (n_windows,)
    block_ids : np.ndarray, shape (n_windows,)
    window_timestamps : np.ndarray, shape (n_windows,)
    window_size_sec : float
    model_factory : callable
        Returns a fresh unfitted model; same as used in ``run_cv``.
    n_permutations : int
    alpha_level : float
    cv_strategy : str
    session_ids : np.ndarray, optional
    random_seed : int
    plot : bool
        If True, plot the null distribution.
    plot_path : str, optional
        If provided, save the plot to this path instead of displaying.

    Returns
    -------
    dict with keys:
        observed_bacc, null_distribution, p_value, reject_null,
        alpha_level, n_permutations
    """
    # ── Observed performance ───────────────────────────────────────────────
    logger.info("Computing observed CV performance …")
    observed_results = run_cv(
        X, y, block_ids, window_timestamps, window_size_sec,
        model_factory, strategy=cv_strategy, session_ids=session_ids,
    )
    observed_summary = summarise_cv(observed_results)
    observed_bacc = observed_summary["mean_balanced_accuracy"]
    logger.info("Observed balanced accuracy: %.4f", observed_bacc)

    # ── Null distribution ──────────────────────────────────────────────────
    rng = np.random.default_rng(random_seed)
    null_baccs: List[float] = []

    t0 = time.monotonic()
    for perm_i in range(n_permutations):
        y_perm = shuffle_labels_block_structure(y, block_ids, rng)
        perm_results = run_cv(
            X, y_perm, block_ids, window_timestamps, window_size_sec,
            model_factory, strategy=cv_strategy, session_ids=session_ids,
        )
        perm_summary = summarise_cv(perm_results)
        null_baccs.append(perm_summary["mean_balanced_accuracy"])

        if (perm_i + 1) % 100 == 0:
            elapsed = time.monotonic() - t0
            eta = elapsed / (perm_i + 1) * (n_permutations - perm_i - 1)
            logger.info(
                "Permutation %d/%d | ETA %.0f s", perm_i + 1, n_permutations, eta
            )

    null_array = np.array(null_baccs)
    p_value = float(np.mean(null_array >= observed_bacc))
    reject_null = p_value < alpha_level

    result = {
        "observed_bacc":    round(observed_bacc, 6),
        "null_mean":        round(float(null_array.mean()), 6),
        "null_std":         round(float(null_array.std()), 6),
        "null_p95":         round(float(np.percentile(null_array, 95)), 6),
        "p_value":          round(p_value, 6),
        "reject_null":      reject_null,
        "alpha_level":      alpha_level,
        "n_permutations":   n_permutations,
        "cv_strategy":      cv_strategy,
        "random_seed":      random_seed,
        "null_distribution": null_array.tolist(),
    }

    status = "REJECTED (significant)" if reject_null else "RETAINED (not significant)"
    logger.info(
        "Permutation test: p=%.4f | H₀ %s | observed_bacc=%.4f",
        p_value, status, observed_bacc,
    )

    if plot:
        _plot_null_distribution(
            null_array, observed_bacc, p_value, alpha_level, plot_path
        )

    return result


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def _plot_null_distribution(
    null_dist: np.ndarray,
    observed: float,
    p_value: float,
    alpha_level: float,
    save_path: Optional[str] = None,
) -> None:
    """Plot the permutation null distribution with observed score marked."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(null_dist, bins=40, density=True, color="#7ab3d4", edgecolor="white",
            label="Null distribution")
    ax.axvline(observed, color="#d62728", linewidth=2.0,
               label=f"Observed = {observed:.3f}")

    # 95th percentile of null
    p95 = float(np.percentile(null_dist, 100 * (1 - alpha_level)))
    ax.axvline(p95, color="#ff7f0e", linewidth=1.5, linestyle="--",
               label=f"α={alpha_level} threshold = {p95:.3f}")

    verdict = "Significant" if p_value < alpha_level else "Not significant"
    ax.set_title(
        f"Permutation Test — {verdict}\n"
        f"p = {p_value:.4f} | observed BACC = {observed:.3f}",
        fontsize=12,
    )
    ax.set_xlabel("Balanced Accuracy (null distribution)")
    ax.set_ylabel("Density")
    ax.legend(frameon=True)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Null distribution plot saved: %s", save_path)
    else:
        plt.show()
    plt.close(fig)
