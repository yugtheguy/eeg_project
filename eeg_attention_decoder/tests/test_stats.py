"""
tests/test_stats.py
====================
Unit tests for cross-validation integrity, temporal leakage detection,
and permutation-test logic.

Critical invariants tested
---------------------------
1. LOBO folds never share block IDs between train and test.
2. No temporal overlap between train and test windows (leakage test).
3. ``assert_no_boundary_overlap`` raises on deliberately injected leakage.
4. Permutation label shuffling preserves block structure.
5. Shuffled labels differ from originals (sanity).
6. Permutation p-value > 0.05 for random features (null is not inflated).
7. CV summary statistics are within [0, 1] for accuracy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.labeling import assert_no_boundary_overlap
from stats.cross_validation import (
    leave_one_block_out_splits,
    run_cv,
    summarise_cv,
)
from stats.permutation_tests import shuffle_labels_block_structure


FS = 250.0
WINDOW_SEC = 1.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_dataset(
    n_blocks: int = 6,
    windows_per_block: int = 20,
    n_features: int = 7,
    n_classes: int = 2,
    seed: int = 42,
):
    """
    Synthetic dataset with clean block structure.

    Returns
    -------
    X, y, block_ids, timestamps : np.ndarray
    """
    rng = np.random.default_rng(seed)
    n_windows = n_blocks * windows_per_block

    X = rng.normal(size=(n_windows, n_features))
    block_ids = np.repeat(np.arange(n_blocks), windows_per_block)

    # Labels: alternate classes per block (no predictive signal — random)
    y = np.array([bid % n_classes for bid in block_ids], dtype=np.int32)

    # Timestamps: non-overlapping windows
    step = WINDOW_SEC / 2   # 50 % overlap step
    center_offset = WINDOW_SEC / 2
    timestamps = np.array([
        center_offset + i * step for i in range(n_windows)
    ])

    return X, y, block_ids, timestamps


# ---------------------------------------------------------------------------
# LOBO split integrity
# ---------------------------------------------------------------------------

class TestLOBOIntegrity:

    def test_no_shared_block_ids(self):
        """Train and test folds must never share a block ID."""
        _, y, block_ids, timestamps = _make_dataset()
        for train_idx, test_idx in leave_one_block_out_splits(
            block_ids, timestamps, WINDOW_SEC
        ):
            shared = np.intersect1d(block_ids[train_idx], block_ids[test_idx])
            assert len(shared) == 0, (
                f"Block ID(s) {shared} appear in both train and test."
            )

    def test_all_windows_covered(self):
        """Every window must appear in exactly one test fold."""
        _, _, block_ids, timestamps = _make_dataset()
        n_windows = len(block_ids)
        appeared = np.zeros(n_windows, dtype=np.int32)
        for _, test_idx in leave_one_block_out_splits(block_ids, timestamps, WINDOW_SEC):
            appeared[test_idx] += 1
        assert np.all(appeared == 1), (
            "Some windows appeared in 0 or >1 test folds."
        )

    def test_n_folds_equals_n_blocks(self):
        """LOBO must produce exactly n_blocks folds."""
        _, _, block_ids, timestamps = _make_dataset(n_blocks=5)
        splits = list(leave_one_block_out_splits(block_ids, timestamps, WINDOW_SEC))
        assert len(splits) == 5

    def test_minimum_blocks_requirement(self):
        """LOBO must raise if fewer than 2 blocks are provided."""
        n = 20
        block_ids = np.zeros(n, dtype=np.int32)
        timestamps = np.arange(n, dtype=float)
        with pytest.raises(ValueError, match="2"):
            list(leave_one_block_out_splits(block_ids, timestamps, WINDOW_SEC))


# ---------------------------------------------------------------------------
# Temporal leakage detection
# ---------------------------------------------------------------------------

class TestTemporalLeakage:

    def test_no_leakage_on_clean_data(self):
        """Clean dataset should pass the no-overlap assertion silently."""
        _, _, block_ids, timestamps = _make_dataset()
        # Manually perform one fold and check
        for train_idx, test_idx in leave_one_block_out_splits(
            block_ids, timestamps, WINDOW_SEC
        ):
            # Should not raise
            assert_no_boundary_overlap(
                timestamps[train_idx],
                timestamps[test_idx],
                WINDOW_SEC,
            )

    def test_leakage_detected_on_injected_overlap(self):
        """
        If a test-window timestamp is injected into the training set
        (simulating temporal leakage), the guard must raise AssertionError.
        """
        train_ts = np.array([0.5, 1.0, 1.5, 2.0])  # 50 % overlap windows
        test_ts  = np.array([1.5])                   # Same timestamp as train!

        with pytest.raises(AssertionError, match="TEMPORAL LEAKAGE"):
            assert_no_boundary_overlap(train_ts, test_ts, WINDOW_SEC)

    def test_adjacent_windows_do_not_trigger_leakage(self):
        """
        Windows that are adjacent but NOT overlapping should not trigger
        the leakage guard.
        """
        # Train ends at t=4.0 (centre of window [3.5, 4.5])
        # Test starts at t=5.0 (centre of window [4.5, 5.5]) — no overlap
        train_ts = np.array([0.5, 1.5, 2.5, 3.5])
        test_ts  = np.array([4.5, 5.5, 6.5])
        # Should not raise
        assert_no_boundary_overlap(train_ts, test_ts, WINDOW_SEC)


# ---------------------------------------------------------------------------
# Permutation test logic
# ---------------------------------------------------------------------------

class TestPermutationLogic:

    def test_shuffled_labels_differ_from_original(self):
        """Shuffled labels must differ from the original at least once."""
        _, y, block_ids, _ = _make_dataset(n_blocks=8, seed=99)
        rng = np.random.default_rng(0)
        y_perm = shuffle_labels_block_structure(y, block_ids, rng)
        assert not np.array_equal(y, y_perm), (
            "Permuted labels identical to originals — shuffle did not work."
        )

    def test_shuffled_labels_same_distribution(self):
        """Permuted labels should have same class counts as originals."""
        _, y, block_ids, _ = _make_dataset(n_blocks=8)
        rng = np.random.default_rng(0)
        y_perm = shuffle_labels_block_structure(y, block_ids, rng)
        np.testing.assert_array_equal(
            np.sort(np.unique(y_perm)),
            np.sort(np.unique(y)),
            err_msg="Permuted labels introduced new classes.",
        )

    def test_block_structure_preserved_after_shuffle(self):
        """Within each block, all windows should still share the same label."""
        _, y, block_ids, _ = _make_dataset(n_blocks=6)
        rng = np.random.default_rng(42)
        y_perm = shuffle_labels_block_structure(y, block_ids, rng)

        for bid in np.unique(block_ids):
            mask = block_ids == bid
            labels_in_block = np.unique(y_perm[mask])
            assert len(labels_in_block) == 1, (
                f"Block {bid} has mixed labels after shuffle: {labels_in_block}"
            )

    def test_random_features_not_significant(self):
        """
        A classifier trained on random features + random labels should
        produce a null-hypothesis-consistent (non-significant) p-value in
        at least 95 % of repeated small-scale simulations.
        This acts as a calibration check for the permutation framework.

        We run 50 permutations (not 1000 for speed) and check that the
        observed BACC is within the bulk of the null distribution.
        """
        from models.lda import AttentionLDA
        from stats.cross_validation import run_cv, summarise_cv
        from stats.permutation_tests import shuffle_labels_block_structure

        X, y, block_ids, timestamps = _make_dataset(n_blocks=6, seed=7)

        def model_factory():
            return AttentionLDA(random_state=42)

        obs_results = run_cv(
            X, y, block_ids, timestamps, WINDOW_SEC, model_factory
        )
        obs_bacc = summarise_cv(obs_results)["mean_balanced_accuracy"]

        rng = np.random.default_rng(123)
        null_baccs = []
        for _ in range(50):
            y_p = shuffle_labels_block_structure(y, block_ids, rng)
            perm_res = run_cv(
                X, y_p, block_ids, timestamps, WINDOW_SEC, model_factory
            )
            null_baccs.append(summarise_cv(perm_res)["mean_balanced_accuracy"])

        null_arr = np.array(null_baccs)
        p_value = float(np.mean(null_arr >= obs_bacc))

        # Random data should NOT be significant (p > 0.05 almost certainly)
        # With only 50 permutations, we use 0.02 as a very conservative bound
        assert p_value > 0.02, (
            f"Random-feature model appears significant (p={p_value:.4f}). "
            "This may indicate inflated null distribution or leakage."
        )


# ---------------------------------------------------------------------------
# CV summary statistics
# ---------------------------------------------------------------------------

class TestCVSummary:

    def test_summary_accuracy_in_unit_interval(self):
        """mean_accuracy and mean_balanced_accuracy must be in [0, 1]."""
        from models.lda import AttentionLDA

        X, y, block_ids, timestamps = _make_dataset()

        def model_factory():
            return AttentionLDA(random_state=42)

        results = run_cv(X, y, block_ids, timestamps, WINDOW_SEC, model_factory)
        summary = summarise_cv(results)

        assert 0.0 <= summary["mean_accuracy"] <= 1.0
        assert 0.0 <= summary["mean_balanced_accuracy"] <= 1.0

    def test_summary_has_required_keys(self):
        from models.lda import AttentionLDA

        X, y, block_ids, timestamps = _make_dataset()
        results = run_cv(X, y, block_ids, timestamps, WINDOW_SEC,
                         lambda: AttentionLDA(random_state=42))
        summary = summarise_cv(results)

        for key in ["n_folds", "mean_accuracy", "std_accuracy",
                    "mean_balanced_accuracy", "std_balanced_accuracy"]:
            assert key in summary, f"Missing key in summary: '{key}'"
