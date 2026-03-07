"""
stats/cross_validation.py
==========================
Block-aware cross-validation strategies for EEG attention decoding.

Design principles (anti-leakage)
----------------------------------
1. Trial blocks are NEVER split across train/test folds.
2. No random shuffling of windows across block boundaries.
3. Every fold is validated for temporal separation before fitting.
4. Leave-One-Block-Out (LOBO) and Leave-One-Session-Out (LOSO) are
   both available; strategy is set in ``configs/default.yaml``.

Multi-subject support
----------------------
``leave_one_session_out`` groups data by (subject_id, session_id) tuples
so that the held-out fold is always a complete session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Generator, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from experiments.labeling import assert_no_boundary_overlap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class CVFoldResult:
    """Stores train/test indices and fitted model scores for one fold."""
    fold_idx: int
    train_idx: np.ndarray          # indices into the full X/y arrays
    test_idx: np.ndarray
    held_out_block_ids: np.ndarray  # block IDs in the test fold
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray            # shape (n_test, n_classes)
    accuracy: float
    balanced_accuracy: float


# ---------------------------------------------------------------------------
# Leave-One-Block-Out
# ---------------------------------------------------------------------------

def leave_one_block_out_splits(
    block_ids: np.ndarray,
    window_timestamps: np.ndarray,
    window_size_sec: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, test_idx) pairs for Leave-One-Block-Out CV.

    Parameters
    ----------
    block_ids : np.ndarray, shape (n_windows,)
        Integer block ID for each window (no -1 values; call
        ``labeling.filter_labeled`` first).
    window_timestamps : np.ndarray, shape (n_windows,)
        Centre timestamps; used to assert no temporal overlap.
    window_size_sec : float

    Yields
    ------
    (train_idx, test_idx) : both np.ndarray of integer indices
    """
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 2:
        raise ValueError("LOBO requires at least 2 distinct blocks.")

    for held_out in unique_blocks:
        test_mask  = block_ids == held_out
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        # ── Anti-leakage assertion ─────────────────────────────────────
        assert_no_boundary_overlap(
            window_timestamps[train_idx],
            window_timestamps[test_idx],
            window_size_sec=window_size_sec,
        )

        # ── Block integrity check ──────────────────────────────────────
        shared = np.intersect1d(block_ids[train_idx], block_ids[test_idx])
        assert len(shared) == 0, (
            f"LOBO integrity violation: block(s) {shared} appear in both "
            "train and test folds."
        )

        yield train_idx, test_idx


def leave_one_session_out_splits(
    session_ids: np.ndarray,
    block_ids: np.ndarray,
    window_timestamps: np.ndarray,
    window_size_sec: float,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate (train_idx, test_idx) pairs for Leave-One-Session-Out CV.

    Parameters
    ----------
    session_ids : np.ndarray, shape (n_windows,)
        String or integer session/subject identifier for each window.
        For multi-subject data pass ``"<sub_id>_<ses_id>"`` strings.
    block_ids : np.ndarray, shape (n_windows,)
    window_timestamps : np.ndarray, shape (n_windows,)
    window_size_sec : float

    Yields
    ------
    (train_idx, test_idx)
    """
    unique_sessions = np.unique(session_ids)
    if len(unique_sessions) < 2:
        raise ValueError("LOSO requires at least 2 sessions.")

    for held_out in unique_sessions:
        test_mask  = session_ids == held_out
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        # ── Anti-leakage assertion ─────────────────────────────────────
        assert_no_boundary_overlap(
            window_timestamps[train_idx],
            window_timestamps[test_idx],
            window_size_sec=window_size_sec,
        )

        # ── No shared block IDs across session boundary ────────────────
        shared_blocks = np.intersect1d(block_ids[train_idx], block_ids[test_idx])
        assert len(shared_blocks) == 0, (
            f"LOSO integrity violation: block IDs {shared_blocks} appear "
            "in both train and test folds."
        )

        yield train_idx, test_idx


# ---------------------------------------------------------------------------
# CV runner
# ---------------------------------------------------------------------------

def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    block_ids: np.ndarray,
    window_timestamps: np.ndarray,
    window_size_sec: float,
    model_factory,
    strategy: str = "leave_one_block_out",
    session_ids: Optional[np.ndarray] = None,
) -> List[CVFoldResult]:
    """
    Execute cross-validation and return per-fold results.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_features)
    y : np.ndarray, shape (n_windows,)
    block_ids : np.ndarray, shape (n_windows,)
    window_timestamps : np.ndarray, shape (n_windows,)
    window_size_sec : float
    model_factory : callable () -> fitted-model-with-predict_proba
        Called each fold to create a fresh model instance.
        E.g. ``lambda: AttentionLDA(shrinkage="auto")``.
    strategy : {"leave_one_block_out", "leave_one_session_out"}
    session_ids : np.ndarray, optional
        Required if ``strategy == "leave_one_session_out"``.

    Returns
    -------
    List[CVFoldResult]
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    if strategy == "leave_one_block_out":
        splits = leave_one_block_out_splits(block_ids, window_timestamps, window_size_sec)
    elif strategy == "leave_one_session_out":
        if session_ids is None:
            raise ValueError("session_ids must be provided for LOSO.")
        splits = leave_one_session_out_splits(
            session_ids, block_ids, window_timestamps, window_size_sec
        )
    else:
        raise ValueError(f"Unknown CV strategy: '{strategy}'")

    results: List[CVFoldResult] = []
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)

        results.append(CVFoldResult(
            fold_idx=fold_idx,
            train_idx=train_idx,
            test_idx=test_idx,
            held_out_block_ids=np.unique(block_ids[test_idx]),
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            accuracy=float(acc),
            balanced_accuracy=float(bacc),
        ))
        logger.info(
            "Fold %d: acc=%.3f | bacc=%.3f | test_blocks=%s",
            fold_idx, acc, bacc,
            np.unique(block_ids[test_idx]).tolist(),
        )

    return results


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def summarise_cv(results: List[CVFoldResult]) -> Dict:
    """
    Compute mean ± 95 % bootstrap CI across folds.

    Returns
    -------
    dict with keys: n_folds, mean_accuracy, mean_balanced_accuracy,
                    std_accuracy, std_balanced_accuracy
    """
    accs  = np.array([r.accuracy           for r in results])
    baccs = np.array([r.balanced_accuracy   for r in results])
    return {
        "n_folds":              len(results),
        "mean_accuracy":        float(accs.mean()),
        "std_accuracy":         float(accs.std()),
        "mean_balanced_accuracy": float(baccs.mean()),
        "std_balanced_accuracy":  float(baccs.std()),
    }
