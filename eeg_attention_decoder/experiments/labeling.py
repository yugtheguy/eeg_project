"""
experiments/labeling.py
========================
EEG label alignment and window annotation.

Temporal alignment strategy
-----------------------------
1. Each trial block has an ``onset_sec`` and ``offset_sec`` from the
   protocol.
2. A physiological delay (default 400 ms) is added to the onset before
   the label is applied — the EEG response lags the stimulus.
3. Labels are assigned to *analysis windows* whose **centre timestamp**
   falls within ``[onset_sec + delay, offset_sec]``.
4. Windows that straddle a block boundary or fall in no block are
   assigned label ``-1`` (UNLABELED) and MUST be excluded before
   training.  This prevents cross-block contamination.

This module never modifies raw EEG data — it only produces a label
array aligned to the windows returned by ``processing.features.extract_windows``.

Multi-subject
-------------
``assign_labels`` is session-agnostic; caller passes the protocol loaded
for the correct subject/session.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from experiments.protocol import SessionProtocol, TrialBlock

logger = logging.getLogger(__name__)

UNLABELED: int = -1


# ---------------------------------------------------------------------------
# Primary labeling function
# ---------------------------------------------------------------------------

def assign_labels(
    center_timestamps: np.ndarray,
    protocol: SessionProtocol,
    physiological_delay_sec: float = 0.4,
    session_start_time: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign integer class labels to analysis windows.

    Parameters
    ----------
    center_timestamps : np.ndarray, shape (n_windows,)
        Wall-clock or relative timestamps (seconds) at the centre of
        each analysis window.  Must be monotonically increasing.
    protocol : SessionProtocol
        Trial block sequence for this session.
    physiological_delay_sec : float
        EEG response lag relative to stimulus onset (default 400 ms).
    session_start_time : float
        Absolute time (seconds) of the first sample, used to convert
        relative protocol times to absolute timestamps.

    Returns
    -------
    labels : np.ndarray, shape (n_windows,)
        Integer labels; ``-1`` (UNLABELED) for boundary/excluded windows.
    block_ids : np.ndarray, shape (n_windows,)
        Integer block ID for each labelled window; ``-1`` for UNLABELED.

    Notes
    -----
    * Windows labelled -1 MUST be removed before cross-validation.
    * ``block_ids`` is propagated to the CV module to ensure block
      integrity across folds.
    """
    if center_timestamps.ndim != 1:
        raise ValueError("center_timestamps must be 1-D.")
    if not np.all(np.diff(center_timestamps) > 0):
        raise ValueError("center_timestamps must be strictly increasing.")

    n_windows = len(center_timestamps)
    labels = np.full(n_windows, UNLABELED, dtype=np.int32)
    block_ids = np.full(n_windows, UNLABELED, dtype=np.int32)

    # Convert protocol-relative times to absolute wall-clock times
    blocks = protocol.blocks

    for block in blocks:
        abs_onset  = session_start_time + block.onset_sec  + physiological_delay_sec
        abs_offset = session_start_time + block.offset_sec

        if abs_onset >= abs_offset:
            # Delay pushes onset past offset — block too short to use
            logger.debug(
                "Block %d skipped: delay pushes onset past offset.", block.block_id
            )
            continue

        # Windows whose centre falls inside [abs_onset, abs_offset)
        mask = (center_timestamps >= abs_onset) & (center_timestamps < abs_offset)
        labels[mask]    = block.label
        block_ids[mask] = block.block_id

    n_labeled   = int(np.sum(labels >= 0))
    n_unlabeled = n_windows - n_labeled
    logger.info(
        "Label assignment: %d labelled | %d unlabeled (boundary/excluded) "
        "| delay=%.0f ms",
        n_labeled, n_unlabeled, physiological_delay_sec * 1000,
    )
    return labels, block_ids


# ---------------------------------------------------------------------------
# Filter out unlabeled windows
# ---------------------------------------------------------------------------

def filter_labeled(
    X: np.ndarray,
    labels: np.ndarray,
    block_ids: np.ndarray,
    center_timestamps: np.ndarray,
    exclude_labels: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Remove UNLABELED (−1) windows and optionally exclude specific labels.

    Parameters
    ----------
    X : np.ndarray, shape (n_windows, n_features)
    labels : np.ndarray, shape (n_windows,)
    block_ids : np.ndarray, shape (n_windows,)
    center_timestamps : np.ndarray, shape (n_windows,)
    exclude_labels : list of int, optional
        Additional integer labels to exclude (e.g. ``[3]`` to drop catch
        trials from binary Left-vs-Right decoding).

    Returns
    -------
    X_clean, labels_clean, block_ids_clean, timestamps_clean
        All with the same n_windows (after filtering).
    """
    keep = labels != UNLABELED
    if exclude_labels:
        for lbl in exclude_labels:
            keep &= (labels != lbl)

    n_removed = np.sum(~keep)
    if n_removed > 0:
        logger.info("Removed %d windows (unlabeled or excluded).", n_removed)

    return (
        X[keep],
        labels[keep],
        block_ids[keep],
        center_timestamps[keep],
    )


# ---------------------------------------------------------------------------
# Leakage guard
# ---------------------------------------------------------------------------

def assert_no_boundary_overlap(
    train_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
    window_size_sec: float,
    tolerance_sec: float = 0.0,
) -> None:
    """
    Assert that train and test windows do not overlap in time.

    A train window overlaps a test window if their temporal extents
    (centre ± half_window) intersect.

    Parameters
    ----------
    train_timestamps, test_timestamps : np.ndarray
        Centre timestamps of the train and test sets respectively.
    window_size_sec : float
    tolerance_sec : float
        Extra margin to account for floating-point imprecision.

    Raises
    ------
    AssertionError
        If any overlap is detected.  This will fail a unit test and
        should abort training if raised during CV.
    """
    half = window_size_sec / 2.0 + tolerance_sec
    for ts_test in test_timestamps:
        overlaps = np.any(np.abs(train_timestamps - ts_test) < half)
        if overlaps:
            raise AssertionError(
                f"TEMPORAL LEAKAGE DETECTED: test window at t={ts_test:.4f} s "
                f"overlaps with a training window (window_size={window_size_sec} s)."
            )
