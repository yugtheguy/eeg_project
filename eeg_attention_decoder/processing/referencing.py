"""
processing/referencing.py
==========================
Electrode referencing routines.

Rules
-----
* n_channels == 2  → ``linked_mastoid`` is a no-op (reference applied
  in hardware).  Function is kept for pipeline uniformity.
* n_channels >= 3  → Common Average Reference (CAR) is available.
* No hardcoded channel count; all logic is driven by the array shape.
* Referencing is applied AFTER filtering (downstream of FilterBank).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

ReferenceType = Literal["linked_mastoid", "CAR"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_reference(
    data: np.ndarray,
    reference_type: ReferenceType,
    n_channels: int,
) -> np.ndarray:
    """
    Apply spatial referencing to an EEG segment.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        Filtered EEG in µV.
    reference_type : {"linked_mastoid", "CAR"}
        Referencing strategy.  ``"linked_mastoid"`` is a no-op for
        2-channel data; ``"CAR"`` requires n_channels >= 3.
    n_channels : int
        Expected number of channels (used for validation).

    Returns
    -------
    referenced : np.ndarray, shape (n_samples, n_channels)
    """
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D array, got shape {data.shape}")
    if data.shape[1] != n_channels:
        raise ValueError(
            f"Expected {n_channels} channels, got {data.shape[1]}"
        )

    if reference_type == "linked_mastoid":
        return _linked_mastoid(data, n_channels)
    elif reference_type == "CAR":
        return _common_average_reference(data, n_channels)
    else:
        raise ValueError(
            f"Unknown reference_type '{reference_type}'. "
            "Choose 'linked_mastoid' or 'CAR'."
        )


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------

def _linked_mastoid(data: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Linked-mastoid reference (hardware).

    For the 2-channel (T7, T8) Phase-1 setup the mastoid reference is
    applied in hardware via the BioAmp EXG Pill reference pin; this
    function is therefore a *passthrough* that documents the referencing
    choice explicitly in the pipeline.

    If n_channels >= 3 (e.g. Cz is added later), the same passthrough
    logic applies — the user is responsible for wiring the reference.
    """
    if n_channels < 2:
        raise ValueError(
            "linked_mastoid referencing requires at least 2 channels."
        )
    if n_channels >= 3:
        logger.info(
            "linked_mastoid passthrough with %d channels — "
            "ensure hardware reference is correctly wired.",
            n_channels,
        )
    return data.copy()


def _common_average_reference(data: np.ndarray, n_channels: int) -> np.ndarray:
    """
    Common Average Reference (CAR).

    Subtracts the mean across all channels at each sample.
    Only meaningful for n_channels >= 3; raises if called with fewer.
    """
    if n_channels < 3:
        raise ValueError(
            f"CAR requires at least 3 channels; got {n_channels}. "
            "Use 'linked_mastoid' for 2-channel setups."
        )
    mean = data.mean(axis=1, keepdims=True)   # (n_samples, 1)
    return data - mean


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def reference_from_config(cfg: dict, data: np.ndarray) -> np.ndarray:
    """
    Apply referencing as specified in the YAML config dict.

    Parameters
    ----------
    cfg : dict
        Full YAML config.  Reads ``processing.reference_type`` and
        ``hardware.n_channels``.
    data : np.ndarray, shape (n_samples, n_channels)

    Returns
    -------
    referenced : np.ndarray
    """
    ref_type: ReferenceType = cfg["processing"]["reference_type"]
    n_channels: int = cfg["hardware"]["n_channels"]
    return apply_reference(data, ref_type, n_channels)
