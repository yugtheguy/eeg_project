"""
experiments/protocol.py
========================
Experimental trial protocol generator for the EEG attention decoder.

Block design
------------
* Randomised mini-blocks of 5–8 seconds.
* Four trial types (labels): Left, Right, Neutral, Catch, Bilateral.
* Catch trials (silent):   20 % of blocks.
* Bilateral trials:        20 % of blocks.
* Remaining 60 % split between Left, Right, Neutral.
* Neutral condition is included to allow 3-class decoding.

Each block is assigned a unique integer ``block_id`` that is used by
the cross-validation module to prevent temporal leakage between folds.

Multi-subject design
--------------------
``generate_session_protocol`` accepts ``subject_id`` and ``session_id``
and generates reproducible sequences via ``subject_seed``.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class TrialBlock:
    """A single experimental mini-block."""
    block_id: int
    trial_type: str          # "left" | "right" | "neutral" | "catch" | "bilateral"
    label: int               # integer class label
    duration_sec: float
    onset_sec: float         # onset relative to session start
    offset_sec: float        # onset + duration


@dataclass
class SessionProtocol:
    """Complete trial sequence for one experimental session."""
    subject_id: str
    session_id: str
    blocks: List[TrialBlock]
    total_duration_sec: float
    n_blocks: int
    label_counts: Dict[str, int]
    seed: int


# ---------------------------------------------------------------------------
# Protocol generator
# ---------------------------------------------------------------------------

_LABEL_MAP = {
    "left":      0,
    "right":     1,
    "neutral":   2,
    "catch":     3,
    "bilateral": 4,
}


def generate_session_protocol(
    n_blocks: int,
    subject_id: str,
    session_id: str,
    block_min_sec: float = 5.0,
    block_max_sec: float = 8.0,
    catch_fraction: float = 0.20,
    bilateral_fraction: float = 0.20,
    seed: Optional[int] = None,
) -> SessionProtocol:
    """
    Generate a randomised block-design protocol.

    Parameters
    ----------
    n_blocks : int
        Total number of trial blocks in the session.
    subject_id, session_id : str
        Used to derive a deterministic per-subject seed when ``seed``
        is not supplied.
    block_min_sec, block_max_sec : float
        Range of block durations.
    catch_fraction, bilateral_fraction : float
        Fraction of blocks assigned to catch and bilateral conditions.
        Left–right–neutral split takes the remaining fraction equally.
    seed : int, optional
        If None, derived from a hash of subject_id + session_id.

    Returns
    -------
    SessionProtocol
    """
    if catch_fraction + bilateral_fraction >= 1.0:
        raise ValueError(
            "catch_fraction + bilateral_fraction must be < 1.0."
        )
    if n_blocks < 5:
        raise ValueError("Need at least 5 blocks for a valid protocol.")

    if seed is None:
        seed = _subject_seed(subject_id, session_id)

    rng = np.random.default_rng(seed)

    # ── Allocate trial types ───────────────────────────────────────────────
    n_catch     = max(1, round(n_blocks * catch_fraction))
    n_bilateral = max(1, round(n_blocks * bilateral_fraction))
    n_directed  = n_blocks - n_catch - n_bilateral   # left + right + neutral

    n_left    = n_directed // 3
    n_right   = n_directed // 3
    n_neutral = n_directed - n_left - n_right

    types: List[str] = (
        ["left"]      * n_left    +
        ["right"]     * n_right   +
        ["neutral"]   * n_neutral +
        ["catch"]     * n_catch   +
        ["bilateral"] * n_bilateral
    )
    # Pad or trim to exactly n_blocks if rounding caused mismatch
    while len(types) < n_blocks:
        types.append(rng.choice(["left", "right"]))
    types = types[:n_blocks]

    # ── Shuffle the block order ────────────────────────────────────────────
    rng.shuffle(types)

    # ── Assign durations and onset times ──────────────────────────────────
    durations = rng.uniform(block_min_sec, block_max_sec, size=n_blocks)
    blocks: List[TrialBlock] = []
    cursor = 0.0

    for bid, (t_type, dur) in enumerate(zip(types, durations)):
        block = TrialBlock(
            block_id=bid,
            trial_type=t_type,
            label=_LABEL_MAP[t_type],
            duration_sec=float(dur),
            onset_sec=float(cursor),
            offset_sec=float(cursor + dur),
        )
        blocks.append(block)
        cursor += dur

    label_counts = {t: types.count(t) for t in _LABEL_MAP}

    protocol = SessionProtocol(
        subject_id=subject_id,
        session_id=session_id,
        blocks=blocks,
        total_duration_sec=float(cursor),
        n_blocks=n_blocks,
        label_counts=label_counts,
        seed=seed,
    )
    logger.info(
        "Protocol generated: %d blocks, %.1f s total, seed=%d",
        n_blocks, cursor, seed,
    )
    return protocol


# ---------------------------------------------------------------------------
# Protocol I/O
# ---------------------------------------------------------------------------

def save_protocol(protocol: SessionProtocol, output_dir: Path) -> Path:
    """
    Save protocol to JSON in the session directory.

    Parameters
    ----------
    protocol : SessionProtocol
    output_dir : Path
        Root data directory; file written to
        ``<output_dir>/<subject_id>/<session_id>/protocol.json``.

    Returns
    -------
    path : Path
    """
    session_dir = Path(output_dir) / protocol.subject_id / protocol.session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / "protocol.json"

    doc = {
        "subject_id": protocol.subject_id,
        "session_id": protocol.session_id,
        "seed": protocol.seed,
        "n_blocks": protocol.n_blocks,
        "total_duration_sec": protocol.total_duration_sec,
        "label_counts": protocol.label_counts,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "blocks": [
            {
                "block_id":    b.block_id,
                "trial_type":  b.trial_type,
                "label":       b.label,
                "onset_sec":   round(b.onset_sec, 4),
                "offset_sec":  round(b.offset_sec, 4),
                "duration_sec": round(b.duration_sec, 4),
            }
            for b in protocol.blocks
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    logger.info("Protocol saved: %s", path)
    return path


def load_protocol(path: Path) -> SessionProtocol:
    """Load a previously saved protocol from JSON."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    blocks = [
        TrialBlock(
            block_id=b["block_id"],
            trial_type=b["trial_type"],
            label=b["label"],
            duration_sec=b["duration_sec"],
            onset_sec=b["onset_sec"],
            offset_sec=b["offset_sec"],
        )
        for b in doc["blocks"]
    ]
    return SessionProtocol(
        subject_id=doc["subject_id"],
        session_id=doc["session_id"],
        blocks=blocks,
        total_duration_sec=doc["total_duration_sec"],
        n_blocks=doc["n_blocks"],
        label_counts=doc["label_counts"],
        seed=doc["seed"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subject_seed(subject_id: str, session_id: str) -> int:
    """Derive a deterministic integer seed from subject + session string."""
    combined = f"{subject_id}_{session_id}"
    # Use sum of UTF-8 bytes mod 2^31 for portability
    raw = sum(b * (i + 1) for i, b in enumerate(combined.encode("utf-8")))
    return raw % (2 ** 31)
