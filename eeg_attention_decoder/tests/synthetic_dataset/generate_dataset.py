"""
generate_dataset.py
===================
Generate a fully synthetic EEG dataset that exactly matches the format
expected by ``main.py train``.  No existing project files are modified.

Generated files
---------------
data/raw/sub-sim/ses-01/
    raw_eeg_simulated.csv   – 250 Hz, T7+T8 channels, ~20 min
    protocol.json           – block design matching protocol.py format
    iaf.json                – individual alpha frequency = 10.2 Hz

Signal model
------------
Each channel contains:
  * Delta (3 Hz), theta (6 Hz), beta (20 Hz) oscillations
  * Pink noise (1/f spectral shape)
  * White (Gaussian) noise
  * Alpha oscillation at IAF_HZ (10.2 Hz) whose amplitude is
    modulated per block to simulate spatial attention lateralization:

    LEFT  attention -> T8 alpha +35%, T7 alpha -10%   (right hemisphere ↑)
    RIGHT attention -> T7 alpha +35%, T8 alpha -10%   (left hemisphere ↑)

Usage
-----
Run from the eeg_attention_decoder/ directory::

    python tests/synthetic_dataset/generate_dataset.py

Or import and call ``generate_all()`` from another script.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow importing from the eeg_attention_decoder package
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent          # tests/synthetic_dataset/
PKG_ROOT = HERE.parent.parent                       # eeg_attention_decoder/
sys.path.insert(0, str(PKG_ROOT))

from experiments.protocol import generate_session_protocol, save_protocol

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUBJECT_ID  = "sub-sim"
SESSION_ID  = "ses-01"
FS          = 250            # Hz — must match config sampling_rate
IAF_HZ      = 10.2          # Individual alpha frequency (within 6–14 Hz range)
N_BLOCKS    = 160            # ~13–21 min depending on random block durations
RANDOM_SEED = 42

# Alpha modulation per condition (applied multiplicatively to amplitude)
ALPHA_BOOST = 0.35           # contralateral hemisphere: +35%
ALPHA_SUPP  = 0.10           # ipsilateral hemisphere:  -10%
JITTER_STD  = 0.18           # per-block amplitude jitter (σ, fraction of boost)

# Oscillatory component amplitudes (µV)
AMP_DELTA      = 35.0        # 3 Hz
AMP_THETA      = 22.0        # 6 Hz
AMP_ALPHA_BASE = 22.0        # 10.2 Hz (modulated by attention envelope)
AMP_BETA       = 9.0         # 20 Hz
AMP_PINK       = 28.0        # 1/f noise sigma
AMP_WHITE      = 6.0         # white noise sigma


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _pink_noise(n: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """
    Generate 1/f (pink) noise via spectral shaping of white noise.
    DC bin is set to 1.0 to avoid division by zero.
    """
    white = rng.standard_normal(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    freqs[0] = 1.0
    fft /= np.sqrt(freqs)
    pink = np.fft.irfft(fft, n=n)
    if pink.std() > 0:
        pink *= sigma / pink.std()
    return pink.astype(np.float64)


def _build_alpha_envelope(
    blocks,
    n_samples: int,
    rng: np.random.Generator,
    channel: str,   # "T7" or "T8"
) -> np.ndarray:
    """
    Build a per-sample multiplicative amplitude envelope for the alpha
    oscillation on one channel, based on the block trial types.

    Envelope = 1.0 (baseline) modulated per block with a 200 ms
    half-Hann onset ramp to avoid sharp amplitude discontinuities.
    """
    env = np.ones(n_samples, dtype=np.float64)

    for block in blocks:
        i_s = int(block.onset_sec  * FS)
        i_e = int(block.offset_sec * FS)
        i_s = max(0, i_s)
        i_e = min(n_samples, i_e)
        if i_e - i_s < 2:
            continue

        # Per-block random jitter in modulation depth
        j = 1.0 + JITTER_STD * rng.standard_normal()

        if block.trial_type == "left":
            mult = (1.0 + ALPHA_BOOST * j) if channel == "T8" else (1.0 - ALPHA_SUPP * j)
        elif block.trial_type == "right":
            mult = (1.0 + ALPHA_BOOST * j) if channel == "T7" else (1.0 - ALPHA_SUPP * j)
        else:
            # neutral / catch / bilateral: no lateralisation
            mult = 1.0

        env[i_s:i_e] = mult

        # 200 ms onset ramp (half-Hann window) to smooth the step
        ramp = min(int(0.20 * FS), (i_e - i_s) // 3)
        if ramp > 1:
            hann_half = np.hanning(ramp * 2)[:ramp]          # 0 → 1
            env[i_s : i_s + ramp] = 1.0 + (mult - 1.0) * hann_half

    # Smooth the full envelope with a 150 ms box filter to remove remaining
    # sharp transitions (simulates gradual attention shifts)
    kernel_len = max(2, int(0.15 * FS))
    kernel = np.ones(kernel_len, dtype=np.float64) / kernel_len
    env = np.convolve(env, kernel, mode="same")
    return env


def _build_channel(
    n_samples: int,
    rng: np.random.Generator,
    alpha_env: np.ndarray,
) -> np.ndarray:
    """
    Synthesise one EEG channel.

    Components
    ----------
    - Delta, theta, beta oscillations (fixed frequency, random phase)
    - Alpha oscillation at IAF_HZ with per-sample amplitude envelope
    - Pink noise (1/f)
    - White noise

    All phases are randomised independently for each channel so T7 and T8
    are not perfectly correlated (realistic for real EEG).
    """
    t = np.arange(n_samples, dtype=np.float64) / FS

    def _osc(freq, amp):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return amp * np.sin(2.0 * np.pi * freq * t + phase)

    # Alpha: amplitude modulated by envelope
    alpha_phase = rng.uniform(0.0, 2.0 * np.pi)
    alpha_sig   = AMP_ALPHA_BASE * alpha_env * np.sin(
        2.0 * np.pi * IAF_HZ * t + alpha_phase
    )

    sig = (
        _osc(3.0,  AMP_DELTA)
        + _osc(6.0,  AMP_THETA)
        + _osc(20.0, AMP_BETA)
        + alpha_sig
        + _pink_noise(n_samples, rng, AMP_PINK)
        + rng.standard_normal(n_samples) * AMP_WHITE
    )
    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_all(output_root: Optional[Path] = None) -> Dict:
    """
    Generate protocol.json, raw_eeg_simulated.csv, and iaf.json.

    Parameters
    ----------
    output_root : Path, optional
        Root directory for data (default: ``<PKG_ROOT>/data/raw``).

    Returns
    -------
    dict with keys:
        protocol, protocol_path, csv_path, iaf_path,
        n_samples, t7, t8, wall_times
    """
    if output_root is None:
        output_root = PKG_ROOT / "data" / "raw"

    session_dir = output_root / SUBJECT_ID / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Generate experimental protocol ─────────────────────────────────
    print("Generating protocol …")
    protocol = generate_session_protocol(
        n_blocks=N_BLOCKS,
        subject_id=SUBJECT_ID,
        session_id=SESSION_ID,
        block_min_sec=5.0,
        block_max_sec=8.0,
        catch_fraction=0.20,
        bilateral_fraction=0.20,
        seed=RANDOM_SEED,
    )
    proto_path = save_protocol(protocol, output_root)
    print(
        f"  Protocol: {proto_path.name}  "
        f"({protocol.n_blocks} blocks, "
        f"{protocol.total_duration_sec:.0f} s = "
        f"{protocol.total_duration_sec / 60:.1f} min)"
    )
    print(f"  Label counts: {protocol.label_counts}")

    # ── 2. Synthesise EEG ─────────────────────────────────────────────────
    # Add 2 s of padding at the end so windows don't run off the edge
    n_samples = int(np.ceil(protocol.total_duration_sec * FS)) + 2 * FS
    print(f"\nSynthesising {n_samples} samples ({n_samples / FS:.1f} s at {FS} Hz) …")

    env_t7 = _build_alpha_envelope(protocol.blocks, n_samples, rng, "T7")
    env_t8 = _build_alpha_envelope(protocol.blocks, n_samples, rng, "T8")

    t7 = _build_channel(n_samples, rng, env_t7)
    t8 = _build_channel(n_samples, rng, env_t8)

    # Wall-clock timestamps: evenly spaced at FS Hz, starting at t=0
    wall_times = np.arange(n_samples, dtype=np.float64) / FS

    # ── 3. Write CSV ──────────────────────────────────────────────────────
    csv_path = session_dir / "raw_eeg_simulated.csv"
    print(f"Writing CSV: {csv_path} …")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        # Comment-prefixed metadata header (skipped by pandas comment='#')
        fh.write(f"# subject_id: {SUBJECT_ID}\n")
        fh.write(f"# session_id: {SESSION_ID}\n")
        fh.write(f"# generated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fh.write(f"# sampling_rate_hz: {FS}\n")
        fh.write(f"# n_samples: {n_samples}\n")
        fh.write(f"# channel_names: ['T7', 'T8']\n")
        fh.write(f"# iaf_hz: {IAF_HZ}\n")
        fh.write(f"# synthetic: true\n")
        fh.write(f"# alpha_boost: {ALPHA_BOOST}\n")
        fh.write(f"# alpha_supp: {ALPHA_SUPP}\n")

        writer = csv.writer(fh)
        # Match the exact column names produced by SerialReader._init_csv
        writer.writerow(["wall_time_s", "arduino_timestamp_ms", "seq_id", "T7", "T8"])

        for i in range(n_samples):
            writer.writerow([
                f"{wall_times[i]:.6f}",
                int(wall_times[i] * 1000),   # simulated Arduino millis()
                i % 65536,                   # simulated seq_id (16-bit wrap)
                f"{t7[i]:.4f}",
                f"{t8[i]:.4f}",
            ])

    size_mb = csv_path.stat().st_size / 1e6
    print(f"  Written: {csv_path.name}  ({size_mb:.1f} MB)")

    # ── 4. Write IAF ──────────────────────────────────────────────────────
    iaf_path = session_dir / "iaf.json"
    iaf_data = {
        "iaf_hz": IAF_HZ,
        "peak_power_db": -4.8,
        "search_range_hz": [6.0, 14.0],
        "used_default": False,
        "source": "synthetic_fixed",
        "note": "Fixed at simulation frequency; not estimated from data.",
    }
    with open(iaf_path, "w", encoding="utf-8") as fh:
        json.dump(iaf_data, fh, indent=2)
    print(f"  IAF: {iaf_path.name}  (iaf_hz={IAF_HZ})")

    # ── 5. Sanity-check lateralisation (raw, pre-filter) ─────────────────
    left_li, right_li = [], []
    for block in protocol.blocks:
        i_s = int(block.onset_sec * FS)
        i_e = int(block.offset_sec * FS)
        if i_s >= i_e or i_e > n_samples:
            continue
        a7 = float(np.std(t7[i_s:i_e]))
        a8 = float(np.std(t8[i_s:i_e]))
        li = (a7 - a8) / (a7 + a8 + 1e-9)
        if block.trial_type == "left":
            left_li.append(li)
        elif block.trial_type == "right":
            right_li.append(li)

    print("\nLateralisation check (raw signal, pre-filter):")
    if left_li:
        print(f"  LEFT  blocks: mean LI = {np.mean(left_li):+.3f}  (expect ≤ 0)")
    if right_li:
        print(f"  RIGHT blocks: mean LI = {np.mean(right_li):+.3f}  (expect ≥ 0)")

    print("\nDataset ready.")
    return {
        "protocol":      protocol,
        "protocol_path": proto_path,
        "csv_path":      csv_path,
        "iaf_path":      iaf_path,
        "n_samples":     n_samples,
        "t7":            t7,
        "t8":            t8,
        "wall_times":    wall_times,
        "left_li":       left_li,
        "right_li":      right_li,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    result = generate_all()
    print("\nNext step — run the full training pipeline:")
    print(f"  cd {PKG_ROOT}")
    print(
        "  python main.py train --subject sub-sim --session ses-01 "
        "--data-dir data/raw"
    )
