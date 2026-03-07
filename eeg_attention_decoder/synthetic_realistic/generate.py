"""
synthetic_realistic/generate.py
================================
Generate a realistic synthetic EEG dataset calibrated to match the
statistical properties measured from sub-01's real recordings.

Calibration targets (measured from data/raw/sub-01/ses-01/):
  IAF              = 8.5 Hz      (from iaf.json)
  ADC mean         ≈ 354 counts  (T7=353.9, T8=353.6)
  ADC std          ≈ 23 counts   (T7=21.51, T8=26.37)
  Cross-ch Pearson r ≈ 0.48
  Alpha LI effect  ≈ +0.055      (LEFT_LI − RIGHT_LI, from _check_lateralization.py)
  Block durations  = 10–15 s     (same protocol as ses-01)
  n_blocks         = 30

Signal model
------------
Two components of 1/f (pink) noise:
  * Shared pink noise  (σ_s=25) — identical in T7 and T8 → drives cross-ch correlation
  * Individual pink noise (σ_i=18) — independent per channel
  * White noise (σ=5) — per channel
  * Delta (3 Hz, 16 µV), Theta (6 Hz, 10 µV), Beta (20 Hz, 6 µV)  — independent phases
  * Alpha (IAF=8.5 Hz, 16 µV base) — amplitude modulated by attention block

Alpha lateralisation
--------------------
  LEFT  attention: T8 alpha +8%, T7 alpha −2%   (right hemisphere ↑)
  RIGHT attention: T7 alpha +8%, T8 alpha −2%   (left hemisphere ↑)
  High jitter (JITTER_STD=0.40) simulates realistic block-to-block variability.

Expected accuracy: 50–60% balanced (realistic, not inflated).

Output
------
  data/raw/sub-realistic/ses-01/
    raw_eeg_synthetic.csv
    protocol.json
    iaf.json
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — import package modules
# ---------------------------------------------------------------------------
HERE     = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent
sys.path.insert(0, str(PKG_ROOT))

from experiments.protocol import generate_session_protocol, save_protocol

# ---------------------------------------------------------------------------
# Calibration constants (derived from real sub-01 data)
# ---------------------------------------------------------------------------
SUBJECT_ID  = "sub-realistic"
SESSION_ID  = "ses-01"
FS          = 250           # Hz — must match config sampling_rate

# Real-data measured values
IAF_HZ      = 8.5           # From data/raw/sub-01/ses-01/iaf.json
ADC_MEAN    = 354.0         # Raw Arduino ADC offset (~1.73 V at 5V Vref)
ADC_STD_T7  = 21.5          # Target ADC std for T7 channel
ADC_STD_T8  = 26.4          # Target ADC std for T8 channel

# Protocol: same block count and duration as real session
N_BLOCKS        = 30
BLOCK_MIN_SEC   = 10.0
BLOCK_MAX_SEC   = 15.0
PROTOCOL_SEED   = 1234      # Different seed → different block order from sub-01

# ---------------------------------------------------------------------------
# Signal component amplitudes (µV units; will be scaled to ADC)
# These are tuned so that cross-channel Pearson r ≈ 0.48
# ---------------------------------------------------------------------------
AMP_PINK_SHARED = 25.0  # σ — shared 1/f noise (both channels identical)
AMP_PINK_INDIV  = 18.0  # σ — independent 1/f noise per channel
AMP_WHITE       = 5.0   # σ — white Gaussian noise per channel
AMP_DELTA       = 16.0  # 3 Hz oscillation amplitude (µV peak)
AMP_THETA       = 10.0  # 6 Hz
AMP_ALPHA_BASE  = 16.0  # 8.5 Hz — modulated per block (µV peak)
AMP_BETA        = 6.0   # 20 Hz

# Alpha modulation: calibrated to produce LI_effect ≈ +0.055
# (matching the real data; weak enough for ~50–60% CV accuracy)
ALPHA_BOOST = 0.22   # contralateral hemisphere alpha +22%
ALPHA_SUPP  = 0.06   # ipsilateral  hemisphere alpha −6%
JITTER_STD  = 0.30   # per-block modulation jitter (fraction) — moderate for realism

RANDOM_SEED = 99


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------

def _pink_noise(n: int, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """Generate 1/f pink noise via spectral shaping of white noise."""
    white  = rng.standard_normal(n)
    fft    = np.fft.rfft(white)
    freqs  = np.fft.rfftfreq(n, d=1.0 / FS)
    freqs[0] = 1.0
    fft   /= np.sqrt(freqs)
    pink   = np.fft.irfft(fft, n=n)
    if pink.std() > 1e-12:
        pink *= sigma / pink.std()
    return pink.astype(np.float64)


def _build_alpha_envelope(
    blocks,
    n_samples: int,
    rng: np.random.Generator,
    channel: str,   # "T7" or "T8"
) -> np.ndarray:
    """
    Per-sample multiplicative amplitude envelope for the alpha oscillation.
    Identical logic to the reference generator but with calibrated boost/supp.
    """
    env = np.ones(n_samples, dtype=np.float64)

    for block in blocks:
        i_s = max(0, int(block.onset_sec  * FS))
        i_e = min(n_samples, int(block.offset_sec * FS))
        if i_e - i_s < 2:
            continue

        j = 1.0 + JITTER_STD * rng.standard_normal()   # per-block random jitter

        if block.trial_type == "left":
            # LEFT attention: right hemisphere (T8) alpha increases, T7 decreases
            mult = (1.0 + ALPHA_BOOST * j) if channel == "T8" else (1.0 - ALPHA_SUPP * j)
        elif block.trial_type == "right":
            # RIGHT attention: left hemisphere (T7) alpha increases, T8 decreases
            mult = (1.0 + ALPHA_BOOST * j) if channel == "T7" else (1.0 - ALPHA_SUPP * j)
        else:
            mult = 1.0   # neutral / catch / bilateral

        env[i_s:i_e] = mult

        # 200 ms onset ramp to smooth step transitions
        ramp = min(int(0.20 * FS), (i_e - i_s) // 3)
        if ramp > 1:
            hann = np.hanning(ramp * 2)[:ramp]
            env[i_s : i_s + ramp] = 1.0 + (mult - 1.0) * hann

    # 150 ms box-filter smoothing (simulates sluggish attention shifts)
    k = max(2, int(0.15 * FS))
    env = np.convolve(env, np.ones(k) / k, mode="same")
    return env


def _build_channel(
    n_samples: int,
    rng: np.random.Generator,
    alpha_env: np.ndarray,
    shared_pink: np.ndarray,
    target_std: float,
) -> np.ndarray:
    """
    Synthesise one EEG channel.

    Includes shared + individual pink noise (for calibrated cross-channel
    correlation), white noise, and oscillations at delta / theta / alpha / beta.
    Signal is normalised so std matches target ADC std, then DC offset added.
    """
    t = np.arange(n_samples, dtype=np.float64) / FS

    def _osc(freq: float, amp: float) -> np.ndarray:
        phase = rng.uniform(0.0, 2.0 * np.pi)
        return amp * np.sin(2.0 * np.pi * freq * t + phase)

    alpha_phase = rng.uniform(0.0, 2.0 * np.pi)
    alpha_sig   = AMP_ALPHA_BASE * alpha_env * np.sin(
        2.0 * np.pi * IAF_HZ * t + alpha_phase
    )

    indiv_pink = _pink_noise(n_samples, rng, AMP_PINK_INDIV)
    white      = rng.standard_normal(n_samples) * AMP_WHITE

    sig = (
        shared_pink                  # shared 1/f (drives cross-channel correlation)
        + indiv_pink                 # per-channel independent 1/f
        + white                      # per-channel white noise
        + _osc(3.0,  AMP_DELTA)
        + _osc(6.0,  AMP_THETA)
        + alpha_sig
        + _osc(20.0, AMP_BETA)
    )

    # Normalise to target ADC std then shift to ADC mean
    sig_std = sig.std()
    if sig_std > 1e-12:
        sig = sig * (target_std / sig_std)
    sig += ADC_MEAN

    return sig.astype(np.float64)


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_all(output_root: Optional[Path] = None) -> dict:
    """
    Generate protocol.json, raw_eeg_synthetic.csv, and iaf.json
    for subject 'sub-realistic', calibrated to real sub-01 recordings.
    """
    if output_root is None:
        output_root = PKG_ROOT / "data" / "raw"

    session_dir = output_root / SUBJECT_ID / SESSION_ID
    session_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── 1. Protocol ──────────────────────────────────────────────────────
    print("Generating protocol …")
    protocol = generate_session_protocol(
        n_blocks=N_BLOCKS,
        subject_id=SUBJECT_ID,
        session_id=SESSION_ID,
        block_min_sec=BLOCK_MIN_SEC,
        block_max_sec=BLOCK_MAX_SEC,
        catch_fraction=0.20,
        bilateral_fraction=0.20,
        seed=PROTOCOL_SEED,
    )
    proto_path = save_protocol(protocol, output_root)
    print(
        f"  Protocol: {N_BLOCKS} blocks, "
        f"{protocol.total_duration_sec:.0f} s = {protocol.total_duration_sec/60:.1f} min"
    )
    print(f"  Label counts: {protocol.label_counts}")

    # ── 2. Synthesise signals ─────────────────────────────────────────────
    n_samples = int(np.ceil(protocol.total_duration_sec * FS)) + 2 * FS
    print(f"\nSynthesising {n_samples} samples ({n_samples/FS:.1f} s at {FS} Hz) …")

    # Shared pink noise — same signal added to both channels
    shared_pink = _pink_noise(n_samples, rng, AMP_PINK_SHARED)

    env_t7 = _build_alpha_envelope(protocol.blocks, n_samples, rng, "T7")
    env_t8 = _build_alpha_envelope(protocol.blocks, n_samples, rng, "T8")

    t7 = _build_channel(n_samples, rng, env_t7, shared_pink, ADC_STD_T7)
    t8 = _build_channel(n_samples, rng, env_t8, shared_pink, ADC_STD_T8)

    wall_times = np.arange(n_samples, dtype=np.float64) / FS

    # ── 3. Sanity check ───────────────────────────────────────────────────
    from scipy.signal import welch as _welch

    corr = float(np.corrcoef(t7, t8)[0, 1])
    freqs, psd0 = _welch(t7, fs=FS, nperseg=512)
    freqs, psd1 = _welch(t8, fs=FS, nperseg=512)
    amask = (freqs >= IAF_HZ - 2) & (freqs <= IAF_HZ + 2)
    a7, a8 = psd0[amask].mean(), psd1[amask].mean()
    global_li = float((a8 - a7) / (a8 + a7 + 1e-30))

    left_li, right_li = [], []
    for blk in protocol.blocks:
        if blk.trial_type not in ("left", "right"):
            continue
        i_s = int(blk.onset_sec * FS)
        i_e = int(blk.offset_sec * FS)
        seg0, seg1 = t7[i_s:i_e], t8[i_s:i_e]
        if len(seg0) < 50:
            continue
        nperseg = min(256, len(seg0) // 2)
        fr, p0 = _welch(seg0, fs=FS, nperseg=nperseg)
        fr, p1 = _welch(seg1, fs=FS, nperseg=nperseg)
        am = (fr >= IAF_HZ - 2) & (fr <= IAF_HZ + 2)
        a0, a1 = p0[am].mean(), p1[am].mean()
        li = float((a1 - a0) / (a1 + a0 + 1e-30))
        (left_li if blk.trial_type == "left" else right_li).append(li)

    print(f"\n  ADC stats:")
    print(f"    T7: mean={t7.mean():.1f}  std={t7.std():.2f}  (target mean={ADC_MEAN}, std={ADC_STD_T7})")
    print(f"    T8: mean={t8.mean():.1f}  std={t8.std():.2f}  (target mean={ADC_MEAN}, std={ADC_STD_T8})")
    print(f"  Cross-channel Pearson r: {corr:.4f}  (target ≈ 0.48)")
    if left_li and right_li:
        li_effect = np.mean(left_li) - np.mean(right_li)
        print(f"  Alpha LI effect (LEFT−RIGHT): {li_effect:+.4f}  (target ≈ +0.055)")
        print(f"    LEFT  blocks mean LI: {np.mean(left_li):+.4f}")
        print(f"    RIGHT blocks mean LI: {np.mean(right_li):+.4f}")

    # ── 4. Write CSV ──────────────────────────────────────────────────────
    csv_path = session_dir / "raw_eeg_synthetic.csv"
    print(f"\nWriting CSV: {csv_path.name} …")

    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        fh.write(f"# subject_id: {SUBJECT_ID}\n")
        fh.write(f"# session_id: {SESSION_ID}\n")
        fh.write(f"# generated_utc: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        fh.write(f"# sampling_rate_hz: {FS}\n")
        fh.write(f"# n_samples: {n_samples}\n")
        fh.write(f"# channel_names: ['T7', 'T8']\n")
        fh.write(f"# iaf_hz: {IAF_HZ}\n")
        fh.write(f"# synthetic: true\n")
        fh.write(f"# calibrated_to: sub-01_ses-01\n")
        fh.write(f"# alpha_boost: {ALPHA_BOOST}\n")
        fh.write(f"# alpha_supp: {ALPHA_SUPP}\n")
        fh.write(f"# cross_channel_r_measured: {corr:.4f}\n")

        writer = csv.writer(fh)
        writer.writerow(["wall_time_s", "arduino_timestamp_ms", "seq_id", "T7", "T8"])
        for i in range(n_samples):
            writer.writerow([
                f"{wall_times[i]:.6f}",
                int(wall_times[i] * 1000),
                i % 65536,
                f"{t7[i]:.2f}",
                f"{t8[i]:.2f}",
            ])

    print(f"  Written: {csv_path.stat().st_size / 1e6:.1f} MB")

    # ── 5. Write IAF ──────────────────────────────────────────────────────
    iaf_path = session_dir / "iaf.json"
    with open(iaf_path, "w", encoding="utf-8") as fh:
        json.dump({
            "iaf_hz":         IAF_HZ,
            "peak_power_db":  12.93,   # Same peak power DB as real sub-01
            "search_range_hz": [6.0, 14.0],
            "used_default":   False,
            "source":         "calibrated_from_sub-01",
            "note":           "IAF fixed at real sub-01 measured value (8.5 Hz).",
        }, fh, indent=2)
    print(f"  IAF: {iaf_path.name}  (iaf_hz={IAF_HZ})")

    print("\n=== Synthetic dataset ready ===")
    print(f"  Output dir: {session_dir}")
    print(f"\nNext — train the model:")
    print(f"  python main.py train --subject {SUBJECT_ID} --session {SESSION_ID} --data-dir data/raw")

    return {
        "protocol":      protocol,
        "csv_path":      csv_path,
        "iaf_path":      iaf_path,
        "n_samples":     n_samples,
        "t7":            t7,
        "t8":            t8,
        "wall_times":    wall_times,
    }


if __name__ == "__main__":
    generate_all()
