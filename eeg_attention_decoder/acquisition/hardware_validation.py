"""
acquisition/hardware_validation.py
====================================
Hardware qualification routines that must pass before any experimental
data is accepted.  All results are saved to a structured JSON report.

Functions
---------
measure_sampling_stability(packets, nominal_fs) -> dict
detect_packet_loss(packets) -> dict
compute_noise_floor(signal, fs) -> dict
estimate_alpha_snr(signal, fs, iaf_hz, iaf_bw) -> dict
run_all_validations(...) -> dict    ← convenience wrapper
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import iqr

from acquisition.serial_reader import EEGPacket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual validation functions
# ---------------------------------------------------------------------------

def measure_sampling_stability(
    packets: Sequence[EEGPacket],
    nominal_fs: float = 250.0,
) -> Dict:
    """
    Compute inter-packet interval statistics from a sequence of packets.

    Parameters
    ----------
    packets : sequence of EEGPacket
        Must be in chronological order and contain at least 10 packets.
    nominal_fs : float
        Expected sampling rate in Hz.

    Returns
    -------
    dict with keys:
        n_packets, nominal_fs, effective_fs,
        iti_mean_ms, iti_std_ms, iti_cv,
        iti_p95_ms, iti_p99_ms,
        ppm_drift, passed
    """
    if len(packets) < 10:
        raise ValueError("Need at least 10 packets to assess stability.")

    wall_times = np.array([p.wall_time for p in packets], dtype=np.float64)
    intervals_ms = np.diff(wall_times) * 1000.0

    nominal_iti = 1000.0 / nominal_fs  # ms
    iti_mean = float(np.mean(intervals_ms))
    iti_std = float(np.std(intervals_ms))
    iti_cv = iti_std / iti_mean if iti_mean > 0 else float("inf")
    iti_p95 = float(np.percentile(intervals_ms, 95))
    iti_p99 = float(np.percentile(intervals_ms, 99))

    # Effective sampling rate from total wall-clock span
    total_sec = wall_times[-1] - wall_times[0]
    eff_fs = (len(packets) - 1) / total_sec if total_sec > 0 else 0.0

    # Parts-per-million drift between effective and nominal rate
    ppm_drift = ((eff_fs - nominal_fs) / nominal_fs) * 1e6

    # Acceptance criterion: CV < 0.15, p99 < 2× nominal ITI
    passed = iti_cv < 0.15 and iti_p99 < 2.0 * nominal_iti

    result = {
        "n_packets": len(packets),
        "nominal_fs": nominal_fs,
        "effective_fs": round(eff_fs, 4),
        "iti_mean_ms": round(iti_mean, 4),
        "iti_std_ms": round(iti_std, 4),
        "iti_cv": round(iti_cv, 4),
        "iti_p95_ms": round(iti_p95, 4),
        "iti_p99_ms": round(iti_p99, 4),
        "ppm_drift": round(ppm_drift, 2),
        "passed": passed,
    }
    if not passed:
        logger.warning("Sampling stability FAILED: CV=%.3f, p99=%.1f ms", iti_cv, iti_p99)
    return result


def detect_packet_loss(
    packets: Sequence[EEGPacket],
    seq_max: int = 65536,
) -> Dict:
    """
    Detect dropped packets by inspecting sequence-ID gaps.

    Parameters
    ----------
    packets : sequence of EEGPacket
    seq_max : int
        Roll-over value for the Arduino seq counter (default 2^16).

    Returns
    -------
    dict with keys:
        n_received, n_expected, n_dropped,
        loss_fraction, gap_list, passed
    """
    if len(packets) < 2:
        raise ValueError("Need at least 2 packets to assess packet loss.")

    seqs = [p.seq_id for p in packets]
    dropped = 0
    gaps: List[Dict] = []

    for i in range(1, len(seqs)):
        expected = (seqs[i - 1] + 1) % seq_max
        if seqs[i] != expected:
            gap = (seqs[i] - seqs[i - 1]) % seq_max - 1
            if gap > 0:
                dropped += gap
                gaps.append({"index": i, "seq_prev": seqs[i - 1], "seq_curr": seqs[i], "n_dropped": gap})

    n_expected = (seqs[-1] - seqs[0]) % seq_max
    loss_fraction = dropped / n_expected if n_expected > 0 else 0.0
    passed = loss_fraction < 0.01  # < 1 % loss

    result = {
        "n_received": len(packets),
        "n_expected": n_expected,
        "n_dropped": dropped,
        "loss_fraction": round(loss_fraction, 6),
        "gap_list": gaps[:20],   # cap to keep JSON compact
        "passed": passed,
    }
    if not passed:
        logger.warning("Packet loss FAILED: %.2f%% loss", loss_fraction * 100)
    return result


def compute_noise_floor(
    signal_uv: np.ndarray,
    fs: float,
    band_hz: tuple[float, float] = (100.0, 120.0),
) -> Dict:
    """
    Estimate the broadband noise floor from a frequency band assumed to
    contain no physiological signal (100–120 Hz by default).

    Parameters
    ----------
    signal_uv : np.ndarray, shape (n_samples,) or (n_samples, n_channels)
        Raw EEG in µV.
    fs : float
        Sampling rate in Hz.
    band_hz : (float, float)
        Frequency band for noise estimation.

    Returns
    -------
    dict with keys:
        noise_rms_uv_per_channel, noise_rms_uv_mean,
        dynamic_range_db, passed
    """
    data = np.atleast_2d(signal_uv)
    if data.shape[0] < data.shape[1]:
        data = data.T   # ensure (n_samples, n_channels)

    n_samples, n_channels = data.shape
    nperseg = min(256, n_samples // 4)

    noise_per_ch: List[float] = []
    for ch in range(n_channels):
        freqs, psd = sp_signal.welch(data[:, ch], fs=fs, nperseg=nperseg)
        band_mask = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
        if not np.any(band_mask):
            noise_per_ch.append(float("nan"))
            continue
        noise_power = float(np.mean(psd[band_mask]))
        noise_rms = float(np.sqrt(noise_power * (band_hz[1] - band_hz[0])))
        noise_per_ch.append(noise_rms)

    noise_mean = float(np.nanmean(noise_per_ch))
    # Dynamic range: signal peak-to-peak vs noise RMS
    ptp = float(np.ptp(data))
    dynamic_range_db = 20.0 * np.log10(ptp / noise_mean) if noise_mean > 0 else float("inf")
    passed = noise_mean < 5.0   # < 5 µV RMS noise floor

    return {
        "noise_rms_uv_per_channel": [round(v, 4) for v in noise_per_ch],
        "noise_rms_uv_mean": round(noise_mean, 4),
        "dynamic_range_db": round(dynamic_range_db, 2),
        "noise_band_hz": list(band_hz),
        "passed": passed,
    }


def estimate_alpha_snr(
    signal_uv: np.ndarray,
    fs: float,
    iaf_hz: float,
    iaf_bw: float = 2.0,
    noise_band_hz: tuple[float, float] = (20.0, 25.0),
) -> Dict:
    """
    Estimate signal-to-noise ratio in the individual alpha band relative
    to a flanking frequency band.

    Parameters
    ----------
    signal_uv : np.ndarray, shape (n_samples,) or (n_samples, n_channels)
    fs : float
    iaf_hz : float
        Individual alpha frequency peak.
    iaf_bw : float
        Half-bandwidth around IAF.
    noise_band_hz : tuple
        Reference noise band (should be free of alpha).

    Returns
    -------
    dict with keys:
        iaf_hz, alpha_band_hz, alpha_power_db_per_channel,
        noise_power_db_per_channel, snr_db_per_channel,
        snr_db_mean, passed
    """
    data = np.atleast_2d(signal_uv)
    if data.shape[0] < data.shape[1]:
        data = data.T

    n_samples, n_channels = data.shape
    nperseg = min(512, n_samples // 4)
    alpha_band = (iaf_hz - iaf_bw, iaf_hz + iaf_bw)

    alpha_powers, noise_powers, snrs = [], [], []
    for ch in range(n_channels):
        freqs, psd = sp_signal.welch(data[:, ch], fs=fs, nperseg=nperseg)
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        noise_mask  = (freqs >= noise_band_hz[0]) & (freqs <= noise_band_hz[1])

        ap = float(np.mean(psd[alpha_mask])) if np.any(alpha_mask) else 0.0
        np_ = float(np.mean(psd[noise_mask])) if np.any(noise_mask) else 0.0

        ap_db  = 10.0 * np.log10(ap + 1e-30)
        np_db  = 10.0 * np.log10(np_ + 1e-30)
        snr_db = ap_db - np_db

        alpha_powers.append(round(ap_db, 2))
        noise_powers.append(round(np_db, 2))
        snrs.append(round(snr_db, 2))

    snr_mean = float(np.mean(snrs))
    passed = snr_mean > 3.0   # at least 3 dB SNR

    return {
        "iaf_hz": iaf_hz,
        "alpha_band_hz": list(alpha_band),
        "noise_band_hz": list(noise_band_hz),
        "alpha_power_db_per_channel": alpha_powers,
        "noise_power_db_per_channel": noise_powers,
        "snr_db_per_channel": snrs,
        "snr_db_mean": round(snr_mean, 2),
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def run_all_validations(
    packets: Sequence[EEGPacket],
    signal_uv: np.ndarray,
    fs: float,
    iaf_hz: float,
    iaf_bw: float = 2.0,
    output_path: Optional[Path] = None,
    subject_id: str = "unknown",
    session_id: str = "unknown",
) -> Dict:
    """
    Run all four hardware-validation checks and save results to JSON.

    Parameters
    ----------
    packets : sequence of EEGPacket
    signal_uv : np.ndarray, shape (n_samples, n_channels)
        Raw EEG array (pre-split from the packet stream).
    fs : float
    iaf_hz : float
    iaf_bw : float
    output_path : Path, optional
        If provided, write JSON report to this path.
    subject_id, session_id : str

    Returns
    -------
    Full validation report dict; also written to ``output_path`` if given.
    """
    report: Dict = {
        "subject_id": subject_id,
        "session_id": session_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fs": fs,
        "iaf_hz": iaf_hz,
    }

    try:
        report["sampling_stability"] = measure_sampling_stability(packets, fs)
    except Exception as exc:
        report["sampling_stability"] = {"error": str(exc), "passed": False}

    try:
        report["packet_loss"] = detect_packet_loss(packets)
    except Exception as exc:
        report["packet_loss"] = {"error": str(exc), "passed": False}

    try:
        report["noise_floor"] = compute_noise_floor(signal_uv, fs)
    except Exception as exc:
        report["noise_floor"] = {"error": str(exc), "passed": False}

    try:
        report["alpha_snr"] = estimate_alpha_snr(signal_uv, fs, iaf_hz, iaf_bw)
    except Exception as exc:
        report["alpha_snr"] = {"error": str(exc), "passed": False}

    checks = [
        report.get("sampling_stability", {}).get("passed", False),
        report.get("packet_loss", {}).get("passed", False),
        report.get("noise_floor", {}).get("passed", False),
        report.get("alpha_snr", {}).get("passed", False),
    ]
    report["overall_passed"] = all(checks)
    report["checks_passed"] = sum(checks)
    report["checks_total"] = len(checks)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        logger.info("Hardware validation report saved: %s", output_path)

    status = "PASSED" if report["overall_passed"] else "FAILED"
    logger.info(
        "Hardware validation %s (%d/%d checks passed)",
        status, report["checks_passed"], report["checks_total"],
    )
    return report
