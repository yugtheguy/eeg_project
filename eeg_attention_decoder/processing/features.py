"""
processing/features.py
=======================
Feature extraction for the EEG spatial attention decoder.

All features operate on 1-second windows with 50 % overlap.
Windows are produced by ``extract_windows()`` which also stores
the center timestamp of each window.

Feature vector (per window, per hemisphere)
-------------------------------------------
1.  Log alpha power   — PSD integrated over [IAF − bw, IAF + bw] Hz
2.  Log beta power    — PSD integrated over [13, 20] Hz
3.  Relative alpha    — alpha power / broadband [4, 25] Hz power
4.  Lateralization index (LI) — scalar across both hemispheres:
        LI = (R − L) / (R + L)

Optional (disabled by default, see config):
5.  Inter-hemispheric coherence in the alpha band

Individual Alpha Frequency (IAF)
---------------------------------
``estimate_iaf()`` must be called on a 2-minute eyes-closed baseline
*before* any windowed features are computed.  The returned value is
stored in session metadata; this module never uses test data to derive
the IAF.

IMPORTANT: DO NOT compute IAF from test or experimental windows.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import welch, csd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IAF estimation (from eyes-closed baseline only)
# ---------------------------------------------------------------------------

def estimate_iaf(
    baseline_data: np.ndarray,
    fs: float,
    search_min_hz: float = 6.0,
    search_max_hz: float = 14.0,
    default_hz: float = 10.0,
    channel_weights: Optional[np.ndarray] = None,
) -> Dict:
    """
    Estimate Individual Alpha Frequency from an eyes-closed baseline.

    Parameters
    ----------
    baseline_data : np.ndarray, shape (n_samples, n_channels)
        Pre-filtered eyes-closed EEG (causal filter already applied).
        Must be at least 10 seconds long for a reliable PSD estimate.
    fs : float
        Sampling rate in Hz.
    search_min_hz, search_max_hz : float
        Frequency range to search for the alpha peak.
    default_hz : float
        Fallback IAF if no clear peak is found.
    channel_weights : np.ndarray, shape (n_channels,), optional
        Per-channel weights for the averaged PSD.
        Defaults to uniform weighting.

    Returns
    -------
    dict with keys:
        iaf_hz, peak_power_db, search_range_hz, used_default,
        psd_freqs, psd_mean_db (for plotting)
    """
    if baseline_data.ndim != 2:
        raise ValueError("baseline_data must be 2-D (n_samples, n_channels).")

    n_samples, n_channels = baseline_data.shape
    min_samples = int(10 * fs)
    if n_samples < min_samples:
        raise ValueError(
            f"Baseline must be ≥ 10 s; got {n_samples / fs:.1f} s."
        )

    nperseg = min(int(4 * fs), n_samples // 4)   # 4-second Welch segments
    nperseg = max(nperseg, 64)

    if channel_weights is None:
        channel_weights = np.ones(n_channels) / n_channels
    else:
        channel_weights = np.asarray(channel_weights, dtype=np.float64)
        channel_weights /= channel_weights.sum()

    # Average PSD across channels
    psd_sum = None
    for ch in range(n_channels):
        freqs, psd = welch(baseline_data[:, ch], fs=fs, nperseg=nperseg)
        if psd_sum is None:
            psd_sum = psd * channel_weights[ch]
        else:
            psd_sum += psd * channel_weights[ch]

    assert psd_sum is not None
    search_mask = (freqs >= search_min_hz) & (freqs <= search_max_hz)

    if not np.any(search_mask):
        logger.warning("No frequencies in IAF search range; using default %.1f Hz.", default_hz)
        return {
            "iaf_hz": default_hz,
            "peak_power_db": float("nan"),
            "search_range_hz": [search_min_hz, search_max_hz],
            "used_default": True,
            "psd_freqs": freqs.tolist(),
            "psd_mean_db": (10 * np.log10(psd_sum + 1e-30)).tolist(),
        }

    search_psd = psd_sum[search_mask]
    search_freqs = freqs[search_mask]
    peak_idx = int(np.argmax(search_psd))
    iaf = float(search_freqs[peak_idx])
    peak_power_db = float(10.0 * np.log10(search_psd[peak_idx] + 1e-30))

    # Validate: peak should be at least 3 dB above neighbours
    flat_threshold = 3.0   # dB
    used_default = False
    if search_psd.max() - search_psd.min() < 10 ** (flat_threshold / 10):
        logger.warning(
            "Alpha peak not prominent (< %.1f dB rise); using default IAF.", flat_threshold
        )
        iaf = default_hz
        used_default = True

    logger.info(
        "IAF estimated at %.2f Hz (peak power = %.1f dB, used_default=%s)",
        iaf, peak_power_db, used_default,
    )
    return {
        "iaf_hz": round(iaf, 3),
        "peak_power_db": round(peak_power_db, 2),
        "search_range_hz": [search_min_hz, search_max_hz],
        "used_default": used_default,
        "psd_freqs": freqs.tolist(),
        "psd_mean_db": (10 * np.log10(psd_sum + 1e-30)).tolist(),
    }


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def extract_windows(
    data: np.ndarray,
    timestamps: np.ndarray,
    window_size_samples: int,
    step_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slice a recording into overlapping analysis windows.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
    timestamps : np.ndarray, shape (n_samples,)
        Wall-clock or Arduino timestamps for each sample.
    window_size_samples : int
    step_samples : int
        ``window_size_samples * (1 - overlap)`` for 50 % overlap.

    Returns
    -------
    windows : np.ndarray, shape (n_windows, n_channels, window_size_samples)
    center_timestamps : np.ndarray, shape (n_windows,)
        Timestamp at the centre of each window.
    """
    n_samples = data.shape[0]

    if n_samples < window_size_samples:
        raise ValueError(
            f"Data length ({n_samples}) < window size ({window_size_samples})."
        )

    starts = np.arange(0, n_samples - window_size_samples + 1, step_samples)
    n_windows = len(starts)

    windows = np.empty(
        (n_windows, data.shape[1], window_size_samples), dtype=np.float64
    )
    center_timestamps = np.empty(n_windows, dtype=np.float64)

    for i, start in enumerate(starts):
        end = start + window_size_samples
        windows[i] = data[start:end].T          # (n_channels, window_size)
        mid = (start + end) // 2
        center_timestamps[i] = timestamps[mid]

    return windows, center_timestamps


# ---------------------------------------------------------------------------
# Band power
# ---------------------------------------------------------------------------

def band_power(
    window: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    nperseg: Optional[int] = None,
) -> float:
    """
    Compute total power in a frequency band via Welch's method.

    Parameters
    ----------
    window : np.ndarray, shape (n_samples,)
        Single-channel, single-window EEG segment.
    fs : float
    low_hz, high_hz : float
        Band edges in Hz.
    nperseg : int, optional
        Welch segment length; defaults to half the window.

    Returns
    -------
    power : float
        Integrated band power (µV²).
    """
    if nperseg is None:
        nperseg = max(len(window) // 2, 32)
    freqs, psd = welch(window, fs=fs, nperseg=min(nperseg, len(window)))
    band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band_mask):
        return 0.0
    df = freqs[1] - freqs[0]   # frequency resolution
    return float(np.sum(psd[band_mask]) * df)


def log_band_power(
    window: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> float:
    """Return log10 of band power (floor at 1e-30 to avoid log(0))."""
    bp = band_power(window, fs, low_hz, high_hz)
    return float(np.log10(bp + 1e-30))


# ---------------------------------------------------------------------------
# Lateralization index
# ---------------------------------------------------------------------------

def lateralization_index(left_power: float, right_power: float) -> float:
    """
    Compute the alpha lateralization index.

        LI = (R − L) / (R + L)

    Returns 0.0 if both powers are zero (undefined).

    Parameters
    ----------
    left_power : float
        Alpha power for left hemisphere channel (e.g. T7).
    right_power : float
        Alpha power for right hemisphere channel (e.g. T8).
    """
    denom = right_power + left_power
    if denom < 1e-30:
        return 0.0
    return float((right_power - left_power) / denom)


# ---------------------------------------------------------------------------
# Inter-hemispheric coherence (optional)
# ---------------------------------------------------------------------------

def alpha_coherence(
    left_window: np.ndarray,
    right_window: np.ndarray,
    fs: float,
    iaf_hz: float,
    iaf_bw: float = 2.0,
) -> float:
    """
    Compute mean squared coherence between T7 and T8 in the alpha band.

    Parameters
    ----------
    left_window, right_window : np.ndarray, shape (n_samples,)
    fs : float
    iaf_hz : float
    iaf_bw : float

    Returns
    -------
    coherence : float in [0, 1]
    """
    nperseg = max(len(left_window) // 2, 32)
    freqs, Pxy = csd(left_window, right_window, fs=fs, nperseg=nperseg)
    _, Pxx = welch(left_window, fs=fs, nperseg=nperseg)
    _, Pyy = welch(right_window, fs=fs, nperseg=nperseg)

    coh = np.abs(Pxy) ** 2 / (Pxx * Pyy + 1e-30)
    band_mask = (freqs >= iaf_hz - iaf_bw) & (freqs <= iaf_hz + iaf_bw)
    if not np.any(band_mask):
        return 0.0
    return float(np.mean(coh[band_mask]))


# ---------------------------------------------------------------------------
# Full feature extraction for one window
# ---------------------------------------------------------------------------

def extract_features_window(
    window: np.ndarray,
    fs: float,
    iaf_hz: float,
    iaf_bw: float,
    beta_low: float,
    beta_high: float,
    broadband_low: float,
    broadband_high: float,
    left_ch_idx: int = 0,
    right_ch_idx: int = 1,
    enable_coherence: bool = False,
) -> np.ndarray:
    """
    Extract the full feature vector from a single analysis window.

    Parameters
    ----------
    window : np.ndarray, shape (n_channels, n_window_samples)
    fs : float
    iaf_hz : float  — individual alpha frequency (from baseline, NOT test data)
    iaf_bw : float  — ± Hz around IAF
    beta_low, beta_high : float
    broadband_low, broadband_high : float
    left_ch_idx, right_ch_idx : int
        Indices of the T7 and T8 channels.
    enable_coherence : bool

    Returns
    -------
    features : np.ndarray, shape (n_features,)
        [log_alpha_left, log_alpha_right, log_beta_left, log_beta_right,
         rel_alpha_left, rel_alpha_right, LI, (coherence?)]
    """
    alpha_low = iaf_hz - iaf_bw
    alpha_high = iaf_hz + iaf_bw

    left  = window[left_ch_idx]
    right = window[right_ch_idx]

    # Log alpha power
    log_alpha_l = log_band_power(left,  fs, alpha_low, alpha_high)
    log_alpha_r = log_band_power(right, fs, alpha_low, alpha_high)

    # Log beta power
    log_beta_l = log_band_power(left,  fs, beta_low, beta_high)
    log_beta_r = log_band_power(right, fs, beta_low, beta_high)

    # Relative alpha: alpha / broadband
    bb_l = band_power(left,  fs, broadband_low, broadband_high)
    bb_r = band_power(right, fs, broadband_low, broadband_high)
    alpha_l = band_power(left,  fs, alpha_low, alpha_high)
    alpha_r = band_power(right, fs, alpha_low, alpha_high)
    rel_alpha_l = float(alpha_l / (bb_l + 1e-30))
    rel_alpha_r = float(alpha_r / (bb_r + 1e-30))

    # Lateralization index
    li = lateralization_index(alpha_l, alpha_r)

    features = [
        log_alpha_l, log_alpha_r,
        log_beta_l,  log_beta_r,
        rel_alpha_l, rel_alpha_r,
        li,
    ]

    if enable_coherence:
        coh = alpha_coherence(left, right, fs, iaf_hz, iaf_bw)
        features.append(coh)

    return np.array(features, dtype=np.float64)


# ---------------------------------------------------------------------------
# Batch feature extraction (offline use)
# ---------------------------------------------------------------------------

def extract_features_batch(
    windows: np.ndarray,
    fs: float,
    iaf_hz: float,
    cfg: Dict,
    left_ch_idx: int = 0,
    right_ch_idx: int = 1,
) -> np.ndarray:
    """
    Extract features for all windows in a recording.

    Parameters
    ----------
    windows : np.ndarray, shape (n_windows, n_channels, window_size)
    fs : float
    iaf_hz : float  — MUST come from the eyes-closed baseline
    cfg : dict      — full YAML config for feature parameters
    left_ch_idx, right_ch_idx : int

    Returns
    -------
    X : np.ndarray, shape (n_windows, n_features)
    """
    feat_cfg = cfg["features"]
    n_windows = windows.shape[0]
    feat_list: List[np.ndarray] = []

    for i in range(n_windows):
        fv = extract_features_window(
            window=windows[i],
            fs=fs,
            iaf_hz=iaf_hz,
            iaf_bw=feat_cfg["iaf_bandwidth"],
            beta_low=feat_cfg["beta_low"],
            beta_high=feat_cfg["beta_high"],
            broadband_low=feat_cfg["broadband_low"],
            broadband_high=feat_cfg["broadband_high"],
            left_ch_idx=left_ch_idx,
            right_ch_idx=right_ch_idx,
            enable_coherence=feat_cfg.get("enable_coherence", False),
        )
        feat_list.append(fv)

    return np.stack(feat_list, axis=0)    # (n_windows, n_features)
