"""
tests/test_features.py
=======================
Unit tests for EEG feature extraction.

Tests
-----
* Band-power estimator correctly integrates known synthetic PSD.
* Log power returns finite values (no log(0) crash).
* Lateralization index formula: LI = (R−L)/(R+L).
* LI is 0 when powers are equal, +1 when left=0, −1 when right=0.
* extract_features_window returns the correct feature-vector length.
* IAF estimation detects peaks within the search band.
* extract_windows produces correct window shapes and center timestamps.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.signal import chirp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.features import (
    alpha_coherence,
    band_power,
    estimate_iaf,
    extract_features_window,
    extract_windows,
    lateralization_index,
    log_band_power,
)

FS = 250.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pure_sine(freq_hz: float, duration_sec: float = 4.0, amplitude: float = 10.0) -> np.ndarray:
    """Return a pure sine waveform (1-D, float64)."""
    t = np.arange(int(duration_sec * FS)) / FS
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _two_channel_window(left_freq: float, right_freq: float, win_sec: float = 1.0) -> np.ndarray:
    """Return (n_channels=2, n_samples) window with pure tones."""
    n = int(win_sec * FS)
    t = np.arange(n) / FS
    left  = np.sin(2 * np.pi * left_freq  * t) * 10.0
    right = np.sin(2 * np.pi * right_freq * t) * 10.0
    return np.stack([left, right], axis=0)   # (2, n)


# ---------------------------------------------------------------------------
# Band power
# ---------------------------------------------------------------------------

class TestBandPower:

    def test_pure_sine_in_band(self):
        """Power of a 10 Hz sine in [8, 12] Hz band should be non-trivial."""
        sig = _pure_sine(10.0, duration_sec=4.0)
        p = band_power(sig, FS, 8.0, 12.0)
        assert p > 0.1, f"Expected detectable alpha power; got {p:.4f}"

    def test_pure_sine_out_of_band(self):
        """Power of a 30 Hz sine in [8, 12] Hz band should be near zero."""
        sig = _pure_sine(30.0, duration_sec=4.0)
        p = band_power(sig, FS, 8.0, 12.0)
        assert p < 0.01, f"Expected near-zero power; got {p:.6f}"

    def test_log_power_finite(self):
        """log_band_power must never return NaN or ±inf."""
        sig = np.zeros(int(4 * FS))   # all zeros — worst case
        lp = log_band_power(sig, FS, 8.0, 12.0)
        assert np.isfinite(lp), "log_band_power returned non-finite value on zero signal."

    def test_log_power_positive_signal(self):
        """Log power of a strong sine > log power of near-zero."""
        sig_strong = _pure_sine(10.0, amplitude=100.0, duration_sec=4.0)
        sig_weak   = _pure_sine(10.0, amplitude=0.01, duration_sec=4.0)
        assert log_band_power(sig_strong, FS, 8.0, 12.0) > \
               log_band_power(sig_weak,   FS, 8.0, 12.0)


# ---------------------------------------------------------------------------
# Lateralization index
# ---------------------------------------------------------------------------

class TestLateralizationIndex:

    def test_equal_power_is_zero(self):
        assert lateralization_index(5.0, 5.0) == pytest.approx(0.0)

    def test_right_dominant(self):
        """LI > 0 when right hemisphere has more alpha."""
        li = lateralization_index(left_power=2.0, right_power=8.0)
        assert li == pytest.approx((8.0 - 2.0) / (8.0 + 2.0))
        assert li > 0

    def test_left_dominant(self):
        """LI < 0 when left hemisphere has more alpha."""
        li = lateralization_index(left_power=8.0, right_power=2.0)
        assert li < 0

    def test_zero_left(self):
        """LI approaches +1 when left power is zero."""
        li = lateralization_index(0.0, 10.0)
        assert li == pytest.approx(1.0)

    def test_zero_right(self):
        """LI approaches −1 when right power is zero."""
        li = lateralization_index(10.0, 0.0)
        assert li == pytest.approx(-1.0)

    def test_both_zero_returns_zero(self):
        """Undefined case (both zero) must return 0.0, not crash."""
        li = lateralization_index(0.0, 0.0)
        assert li == 0.0

    def test_range_bounded(self):
        """LI must always be in [−1, +1]."""
        rng = np.random.default_rng(42)
        powers = rng.exponential(scale=5.0, size=(200, 2))
        for left, right in powers:
            li = lateralization_index(left, right)
            assert -1.0 <= li <= 1.0, f"LI={li} out of range for L={left}, R={right}"


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

class TestExtractFeaturesWindow:

    def test_feature_vector_length_without_coherence(self):
        """Default feature vector has 7 elements."""
        win = _two_channel_window(10.0, 10.0)
        fv = extract_features_window(
            window=win, fs=FS, iaf_hz=10.0, iaf_bw=2.0,
            beta_low=13.0, beta_high=20.0,
            broadband_low=4.0, broadband_high=25.0,
            enable_coherence=False,
        )
        assert fv.shape == (7,), f"Expected 7 features, got {fv.shape[0]}"

    def test_feature_vector_length_with_coherence(self):
        """With coherence enabled, vector has 8 elements."""
        win = _two_channel_window(10.0, 10.0)
        fv = extract_features_window(
            window=win, fs=FS, iaf_hz=10.0, iaf_bw=2.0,
            beta_low=13.0, beta_high=20.0,
            broadband_low=4.0, broadband_high=25.0,
            enable_coherence=True,
        )
        assert fv.shape == (8,), f"Expected 8 features with coherence, got {fv.shape[0]}"

    def test_all_features_finite(self):
        """All feature values must be finite."""
        win = _two_channel_window(10.0, 15.0)
        fv = extract_features_window(
            window=win, fs=FS, iaf_hz=10.0, iaf_bw=2.0,
            beta_low=13.0, beta_high=20.0,
            broadband_low=4.0, broadband_high=25.0,
        )
        assert np.all(np.isfinite(fv)), f"Non-finite feature(s): {fv}"

    def test_li_sign_in_feature_vector(self):
        """Feature vector index 6 (LI) should be positive when right >> left."""
        # Right channel has strong alpha, left has none
        n = int(1.0 * FS)
        t = np.arange(n) / FS
        win = np.stack([
            0.001 * np.random.randn(n),        # left — no alpha
            10.0 * np.sin(2 * np.pi * 10.0 * t),  # right — strong alpha
        ], axis=0)   # (2, n)
        fv = extract_features_window(
            window=win, fs=FS, iaf_hz=10.0, iaf_bw=2.0,
            beta_low=13.0, beta_high=20.0,
            broadband_low=4.0, broadband_high=25.0,
        )
        assert fv[6] > 0, f"Expected positive LI (right dominant), got {fv[6]:.4f}"


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

class TestExtractWindows:

    def test_window_shape(self):
        n_samples = 1500
        n_channels = 2
        window_samples = 250    # 1 s at 250 Hz
        step_samples   = 125    # 50 % overlap
        data = np.random.randn(n_samples, n_channels)
        timestamps = np.arange(n_samples) / FS

        windows, centers = extract_windows(data, timestamps, window_samples, step_samples)
        expected_n = (n_samples - window_samples) // step_samples + 1
        assert windows.shape == (expected_n, n_channels, window_samples), (
            f"Unexpected window shape: {windows.shape}"
        )
        assert centers.shape == (expected_n,)

    def test_center_timestamps_in_range(self):
        n_samples = 500
        n_channels = 2
        window_samples = 250
        step_samples   = 125
        data = np.zeros((n_samples, n_channels))
        timestamps = np.arange(n_samples) / FS

        _, centers = extract_windows(data, timestamps, window_samples, step_samples)
        half_win = (window_samples / 2) / FS
        assert np.all(centers >= timestamps[0] + half_win - 1e-9)
        assert np.all(centers <= timestamps[-1] + 1e-9)

    def test_too_short_data_raises(self):
        data = np.zeros((100, 2))
        ts   = np.arange(100) / FS
        with pytest.raises(ValueError, match="window size"):
            extract_windows(data, ts, window_size_samples=250, step_samples=125)


# ---------------------------------------------------------------------------
# IAF estimation
# ---------------------------------------------------------------------------

class TestEstimateIAF:

    def test_detects_synthetic_peak(self):
        """IAF estimator should find a prominent 10 Hz peak."""
        duration = 120.0
        t = np.arange(int(duration * FS)) / FS
        signal_ch = 20.0 * np.sin(2 * np.pi * 10.0 * t) + \
                    np.random.default_rng(7).normal(scale=1.0, size=len(t))
        data = np.column_stack([signal_ch, signal_ch])  # 2-channel

        result = estimate_iaf(
            data, FS, search_min_hz=6.0, search_max_hz=14.0
        )
        assert not result["used_default"], "IAF estimator should find the peak, not use default."
        assert abs(result["iaf_hz"] - 10.0) < 1.0, (
            f"IAF estimation error > 1 Hz: got {result['iaf_hz']}"
        )

    def test_flat_spectrum_uses_default(self):
        """White noise should yield a flat spectrum → use default IAF."""
        duration = 30.0   # 30 s of white noise
        rng = np.random.default_rng(0)
        data = rng.normal(scale=1.0, size=(int(duration * FS), 2))
        result = estimate_iaf(data, FS, default_hz=10.0)
        assert result["used_default"] is True

    def test_raises_on_short_baseline(self):
        data = np.zeros((int(5 * FS), 2))  # only 5 s — need ≥ 10 s
        with pytest.raises(ValueError, match="10 s"):
            estimate_iaf(data, FS)
