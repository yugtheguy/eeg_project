"""
tests/test_filters.py
======================
Unit tests for the causal EEG filter bank.

Tests
-----
* Passband flatness within [5, 20] Hz (< 3 dB ripple).
* Stopband attenuation at DC (< –30 dB).
* Notch attenuation at 50 Hz (< –20 dB).
* Stateful filtering produces output with the same shape as input.
* ``filtfilt`` is NOT imported or called (causal-only guarantee).
* Filter state carries across consecutive calls (no discontinuity).
"""

from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
from scipy.signal import freqz

# Make the project importable from the tests directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from processing.filters import (
    FilterBank,
    build_bandpass_sos,
    build_filter_bank,
    build_notch_sos,
    frequency_response,
)

FS = 250.0          # Hz — matches Arduino hardware
N_CH = 2
PASSBAND = (5.0, 20.0)   # Hz — fully inside [2, 25] Hz bandpass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bank() -> FilterBank:
    cfg = {
        "hardware": {"sampling_rate": FS, "n_channels": N_CH},
        "processing": {
            "bandpass_low": 2.0,
            "bandpass_high": 25.0,
            "bandpass_order": 4,
            "notch_freq": 50.0,
            "notch_q": 30.0,
        },
    }
    return build_filter_bank(cfg, n_channels=N_CH)


def _pure_sine(freq_hz: float, duration_sec: float, fs: float, n_ch: int) -> np.ndarray:
    t = np.arange(int(duration_sec * fs)) / fs
    s = np.sin(2 * np.pi * freq_hz * t)
    return np.column_stack([s] * n_ch)


# ---------------------------------------------------------------------------
# Frequency-response tests
# ---------------------------------------------------------------------------

class TestFrequencyResponse:
    """Verify the combined bandpass + notch frequency response."""

    def setup_method(self):
        self.bank = _make_bank()
        self.freqs, self.mag_db = frequency_response(self.bank, FS)

    def test_passband_flatness(self):
        """Passband [5, 20] Hz should have attenuation < 3 dB."""
        mask = (self.freqs >= PASSBAND[0]) & (self.freqs <= PASSBAND[1])
        passband_db = self.mag_db[mask]
        # All values should be above –3 dB
        assert np.all(passband_db > -3.0), (
            f"Passband attenuation exceeded 3 dB; min={passband_db.min():.1f} dB."
        )

    def test_dc_stopband(self):
        """DC (0.5 Hz) should be strongly attenuated (< –30 dB)."""
        mask = self.freqs < 0.8
        if not np.any(mask):
            pytest.skip("Frequency resolution too low for DC test.")
        dc_db = self.mag_db[mask].max()
        assert dc_db < -30.0, (
            f"DC attenuation insufficient: {dc_db:.1f} dB (expected < –30 dB)."
        )

    def test_notch_attenuation(self):
        """50 Hz notch should attenuate by at least 20 dB."""
        mask = (self.freqs >= 49.0) & (self.freqs <= 51.0)
        if not np.any(mask):
            pytest.skip("Frequency bins do not cover 50 Hz range.")
        notch_db = self.mag_db[mask].min()
        assert notch_db < -20.0, (
            f"Notch attenuation at 50 Hz insufficient: {notch_db:.1f} dB."
        )

    def test_highfreq_stopband(self):
        """Frequencies > 30 Hz (above passband) should be attenuated."""
        mask = self.freqs > 30.0
        high_db = self.mag_db[mask]
        assert np.all(high_db < -6.0), (
            "High-frequency stopband attenuation insufficient."
        )


# ---------------------------------------------------------------------------
# Causal-only guarantee
# ---------------------------------------------------------------------------

class TestCausalOnly:
    """Verify that filtfilt is never called in the filters module."""

    def test_filtfilt_not_imported_in_filters(self):
        """
        ``scipy.signal.filtfilt`` must NOT be called in processing/filters.py.
        This enforces the causal-filtering-only policy.
        """
        import processing.filters as filt_module
        source = inspect.getsource(filt_module)
        assert "filtfilt" not in source, (
            "VIOLATION: 'filtfilt' appears in processing/filters.py. "
            "All filtering must be causal (sosfilt only)."
        )

    def test_sosfiltz_not_called_in_filters(self):
        """sosfiltfilt is also zero-phase — must not appear."""
        import processing.filters as filt_module
        source = inspect.getsource(filt_module)
        assert "sosfiltfilt" not in source.replace("from scipy.signal import", "###"), (
            "VIOLATION: 'sosfiltfilt' (zero-phase) found in processing/filters.py."
        )


# ---------------------------------------------------------------------------
# Shape and state tests
# ---------------------------------------------------------------------------

class TestFilterBankBehaviour:

    def setup_method(self):
        self.bank = _make_bank()

    def test_output_shape_matches_input(self):
        data = np.random.randn(500, N_CH)
        out = self.bank.apply(data)
        assert out.shape == data.shape

    def test_single_sample_api(self):
        sample = np.zeros(N_CH)
        out = self.bank.apply_single(sample)
        assert out.shape == (N_CH,)

    def test_filter_state_continuity(self):
        """
        Processing two consecutive chunks must yield the same result as
        processing the concatenated chunk — verifying state is carried.
        """
        n_total = 1000
        data = _pure_sine(10.0, n_total / FS, FS, N_CH)

        # Fresh bank for reference (full chunk)
        bank_ref = _make_bank()
        out_full = bank_ref.apply(data)

        # Split into two halves
        half = n_total // 2
        bank_split = _make_bank()
        out_first  = bank_split.apply(data[:half])
        out_second = bank_split.apply(data[half:])
        out_split = np.vstack([out_first, out_second])

        # After initial transient (first 100 samples), outputs should match
        np.testing.assert_allclose(
            out_full[100:], out_split[100:], atol=1e-10,
            err_msg="Filter state discontinuity detected between chunks.",
        )

    def test_reset_clears_state(self):
        """After reset, filter output should match a freshly built bank."""
        data = np.random.randn(500, N_CH)
        bank_a = _make_bank()
        bank_a.apply(data)           # warm up

        bank_b = _make_bank()        # fresh
        bank_a.reset()

        out_a = bank_a.apply(data)
        out_b = bank_b.apply(data)
        np.testing.assert_allclose(out_a, out_b, atol=1e-12)

    def test_wrong_channel_count_raises(self):
        data = np.zeros((100, N_CH + 1))   # wrong number of channels
        with pytest.raises(ValueError, match="channels"):
            self.bank.apply(data)

    def test_wrong_ndim_raises(self):
        data = np.zeros(100)   # 1-D — should be 2-D
        with pytest.raises(ValueError, match="2-D"):
            self.bank.apply(data)
