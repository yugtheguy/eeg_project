"""
processing/filters.py
======================
Causal-only digital filters for EEG preprocessing.

DESIGN CONSTRAINTS (publication-grade)
---------------------------------------
* ALL filtering is CAUSAL (``sosfilt`` with a persistent ``zi`` state).
* Zero-phase / forward-backward filtering is EXPLICITLY FORBIDDEN — it
  introduces non-causal information and invalidates temporal
  alignment between offline and realtime pipelines.
* The same ``FilterBank`` instance is reused across offline windows to
  maintain continuous filter state, preventing transient artefacts at
  window boundaries.
* For offline batch processing initialise a fresh ``FilterBank`` per
  recording segment; never share state across sessions.

Public API
----------
FilterBank           – holds sos coefficients + zi state, applies filters.
build_filter_bank()  – construct from config dict.
reset_filter_state() – zero the zi (use at start of each new recording).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi

logger = logging.getLogger(__name__)

# Guard against accidental zero-phase imports in this module
_FILTFILT_FORBIDDEN = True


# ---------------------------------------------------------------------------
# Filter-bank dataclass
# ---------------------------------------------------------------------------

@dataclass
class FilterBank:
    """
    Stateful causal filter bank for a fixed-channel EEG stream.

    Attributes
    ----------
    bp_sos : np.ndarray
        Butterworth bandpass second-order sections.
    notch_sos : np.ndarray
        IIR notch second-order sections.
    n_channels : int
        Number of EEG channels (must match input data).
    bp_zi : np.ndarray
        Bandpass filter state, shape (n_sos_sections, 2, n_channels).
    notch_zi : np.ndarray
        Notch filter state, shape (n_sos_sections, 2, n_channels).
    """

    bp_sos: np.ndarray
    notch_sos: np.ndarray
    n_channels: int
    bp_zi: np.ndarray = field(init=False)
    notch_zi: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._init_zi()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset filter state to zero-input-response initial conditions."""
        self._init_zi()

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass → notch in sequence (causal).

        Parameters
        ----------
        data : np.ndarray, shape (n_samples, n_channels)
            Raw EEG segment in any amplitude unit.

        Returns
        -------
        filtered : np.ndarray, shape (n_samples, n_channels)
            Filtered EEG, same unit as input.
        """
        self._validate_input(data)
        out = np.empty_like(data)

        for ch in range(self.n_channels):
            x = data[:, ch]
            # Bandpass (causal)
            y_bp, self.bp_zi[:, :, ch] = sosfilt(
                self.bp_sos, x, zi=self.bp_zi[:, :, ch]
            )
            # Notch (causal)
            y_notch, self.notch_zi[:, :, ch] = sosfilt(
                self.notch_sos, y_bp, zi=self.notch_zi[:, :, ch]
            )
            out[:, ch] = y_notch

        return out

    def apply_single(self, sample: np.ndarray) -> np.ndarray:
        """
        Apply filters to a single sample (n_channels,) in streaming mode.

        This is an alias for ``apply`` with a (1, n_channels) input,
        kept as a convenience for realtime callers.
        """
        return self.apply(sample.reshape(1, -1)).ravel()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_zi(self) -> None:
        """Initialise zi for all channels and both filters."""
        bp_zi_1ch = sosfilt_zi(self.bp_sos)       # (n_sections, 2)
        n_bp_sec = bp_zi_1ch.shape[0]
        self.bp_zi = np.zeros((n_bp_sec, 2, self.n_channels), dtype=np.float64)
        for ch in range(self.n_channels):
            self.bp_zi[:, :, ch] = bp_zi_1ch

        notch_zi_1ch = sosfilt_zi(self.notch_sos)
        n_notch_sec = notch_zi_1ch.shape[0]
        self.notch_zi = np.zeros((n_notch_sec, 2, self.n_channels), dtype=np.float64)
        for ch in range(self.n_channels):
            self.notch_zi[:, :, ch] = notch_zi_1ch

    def _validate_input(self, data: np.ndarray) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected 2-D input, got shape {data.shape}")
        if data.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels, got {data.shape[1]}"
            )


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_bandpass_sos(
    low_hz: float,
    high_hz: float,
    order: int,
    fs: float,
) -> np.ndarray:
    """
    Design a Butterworth bandpass filter.

    Parameters
    ----------
    low_hz, high_hz : float
        Passband edges in Hz.
    order : int
        Filter order (applied once — do NOT double for forward-backward).
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sos : np.ndarray, shape (n_sections, 6)
    """
    nyq = fs / 2.0
    if not (0 < low_hz < high_hz < nyq):
        raise ValueError(
            f"Invalid bandpass [{low_hz}, {high_hz}] Hz for fs={fs} Hz "
            f"(Nyquist={nyq} Hz)."
        )
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="bandpass", output="sos")
    logger.debug("Bandpass SOS: order=%d, [%.1f, %.1f] Hz", order, low_hz, high_hz)
    return sos


def build_notch_sos(
    notch_hz: float,
    q_factor: float,
    fs: float,
) -> np.ndarray:
    """
    Design an IIR notch filter.

    Parameters
    ----------
    notch_hz : float
        Centre frequency to attenuate (Hz).
    q_factor : float
        Quality factor; higher Q → narrower notch.
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    sos : np.ndarray, shape (1, 6) — single biquad section.
    """
    nyq = fs / 2.0
    if notch_hz >= nyq:
        raise ValueError(
            f"Notch frequency {notch_hz} Hz exceeds Nyquist {nyq} Hz."
        )
    b, a = iirnotch(notch_hz / nyq, q_factor)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    logger.debug("Notch SOS: %.1f Hz, Q=%.1f", notch_hz, q_factor)
    return sos


def build_filter_bank(cfg: Dict, n_channels: int) -> FilterBank:
    """
    Construct a ``FilterBank`` from the ``processing`` section of the
    YAML config dict.

    Parameters
    ----------
    cfg : dict
        Must contain keys: ``bandpass_low``, ``bandpass_high``,
        ``bandpass_order``, ``notch_freq``, ``notch_q``, ``notch_order``
        (notch_order is ignored — IIR notch is always a single biquad).
        Also requires a parent key ``hardware.sampling_rate``.
    n_channels : int

    Returns
    -------
    FilterBank
    """
    proc = cfg["processing"]
    fs: float = cfg["hardware"]["sampling_rate"]

    bp_sos = build_bandpass_sos(
        low_hz=proc["bandpass_low"],
        high_hz=proc["bandpass_high"],
        order=proc["bandpass_order"],
        fs=fs,
    )
    notch_sos = build_notch_sos(
        notch_hz=proc["notch_freq"],
        q_factor=proc.get("notch_q", 30.0),
        fs=fs,
    )
    return FilterBank(bp_sos=bp_sos, notch_sos=notch_sos, n_channels=n_channels)


# ---------------------------------------------------------------------------
# Frequency-response inspector (for tests and reporting only)
# ---------------------------------------------------------------------------

def frequency_response(
    filter_bank: FilterBank,
    fs: float,
    n_points: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the combined frequency response of bandpass + notch.

    Returns
    -------
    freqs : np.ndarray, shape (n_points // 2 + 1,)
    magnitude_db : np.ndarray, shape (n_points // 2 + 1,)
    """
    from scipy.signal import sosfreqz
    n_half = n_points // 2 + 1

    _, h_bp = sosfreqz(filter_bank.bp_sos, worN=n_half, fs=fs)
    _, h_notch = sosfreqz(filter_bank.notch_sos, worN=n_half, fs=fs)
    freqs = np.linspace(0, fs / 2, n_half)
    h_combined = h_bp * h_notch
    magnitude_db = 20.0 * np.log10(np.abs(h_combined) + 1e-30)
    return freqs, magnitude_db
