"""
Signal Filtering Module.

Implements digital filters for EEG signal preprocessing:
- Notch filter for power line interference (50/60 Hz)
- Bandpass filter (1-40 Hz) for valid EEG frequency range
- Alpha band extraction (8-12 Hz)
- Beta band extraction (13-30 Hz) for artifact detection

All filters use zero-phase filtering (filtfilt) to avoid phase distortion.
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt
import logging
from typing import Tuple, Optional

from config import SignalConfig, get_config


# Configure logging
logger = logging.getLogger(__name__)


class SignalFilter:
    """
    Digital filter implementation for EEG signal processing.
    
    Uses scipy.signal for Butterworth filters and zero-phase filtering.
    """
    
    def __init__(self, signal_config: Optional[SignalConfig] = None):
        """
        Initialize filter with configuration.
        
        Args:
            signal_config: Signal configuration object. If None, uses global config.
        """
        if signal_config is None:
            signal_config = get_config().signal
        
        self.config = signal_config
        self.fs = signal_config.sampling_rate
        
        # Pre-compute filter coefficients for efficiency
        self._init_filters()
        
        logger.info(f"SignalFilter initialized (Fs={self.fs} Hz)")
    
    def _init_filters(self) -> None:
        """Pre-compute filter coefficients for all required filters."""
        # Notch filter for power line interference
        self.notch_b, self.notch_a = iirnotch(
            w0=self.config.notch_freq,
            Q=self.config.notch_quality,
            fs=self.fs
        )
        logger.debug(f"Notch filter: {self.config.notch_freq} Hz, Q={self.config.notch_quality}")
        
        # Bandpass filter (1-40 Hz) for general EEG
        self.bandpass_sos = butter(
            N=self.config.filter_order,
            Wn=[self.config.bandpass_low, self.config.bandpass_high],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        logger.debug(f"Bandpass filter: {self.config.bandpass_low}-{self.config.bandpass_high} Hz")
        
        # Alpha band filter (8-12 Hz)
        self.alpha_sos = butter(
            N=self.config.filter_order,
            Wn=[self.config.alpha_low, self.config.alpha_high],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        logger.debug(f"Alpha band filter: {self.config.alpha_low}-{self.config.alpha_high} Hz")
        
        # Beta band filter (13-30 Hz) for artifact detection
        self.beta_sos = butter(
            N=self.config.filter_order,
            Wn=[self.config.beta_low, self.config.beta_high],
            btype='bandpass',
            fs=self.fs,
            output='sos'
        )
        logger.debug(f"Beta band filter: {self.config.beta_low}-{self.config.beta_high} Hz")
    
    def apply_notch_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply notch filter to remove power line interference.
        
        Args:
            data: Input signal array
        
        Returns:
            np.ndarray: Filtered signal
        """
        if len(data) < 3 * self.config.filter_order:
            logger.warning(f"Signal too short for notch filter ({len(data)} samples)")
            return data
        
        try:
            filtered = filtfilt(self.notch_b, self.notch_a, data)
            return filtered
        except Exception as e:
            logger.error(f"Notch filter error: {e}")
            return data
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filter (1-40 Hz) for general EEG preprocessing.
        
        Args:
            data: Input signal array
        
        Returns:
            np.ndarray: Filtered signal
        """
        if len(data) < 3 * self.config.filter_order:
            logger.warning(f"Signal too short for bandpass filter ({len(data)} samples)")
            return data
        
        try:
            filtered = sosfiltfilt(self.bandpass_sos, data)
            return filtered
        except Exception as e:
            logger.error(f"Bandpass filter error: {e}")
            return data
    
    def extract_alpha_band(self, data: np.ndarray) -> np.ndarray:
        """
        Extract alpha band (8-12 Hz) from signal.
        
        Args:
            data: Input signal array
        
        Returns:
            np.ndarray: Alpha band signal
        """
        if len(data) < 3 * self.config.filter_order:
            logger.warning(f"Signal too short for alpha filter ({len(data)} samples)")
            return np.zeros_like(data)
        
        try:
            alpha_signal = sosfiltfilt(self.alpha_sos, data)
            return alpha_signal
        except Exception as e:
            logger.error(f"Alpha filter error: {e}")
            return np.zeros_like(data)
    
    def extract_beta_band(self, data: np.ndarray) -> np.ndarray:
        """
        Extract beta band (13-30 Hz) from signal.
        Used primarily for artifact detection.
        
        Args:
            data: Input signal array
        
        Returns:
            np.ndarray: Beta band signal
        """
        if len(data) < 3 * self.config.filter_order:
            logger.warning(f"Signal too short for beta filter ({len(data)} samples)")
            return np.zeros_like(data)
        
        try:
            beta_signal = sosfiltfilt(self.beta_sos, data)
            return beta_signal
        except Exception as e:
            logger.error(f"Beta filter error: {e}")
            return np.zeros_like(data)
    
    def preprocess_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Full preprocessing pipeline: notch filter + bandpass filter.
        
        Args:
            data: Raw input signal
        
        Returns:
            np.ndarray: Preprocessed signal
        """
        if len(data) == 0:
            return data
        
        # Remove DC offset
        data_centered = data - np.mean(data)
        
        # Apply notch filter to remove power line interference
        notched = self.apply_notch_filter(data_centered)
        
        # Apply bandpass filter (1-40 Hz)
        filtered = self.apply_bandpass_filter(notched)
        
        return filtered
    
    def compute_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Compute signal envelope using Hilbert transform.
        
        Args:
            data: Input signal (typically band-limited)
        
        Returns:
            np.ndarray: Signal envelope (instantaneous amplitude)
        """
        try:
            analytic_signal = signal.hilbert(data)
            envelope = np.abs(analytic_signal)
            return envelope
        except Exception as e:
            logger.error(f"Hilbert transform error: {e}")
            return np.abs(data)
    
    def compute_power_spectrum(
        self,
        data: np.ndarray,
        nfft: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density using Welch's method.
        
        Args:
            data: Input signal
            nfft: FFT length. If None, uses signal length
        
        Returns:
            tuple: (frequencies, power_spectral_density)
        """
        if len(data) < self.fs:
            logger.warning("Signal too short for reliable PSD estimation")
        
        try:
            freqs, psd = signal.welch(
                data,
                fs=self.fs,
                nperseg=min(len(data), int(self.fs)),
                nfft=nfft,
                scaling='density'
            )
            return freqs, psd
        except Exception as e:
            logger.error(f"PSD computation error: {e}")
            return np.array([]), np.array([])
    
    def compute_band_power(
        self,
        data: np.ndarray,
        freq_low: float,
        freq_high: float
    ) -> float:
        """
        Compute power in a specific frequency band.
        
        Args:
            data: Input signal
            freq_low: Lower frequency bound (Hz)
            freq_high: Upper frequency bound (Hz)
        
        Returns:
            float: Band power
        """
        if len(data) < 2 * self.config.filter_order:
            return 0.0
        
        try:
            # Compute PSD
            freqs, psd = self.compute_power_spectrum(data)
            
            if len(freqs) == 0:
                return 0.0
            
            # Find frequency indices
            freq_mask = (freqs >= freq_low) & (freqs <= freq_high)
            
            # Integrate power in band
            band_power = np.trapezoid(psd[freq_mask], freqs[freq_mask])
            
            return band_power
            
        except Exception as e:
            logger.error(f"Band power computation error: {e}")
            return 0.0
    
    def compute_alpha_power(self, data: np.ndarray) -> float:
        """
        Compute power in alpha band (8-12 Hz).
        
        Args:
            data: Preprocessed EEG signal
        
        Returns:
            float: Alpha band power
        """
        return self.compute_band_power(
            data,
            self.config.alpha_low,
            self.config.alpha_high
        )
    
    def compute_beta_power(self, data: np.ndarray) -> float:
        """
        Compute power in beta band (13-30 Hz).
        
        Args:
            data: Preprocessed EEG signal
        
        Returns:
            float: Beta band power
        """
        return self.compute_band_power(
            data,
            self.config.beta_low,
            self.config.beta_high
        )
    
    def smooth_signal(self, data: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Smooth signal using moving average.
        
        Args:
            data: Input signal
            window_size: Size of moving average window
        
        Returns:
            np.ndarray: Smoothed signal
        """
        if len(data) < window_size:
            return data
        
        try:
            kernel = np.ones(window_size) / window_size
            smoothed = np.convolve(data, kernel, mode='same')
            return smoothed
        except Exception as e:
            logger.error(f"Smoothing error: {e}")
            return data
    
    def remove_baseline_drift(
        self,
        data: np.ndarray,
        cutoff_freq: float = 0.5
    ) -> np.ndarray:
        """
        Remove baseline drift using high-pass filter.
        
        Args:
            data: Input signal
            cutoff_freq: Cutoff frequency (Hz) for high-pass filter
        
        Returns:
            np.ndarray: Signal with baseline drift removed
        """
        if len(data) < 3 * self.config.filter_order:
            return data - np.mean(data)
        
        try:
            # Design high-pass filter
            sos = butter(
                N=2,
                Wn=cutoff_freq,
                btype='highpass',
                fs=self.fs,
                output='sos'
            )
            
            filtered = sosfiltfilt(sos, data)
            return filtered
            
        except Exception as e:
            logger.error(f"Baseline removal error: {e}")
            return data - np.mean(data)
    
    def detect_line_noise(self, data: np.ndarray) -> float:
        """
        Detect power line noise (50 Hz) level in signal.
        
        Args:
            data: Input signal
        
        Returns:
            float: Power at line frequency
        """
        return self.compute_band_power(
            data,
            self.config.notch_freq - 2,
            self.config.notch_freq + 2
        )


def create_filter(signal_config: Optional[SignalConfig] = None) -> SignalFilter:
    """
    Factory function to create a SignalFilter instance.
    
    Args:
        signal_config: Signal configuration. If None, uses global config.
    
    Returns:
        SignalFilter: Configured filter instance
    """
    return SignalFilter(signal_config)
