"""
Feature Extraction Module.

Extracts relevant features from EEG signals for attention detection:
- Alpha band power (left and right hemispheres)
- Alpha envelope (instantaneous amplitude)
- Hemispheric lateralization features
- Time-domain features (variance, peak amplitude)
- Frequency-domain features (spectral edge, median frequency)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis
import logging
from typing import Tuple, Dict, Optional

from config import SignalConfig, get_config
from filters import SignalFilter


# Configure logging
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from EEG signals for attention direction classification.
    
    Focuses on alpha band asymmetry as primary indicator of spatial attention.
    """
    
    def __init__(self, signal_config: Optional[SignalConfig] = None):
        """
        Initialize feature extractor.
        
        Args:
            signal_config: Signal configuration. If None, uses global config.
        """
        if signal_config is None:
            signal_config = get_config().signal
        
        self.config = signal_config
        self.filter = SignalFilter(signal_config)
        
        logger.info("FeatureExtractor initialized")
    
    def extract_alpha_power(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> Tuple[float, float]:
        """
        Extract alpha band power from both hemispheres.
        
        Args:
            left_signal: Preprocessed EEG from left hemisphere
            right_signal: Preprocessed EEG from right hemisphere
        
        Returns:
            tuple: (left_alpha_power, right_alpha_power)
        """
        try:
            # Extract alpha band
            left_alpha = self.filter.extract_alpha_band(left_signal)
            right_alpha = self.filter.extract_alpha_band(right_signal)
            
            # Compute power (mean squared amplitude)
            left_power = np.mean(left_alpha ** 2)
            right_power = np.mean(right_alpha ** 2)
            
            return left_power, right_power
            
        except Exception as e:
            logger.error(f"Alpha power extraction error: {e}")
            return 0.0, 0.0
    
    def extract_alpha_envelope(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract alpha band envelope using Hilbert transform.
        
        Args:
            left_signal: Preprocessed EEG from left hemisphere
            right_signal: Preprocessed EEG from right hemisphere
        
        Returns:
            tuple: (left_alpha_envelope, right_alpha_envelope)
        """
        try:
            # Extract alpha band
            left_alpha = self.filter.extract_alpha_band(left_signal)
            right_alpha = self.filter.extract_alpha_band(right_signal)
            
            # Compute envelope
            left_envelope = self.filter.compute_envelope(left_alpha)
            right_envelope = self.filter.compute_envelope(right_alpha)
            
            return left_envelope, right_envelope
            
        except Exception as e:
            logger.error(f"Alpha envelope extraction error: {e}")
            return np.zeros_like(left_signal), np.zeros_like(right_signal)
    
    def compute_spectral_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute frequency-domain features.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            dict: Spectral features including band powers and spectral metrics
        """
        try:
            # Compute power spectrum
            freqs, psd = self.filter.compute_power_spectrum(signal_data)
            
            if len(freqs) == 0:
                return self._empty_spectral_features()
            
            # Compute band powers
            delta_power = self.filter.compute_band_power(signal_data, 0.5, 4.0)
            theta_power = self.filter.compute_band_power(signal_data, 4.0, 8.0)
            alpha_power = self.filter.compute_alpha_power(signal_data)
            beta_power = self.filter.compute_beta_power(signal_data)
            gamma_power = self.filter.compute_band_power(signal_data, 30.0, 40.0)
            
            # Total power
            total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
            
            # Relative band powers
            rel_alpha = alpha_power / total_power if total_power > 0 else 0.0
            rel_beta = beta_power / total_power if total_power > 0 else 0.0
            
            # Spectral edge frequency (95% power)
            cumulative_power = np.cumsum(psd)
            total_psd_power = cumulative_power[-1]
            edge_95_idx = np.where(cumulative_power >= 0.95 * total_psd_power)[0]
            spectral_edge_95 = freqs[edge_95_idx[0]] if len(edge_95_idx) > 0 else 0.0
            
            # Median frequency
            median_power = 0.5 * total_psd_power
            median_freq_idx = np.where(cumulative_power >= median_power)[0]
            median_frequency = freqs[median_freq_idx[0]] if len(median_freq_idx) > 0 else 0.0
            
            # Peak frequency in alpha band
            alpha_mask = (freqs >= self.config.alpha_low) & (freqs <= self.config.alpha_high)
            alpha_psd = psd[alpha_mask]
            alpha_freqs = freqs[alpha_mask]
            peak_alpha_freq = alpha_freqs[np.argmax(alpha_psd)] if len(alpha_psd) > 0 else 0.0
            
            return {
                'delta_power': delta_power,
                'theta_power': theta_power,
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'gamma_power': gamma_power,
                'total_power': total_power,
                'relative_alpha': rel_alpha,
                'relative_beta': rel_beta,
                'spectral_edge_95': spectral_edge_95,
                'median_frequency': median_frequency,
                'peak_alpha_frequency': peak_alpha_freq
            }
            
        except Exception as e:
            logger.error(f"Spectral feature extraction error: {e}")
            return self._empty_spectral_features()
    
    def compute_time_domain_features(
        self,
        signal_data: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute time-domain features.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            dict: Time-domain features
        """
        try:
            if len(signal_data) == 0:
                return self._empty_time_features()
            
            # Basic statistics
            mean_val = np.mean(signal_data)
            std_val = np.std(signal_data)
            variance = np.var(signal_data)
            
            # Peak-to-peak amplitude
            peak_to_peak = np.ptp(signal_data)
            
            # RMS (root mean square)
            rms = np.sqrt(np.mean(signal_data ** 2))
            
            # Higher order statistics
            skewness = skew(signal_data)
            kurt = kurtosis(signal_data)
            
            # Zero crossing rate
            zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
            zero_crossing_rate = zero_crossings / len(signal_data)
            
            # Signal energy
            energy = np.sum(signal_data ** 2)
            
            # Maximum absolute amplitude
            max_amplitude = np.max(np.abs(signal_data))
            
            return {
                'mean': mean_val,
                'std': std_val,
                'variance': variance,
                'peak_to_peak': peak_to_peak,
                'rms': rms,
                'skewness': skewness,
                'kurtosis': kurt,
                'zero_crossing_rate': zero_crossing_rate,
                'energy': energy,
                'max_amplitude': max_amplitude
            }
            
        except Exception as e:
            logger.error(f"Time-domain feature extraction error: {e}")
            return self._empty_time_features()
    
    def extract_all_features(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> Dict[str, any]:
        """
        Extract complete feature set from both hemispheres.
        
        Args:
            left_signal: Preprocessed EEG from left hemisphere
            right_signal: Preprocessed EEG from right hemisphere
        
        Returns:
            dict: Complete feature dictionary
        """
        features = {}
        
        try:
            # Alpha band features
            left_alpha_power, right_alpha_power = self.extract_alpha_power(
                left_signal, right_signal
            )
            features['left_alpha_power'] = left_alpha_power
            features['right_alpha_power'] = right_alpha_power
            
            # Alpha envelope statistics
            left_env, right_env = self.extract_alpha_envelope(left_signal, right_signal)
            features['left_alpha_envelope_mean'] = np.mean(left_env)
            features['right_alpha_envelope_mean'] = np.mean(right_env)
            features['left_alpha_envelope_std'] = np.std(left_env)
            features['right_alpha_envelope_std'] = np.std(right_env)
            
            # Spectral features for each channel
            left_spectral = self.compute_spectral_features(left_signal)
            right_spectral = self.compute_spectral_features(right_signal)
            
            for key, value in left_spectral.items():
                features[f'left_{key}'] = value
            for key, value in right_spectral.items():
                features[f'right_{key}'] = value
            
            # Time-domain features for each channel
            left_time = self.compute_time_domain_features(left_signal)
            right_time = self.compute_time_domain_features(right_signal)
            
            for key, value in left_time.items():
                features[f'left_{key}'] = value
            for key, value in right_time.items():
                features[f'right_{key}'] = value
            
            # Cross-hemisphere features
            features['alpha_asymmetry'] = left_alpha_power - right_alpha_power
            features['alpha_ratio'] = (
                left_alpha_power / right_alpha_power
                if right_alpha_power > 0 else 0.0
            )
            
            # Hemispheric coherence (correlation)
            if len(left_signal) == len(right_signal) and len(left_signal) > 1:
                left_alpha = self.filter.extract_alpha_band(left_signal)
                right_alpha = self.filter.extract_alpha_band(right_signal)
                coherence = np.corrcoef(left_alpha, right_alpha)[0, 1]
                features['alpha_coherence'] = coherence if not np.isnan(coherence) else 0.0
            else:
                features['alpha_coherence'] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return features
    
    def extract_minimal_features(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract minimal feature set optimized for real-time processing.
        
        Focuses on alpha power asymmetry for attention direction detection.
        
        Args:
            left_signal: Preprocessed EEG from left hemisphere
            right_signal: Preprocessed EEG from right hemisphere
        
        Returns:
            dict: Minimal feature set
        """
        try:
            # Primary features: alpha power
            left_alpha_power, right_alpha_power = self.extract_alpha_power(
                left_signal, right_signal
            )
            
            # Signal quality indicators
            left_variance = np.var(left_signal)
            right_variance = np.var(right_signal)
            
            # Beta power for artifact detection
            left_beta_power = self.filter.compute_beta_power(left_signal)
            right_beta_power = self.filter.compute_beta_power(right_signal)
            
            return {
                'left_alpha_power': left_alpha_power,
                'right_alpha_power': right_alpha_power,
                'left_variance': left_variance,
                'right_variance': right_variance,
                'left_beta_power': left_beta_power,
                'right_beta_power': right_beta_power
            }
            
        except Exception as e:
            logger.error(f"Minimal feature extraction error: {e}")
            return {
                'left_alpha_power': 0.0,
                'right_alpha_power': 0.0,
                'left_variance': 0.0,
                'right_variance': 0.0,
                'left_beta_power': 0.0,
                'right_beta_power': 0.0
            }
    
    @staticmethod
    def _empty_spectral_features() -> Dict[str, float]:
        """Return empty spectral features dictionary."""
        return {
            'delta_power': 0.0,
            'theta_power': 0.0,
            'alpha_power': 0.0,
            'beta_power': 0.0,
            'gamma_power': 0.0,
            'total_power': 0.0,
            'relative_alpha': 0.0,
            'relative_beta': 0.0,
            'spectral_edge_95': 0.0,
            'median_frequency': 0.0,
            'peak_alpha_frequency': 0.0
        }
    
    @staticmethod
    def _empty_time_features() -> Dict[str, float]:
        """Return empty time-domain features dictionary."""
        return {
            'mean': 0.0,
            'std': 0.0,
            'variance': 0.0,
            'peak_to_peak': 0.0,
            'rms': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0,
            'zero_crossing_rate': 0.0,
            'energy': 0.0,
            'max_amplitude': 0.0
        }


def create_feature_extractor(
    signal_config: Optional[SignalConfig] = None
) -> FeatureExtractor:
    """
    Factory function to create a FeatureExtractor instance.
    
    Args:
        signal_config: Signal configuration. If None, uses global config.
    
    Returns:
        FeatureExtractor: Configured feature extractor instance
    """
    return FeatureExtractor(signal_config)
