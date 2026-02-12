"""
Signal Quality Metrics and Artifact Detection Module.

Implements quality assessment and artifact detection for EEG signals:
- Signal-to-Noise Ratio (SNR) computation
- Power-based quality metrics
- Artifact detection (muscle artifacts, eye blinks, saturation)
- Data quality scoring
"""

import numpy as np
from scipy import signal as scipy_signal
import logging
from typing import Tuple, Dict, Optional
from enum import Enum

from config import ArtifactConfig, SignalConfig, get_config
from filters import SignalFilter


# Configure logging
logger = logging.getLogger(__name__)


class ArtifactType(Enum):
    """Types of artifacts that can be detected."""
    CLEAN = "clean"
    HIGH_VARIANCE = "high_variance"
    MUSCLE_ARTIFACT = "muscle_artifact"
    SATURATION = "saturation"
    LINE_NOISE = "line_noise"
    LOW_SIGNAL = "low_signal"


class SignalQualityMetrics:
    """
    Compute signal quality metrics and detect artifacts in EEG data.
    """
    
    def __init__(
        self,
        artifact_config: Optional[ArtifactConfig] = None,
        signal_config: Optional[SignalConfig] = None
    ):
        """
        Initialize signal quality metrics calculator.
        
        Args:
            artifact_config: Artifact detection configuration
            signal_config: Signal processing configuration
        """
        if artifact_config is None:
            artifact_config = get_config().artifact
        if signal_config is None:
            signal_config = get_config().signal
        
        self.artifact_config = artifact_config
        self.signal_config = signal_config
        self.filter = SignalFilter(signal_config)
        
        # Adaptive baseline statistics
        self.variance_history = []
        self.median_variance = None
        
        logger.info("SignalQualityMetrics initialized")
    
    def compute_snr(
        self,
        signal_data: np.ndarray,
        noise_band: Tuple[float, float] = (30.0, 40.0)
    ) -> float:
        """
        Compute Signal-to-Noise Ratio (SNR) in dB.
        
        Uses alpha band (8-12 Hz) as signal and high frequencies as noise.
        
        Args:
            signal_data: Preprocessed EEG signal
            noise_band: Frequency range to consider as noise (Hz)
        
        Returns:
            float: SNR in decibels
        """
        try:
            if len(signal_data) < self.signal_config.sampling_rate:
                logger.warning("Signal too short for reliable SNR estimation")
                return 0.0
            
            # Compute signal power (alpha band)
            signal_power = self.filter.compute_alpha_power(signal_data)
            
            # Compute noise power (high frequency band)
            noise_power = self.filter.compute_band_power(
                signal_data,
                noise_band[0],
                noise_band[1]
            )
            
            # Avoid division by zero
            if noise_power <= 0:
                return 0.0
            
            # SNR in dB
            snr_db = 10 * np.log10(signal_power / noise_power)
            
            return snr_db
            
        except Exception as e:
            logger.error(f"SNR computation error: {e}")
            return 0.0
    
    def detect_saturation(
        self,
        signal_data: np.ndarray,
        adc_range: Tuple[float, float] = (0.0, 1023.0)
    ) -> bool:
        """
        Detect signal saturation (clipping).
        
        Args:
            signal_data: Raw or preprocessed signal
            adc_range: Expected ADC range (min, max)
        
        Returns:
            bool: True if saturation detected
        """
        try:
            adc_min, adc_max = adc_range
            threshold_low = adc_min + (adc_max - adc_min) * 0.05
            threshold_high = adc_max - (adc_max - adc_min) * 0.05
            
            # Count samples near saturation
            saturated_samples = np.sum(
                (signal_data <= threshold_low) | (signal_data >= threshold_high)
            )
            
            saturation_ratio = saturated_samples / len(signal_data)
            
            is_saturated = saturation_ratio > self.artifact_config.saturation_threshold
            
            if is_saturated:
                logger.warning(f"Saturation detected: {saturation_ratio:.2%} of samples")
            
            return is_saturated
            
        except Exception as e:
            logger.error(f"Saturation detection error: {e}")
            return False
    
    def detect_muscle_artifact(self, signal_data: np.ndarray) -> bool:
        """
        Detect muscle artifacts based on high beta power.
        
        Muscle artifacts typically have high power in beta/gamma frequencies.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            bool: True if muscle artifact detected
        """
        try:
            beta_power = self.filter.compute_beta_power(signal_data)
            
            is_artifact = beta_power > self.artifact_config.beta_power_threshold
            
            if is_artifact:
                logger.debug(f"Muscle artifact detected: beta power = {beta_power:.2f}")
            
            return is_artifact
            
        except Exception as e:
            logger.error(f"Muscle artifact detection error: {e}")
            return False
    
    def detect_high_variance_artifact(self, signal_data: np.ndarray) -> bool:
        """
        Detect artifacts based on abnormally high variance.
        
        Uses adaptive threshold based on variance history.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            bool: True if high variance artifact detected
        """
        try:
            current_variance = np.var(signal_data)
            
            # Update variance history
            self.variance_history.append(current_variance)
            if len(self.variance_history) > 100:
                self.variance_history.pop(0)
            
            # Compute median variance
            if len(self.variance_history) >= 10:
                self.median_variance = np.median(self.variance_history)
            else:
                # Not enough history yet
                return False
            
            # Check if current variance exceeds threshold
            threshold = (
                self.median_variance *
                self.artifact_config.variance_threshold_multiplier
            )
            
            is_artifact = current_variance > threshold
            
            if is_artifact:
                logger.debug(
                    f"High variance artifact: current={current_variance:.2f}, "
                    f"threshold={threshold:.2f}"
                )
            
            return is_artifact
            
        except Exception as e:
            logger.error(f"Variance artifact detection error: {e}")
            return False
    
    def detect_line_noise(self, signal_data: np.ndarray) -> bool:
        """
        Detect excessive power line noise (50/60 Hz).
        
        Args:
            signal_data: Raw or minimally filtered signal
        
        Returns:
            bool: True if excessive line noise detected
        """
        try:
            line_noise_power = self.filter.detect_line_noise(signal_data)
            
            is_noisy = line_noise_power > self.artifact_config.max_line_noise_power
            
            if is_noisy:
                logger.debug(f"Line noise detected: power = {line_noise_power:.2f}")
            
            return is_noisy
            
        except Exception as e:
            logger.error(f"Line noise detection error: {e}")
            return False
    
    def detect_low_signal(self, signal_data: np.ndarray) -> bool:
        """
        Detect abnormally low signal amplitude.
        
        May indicate disconnected electrode or hardware issue.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            bool: True if signal is abnormally low
        """
        try:
            signal_power = np.mean(signal_data ** 2)
            
            # Very low threshold - just checking if signal exists
            min_expected_power = 1.0
            
            is_low = signal_power < min_expected_power
            
            if is_low:
                logger.warning(f"Low signal detected: power = {signal_power:.4f}")
            
            return is_low
            
        except Exception as e:
            logger.error(f"Low signal detection error: {e}")
            return False
    
    def detect_artifacts(self, signal_data: np.ndarray) -> Tuple[bool, ArtifactType]:
        """
        Comprehensive artifact detection.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            tuple: (is_artifact, artifact_type)
        """
        try:
            # Check for low signal first
            if self.detect_low_signal(signal_data):
                return True, ArtifactType.LOW_SIGNAL
            
            # Check for saturation
            if self.detect_saturation(signal_data):
                return True, ArtifactType.SATURATION
            
            # Check for muscle artifacts
            if self.detect_muscle_artifact(signal_data):
                return True, ArtifactType.MUSCLE_ARTIFACT
            
            # Check for high variance
            if self.detect_high_variance_artifact(signal_data):
                return True, ArtifactType.HIGH_VARIANCE
            
            # Check for line noise
            if self.detect_line_noise(signal_data):
                return True, ArtifactType.LINE_NOISE
            
            # No artifacts detected
            return False, ArtifactType.CLEAN
            
        except Exception as e:
            logger.error(f"Artifact detection error: {e}")
            return True, ArtifactType.HIGH_VARIANCE
    
    def compute_quality_score(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> float:
        """
        Compute overall signal quality score (0-100).
        
        Higher score indicates better signal quality.
        
        Args:
            left_signal: Preprocessed EEG from left hemisphere
            right_signal: Preprocessed EEG from right hemisphere
        
        Returns:
            float: Quality score (0-100)
        """
        try:
            score = 100.0
            
            # SNR contribution (up to -30 points)
            left_snr = self.compute_snr(left_signal)
            right_snr = self.compute_snr(right_signal)
            avg_snr = (left_snr + right_snr) / 2
            
            if avg_snr < self.artifact_config.min_snr_db:
                snr_penalty = (self.artifact_config.min_snr_db - avg_snr) * 3
                score -= min(snr_penalty, 30)
            
            # Artifact detection (up to -40 points)
            left_artifact, left_type = self.detect_artifacts(left_signal)
            right_artifact, right_type = self.detect_artifacts(right_signal)
            
            if left_artifact or right_artifact:
                score -= 40
            
            # Variance stability (up to -15 points)
            left_variance = np.var(left_signal)
            right_variance = np.var(right_signal)
            
            if self.median_variance is not None and self.median_variance > 0:
                variance_ratio = abs(left_variance - right_variance) / self.median_variance
                if variance_ratio > 2.0:
                    score -= 15
            
            # Signal symmetry (up to -15 points)
            left_power = np.mean(left_signal ** 2)
            right_power = np.mean(right_signal ** 2)
            
            if left_power > 0 and right_power > 0:
                power_ratio = max(left_power, right_power) / min(left_power, right_power)
                if power_ratio > 10:  # More than 10x difference
                    score -= 15
            
            # Ensure score is in valid range
            score = max(0.0, min(100.0, score))
            
            return score
            
        except Exception as e:
            logger.error(f"Quality score computation error: {e}")
            return 0.0
    
    def compute_channel_quality(self, signal_data: np.ndarray) -> Dict[str, any]:
        """
        Compute detailed quality metrics for a single channel.
        
        Args:
            signal_data: Preprocessed EEG signal
        
        Returns:
            dict: Quality metrics dictionary
        """
        try:
            # SNR
            snr = self.compute_snr(signal_data)
            
            # Artifact detection
            has_artifact, artifact_type = self.detect_artifacts(signal_data)
            
            # Power metrics
            signal_power = np.mean(signal_data ** 2)
            signal_variance = np.var(signal_data)
            
            # Alpha power
            alpha_power = self.filter.compute_alpha_power(signal_data)
            
            # Beta power
            beta_power = self.filter.compute_beta_power(signal_data)
            
            return {
                'snr_db': snr,
                'has_artifact': has_artifact,
                'artifact_type': artifact_type.value,
                'signal_power': signal_power,
                'variance': signal_variance,
                'alpha_power': alpha_power,
                'beta_power': beta_power,
                'quality_ok': not has_artifact and snr >= self.artifact_config.min_snr_db
            }
            
        except Exception as e:
            logger.error(f"Channel quality computation error: {e}")
            return {
                'snr_db': 0.0,
                'has_artifact': True,
                'artifact_type': 'unknown',
                'signal_power': 0.0,
                'variance': 0.0,
                'alpha_power': 0.0,
                'beta_power': 0.0,
                'quality_ok': False
            }
    
    def reset_adaptive_metrics(self) -> None:
        """Reset adaptive baseline statistics."""
        self.variance_history.clear()
        self.median_variance = None
        logger.info("Adaptive metrics reset")


def create_quality_metrics(
    artifact_config: Optional[ArtifactConfig] = None,
    signal_config: Optional[SignalConfig] = None
) -> SignalQualityMetrics:
    """
    Factory function to create a SignalQualityMetrics instance.
    
    Args:
        artifact_config: Artifact detection configuration
        signal_config: Signal processing configuration
    
    Returns:
        SignalQualityMetrics: Configured quality metrics instance
    """
    return SignalQualityMetrics(artifact_config, signal_config)
