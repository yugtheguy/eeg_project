"""
Configuration module for EEG Signal Processing System.

Contains all parameters, thresholds, and tuning constants used across modules.
Uses dataclasses for clean configuration management.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class SerialConfig:
    """Configuration for serial communication with Arduino."""
    port: str = "COM7"  # Default port, will auto-detect if None
    baudrate: int = 115200
    timeout: float = 1.0
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 10


@dataclass
class SignalConfig:
    """Configuration for signal processing parameters."""
    sampling_rate: float = 250.0  # Hz
    window_size: float = 2.0  # seconds
    window_overlap: float = 0.5  # 50% overlap
    
    # Filter parameters
    notch_freq: float = 50.0  # Hz (power line frequency)
    notch_quality: float = 30.0  # Q factor for notch filter
    bandpass_low: float = 1.0  # Hz
    bandpass_high: float = 40.0  # Hz
    filter_order: int = 4
    
    # Alpha band parameters
    alpha_low: float = 8.0  # Hz
    alpha_high: float = 12.0  # Hz
    
    # Beta band for artifact detection
    beta_low: float = 13.0  # Hz
    beta_high: float = 30.0  # Hz


@dataclass
class DecisionConfig:
    """Configuration for focus detection based on alpha suppression."""
    # Focus detection thresholds (suppression ratio)
    focus_threshold: float = 0.7  # Ratio < this → FOCUSED
    relax_threshold: float = 1.1  # Ratio > this → RELAXED
    # Between thresholds → NEUTRAL
    
    # Smoothing parameters
    decision_smoothing_window: int = 5  # Number of decisions to smooth
    suppression_history_size: int = 20  # Number of ratios to keep
    min_confidence: float = 0.6  # Minimum confidence for non-NEUTRAL decision
    
    # Baseline calibration
    calibration_duration: float = 10.0  # Seconds for baseline calibration
    min_quality_score: float = 50.0  # Minimum quality to accept data
    min_snr_threshold: float = 0.0  # Minimum SNR in dB (lowered for testing)


@dataclass
class ArtifactConfig:
    """Configuration for artifact detection and rejection."""
    # Variance-based artifact detection
    variance_threshold_multiplier: float = 3.0  # Multiple of median variance
    
    # High beta power artifact detection (muscle artifacts)
    beta_power_threshold: float = 100.0  # μV²
    
    # Signal quality thresholds
    min_snr_db: float = 5.0  # Minimum acceptable SNR in dB
    max_line_noise_power: float = 50.0  # Maximum allowable 50Hz noise power
    
    # Saturation detection
    saturation_threshold: float = 0.95  # Fraction of max ADC range


@dataclass
class LoggingConfig:
    """Configuration for data logging."""
    enable_csv_logging: bool = True
    csv_filename: str = "eeg_data_log.csv"
    log_interval: int = 1  # Log every N samples
    
    # What to log
    log_raw_signal: bool = True
    log_filtered_signal: bool = True
    log_band_powers: bool = True
    log_lateralization_index: bool = True
    log_decisions: bool = True
    log_quality_metrics: bool = True


@dataclass
class VisualizationConfig:
    """Configuration for real-time visualization."""
    enable_visualization: bool = True
    plot_update_interval: float = 0.1  # seconds
    display_window_duration: float = 10.0  # seconds of data to display
    
    # What to visualize
    show_raw_signal: bool = True
    show_filtered_signal: bool = True
    show_alpha_power: bool = True
    show_lateralization_index: bool = True
    show_decisions: bool = True


@dataclass
class SystemConfig:
    """Master configuration containing all sub-configurations."""
    serial: SerialConfig = None
    signal: SignalConfig = None
    decision: DecisionConfig = None
    artifact: ArtifactConfig = None
    logging: LoggingConfig = None
    visualization: VisualizationConfig = None
    
    def __post_init__(self):
        """Initialize sub-configurations with defaults if not provided."""
        if self.serial is None:
            self.serial = SerialConfig()
        if self.signal is None:
            self.signal = SignalConfig()
        if self.decision is None:
            self.decision = DecisionConfig()
        if self.artifact is None:
            self.artifact = ArtifactConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
    
    @property
    def window_samples(self) -> int:
        """Calculate number of samples in processing window."""
        return int(self.signal.window_size * self.signal.sampling_rate)
    
    @property
    def overlap_samples(self) -> int:
        """Calculate number of overlapping samples between windows."""
        return int(self.window_samples * self.signal.window_overlap)
    
    @property
    def hop_samples(self) -> int:
        """Calculate hop size (number of new samples per window)."""
        return self.window_samples - self.overlap_samples


# Global configuration instance
config = SystemConfig()


def get_config() -> SystemConfig:
    """
    Get the global configuration instance.
    
    Returns:
        SystemConfig: The system configuration object
    """
    return config


def update_config(**kwargs) -> None:
    """
    Update configuration parameters.
    
    Args:
        **kwargs: Configuration parameters to update
    
    Example:
        update_config(serial__port="COM4", signal__sampling_rate=500.0)
    """
    global config
    
    for key, value in kwargs.items():
        parts = key.split("__")
        if len(parts) == 2:
            section, param = parts
            if hasattr(config, section):
                section_config = getattr(config, section)
                if hasattr(section_config, param):
                    setattr(section_config, param, value)
                else:
                    raise ValueError(f"Unknown parameter: {param} in section {section}")
            else:
                raise ValueError(f"Unknown section: {section}")
        else:
            raise ValueError(f"Invalid configuration key format: {key}")
