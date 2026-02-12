"""
EEG System Validation & Verification Module.

Industry-grade testing protocol for real-time EEG attention decoding system.
Validates signal integrity, filtering, feature extraction, artifact rejection,
lateralization index, decision stability, and real-time performance.

NO MOCK DATA - All tests use real EEG signals.
"""

import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from pathlib import Path
import json
import csv
from scipy import signal as scipy_signal
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from config import get_config, SystemConfig
from acquisition import SerialAcquisition
from filters import SignalFilter
from features import FeatureExtractor
from metrics import SignalQualityMetrics
from decision import DecisionEngine
from realtime_engine import RealtimeEEGEngine


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ValidationResults:
    """Container for validation test results."""
    
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0
        self.results = {}
        self.metrics = {}
        self.pass_criteria = {}
    
    def add_result(self, test_name: str, passed: bool, metrics: Dict, criteria: str = ""):
        """Add a test result."""
        self.results[test_name] = {
            'passed': passed,
            'metrics': metrics,
            'criteria': criteria,
            'timestamp': time.time()
        }
        
        if passed:
            self.tests_passed += 1
        else:
            self.tests_failed += 1
        
        self.metrics.update(metrics)
    
    def get_summary(self) -> Dict:
        """Get validation summary."""
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        return {
            'total_tests': total_tests,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'pass_rate_percent': pass_rate,
            'all_metrics': self.metrics,
            'individual_results': self.results
        }
    
    def export_json(self, filename: str = "validation_report.json"):
        """Export results to JSON."""
        with open(filename, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)
        logger.info(f"Validation report exported to {filename}")
    
    def export_csv(self, filename: str = "validation_report.csv"):
        """Export results to CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test Name', 'Result', 'Metric', 'Value'])
            
            for test_name, data in self.results.items():
                result_str = 'PASS' if data['passed'] else 'FAIL'
                for metric_name, value in data['metrics'].items():
                    writer.writerow([test_name, result_str, metric_name, value])
        
        logger.info(f"Validation report exported to {filename}")


class EEGValidator:
    """
    Comprehensive EEG system validator.
    
    Performs industry-grade testing on all system components.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize validator."""
        if config is None:
            config = get_config()
        
        self.config = config
        self.results = ValidationResults()
        
        # Initialize components
        self.acquisition = SerialAcquisition(config.serial)
        self.filter = SignalFilter(config.signal)
        self.feature_extractor = FeatureExtractor(config.signal)
        self.quality_metrics = SignalQualityMetrics(config.artifact, config.signal)
        self.decision_engine = DecisionEngine(config.decision)
        
        logger.info("EEGValidator initialized")
    
    # =============================================================================
    # SECTION 1: DATA ACQUISITION VALIDATION
    # =============================================================================
    
    def validate_sampling_rate(self, duration: float = 60.0) -> bool:
        """
        1.1 Sampling Rate Verification
        
        Logs timestamps for specified duration and computes sampling statistics.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            bool: True if sampling rate is stable
        """
        logger.info("="*70)
        logger.info("TEST 1.1: Sampling Rate Verification")
        logger.info("="*70)
        
        if not self.acquisition.is_connected:
            if not self.acquisition.connect():
                logger.error("Cannot validate sampling rate - no connection")
                self.results.add_result(
                    "sampling_rate",
                    False,
                    {'error': 'no_connection'},
                    "Connection required"
                )
                return False
        
        timestamps = []
        start_time = time.time()
        
        logger.info(f"Collecting timestamps for {duration} seconds...")
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                timestamp, _, _ = data
                timestamps.append(time.time())
            time.sleep(0.001)
        
        if len(timestamps) < 2:
            logger.error("Insufficient samples collected")
            self.results.add_result(
                "sampling_rate",
                False,
                {'samples_collected': len(timestamps)},
                "Need at least 2 samples"
            )
            return False
        
        # Compute sampling intervals
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        effective_fs = 1.0 / mean_interval if mean_interval > 0 else 0
        
        # Compute jitter
        expected_interval = 1.0 / self.config.signal.sampling_rate
        jitter_percent = (std_interval / expected_interval) * 100
        
        # Log results
        logger.info(f"Samples collected: {len(timestamps)}")
        logger.info(f"Mean interval: {mean_interval*1000:.3f} ms")
        logger.info(f"Std interval: {std_interval*1000:.3f} ms")
        logger.info(f"Effective sampling rate: {effective_fs:.2f} Hz")
        logger.info(f"Expected: {self.config.signal.sampling_rate:.2f} Hz")
        logger.info(f"Jitter: {jitter_percent:.2f}%")
        
        # Pass criteria: jitter < 10%
        passed = jitter_percent < 10.0
        
        if passed:
            logger.info("✅ PASS: Sampling rate stable")
        else:
            logger.error(f"❌ FAIL: Jitter ({jitter_percent:.2f}%) exceeds 10%")
        
        self.results.add_result(
            "sampling_rate",
            passed,
            {
                'samples_collected': len(timestamps),
                'mean_interval_ms': mean_interval * 1000,
                'std_interval_ms': std_interval * 1000,
                'effective_fs_hz': effective_fs,
                'jitter_percent': jitter_percent
            },
            "Jitter < 10%"
        )
        
        return passed
    
    def validate_adc_scaling(self, duration: float = 10.0) -> bool:
        """
        1.2 ADC Scaling Validation
        
        Validates raw ADC values are in expected range.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            bool: True if scaling is correct
        """
        logger.info("="*70)
        logger.info("TEST 1.2: ADC Scaling Validation")
        logger.info("="*70)
        
        if not self.acquisition.is_connected:
            logger.error("Cannot validate ADC - no connection")
            self.results.add_result(
                "adc_scaling",
                False,
                {'error': 'no_connection'},
                "Connection required"
            )
            return False
        
        left_values = []
        right_values = []
        start_time = time.time()
        
        logger.info(f"Collecting ADC values for {duration} seconds...")
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_values.append(left)
                right_values.append(right)
            time.sleep(0.001)
        
        if len(left_values) < 10:
            logger.error("Insufficient samples for ADC validation")
            self.results.add_result(
                "adc_scaling",
                False,
                {'samples': len(left_values)},
                "Need at least 10 samples"
            )
            return False
        
        left_arr = np.array(left_values)
        right_arr = np.array(right_values)
        
        # Statistics
        metrics = {
            'left_min': float(np.min(left_arr)),
            'left_max': float(np.max(left_arr)),
            'left_mean': float(np.mean(left_arr)),
            'left_std': float(np.std(left_arr)),
            'right_min': float(np.min(right_arr)),
            'right_max': float(np.max(right_arr)),
            'right_mean': float(np.mean(right_arr)),
            'right_std': float(np.std(right_arr)),
            'samples': len(left_values)
        }
        
        logger.info(f"Left channel:  Min={metrics['left_min']:.2f}, Max={metrics['left_max']:.2f}, Mean={metrics['left_mean']:.2f}")
        logger.info(f"Right channel: Min={metrics['right_min']:.2f}, Max={metrics['right_max']:.2f}, Mean={metrics['right_mean']:.2f}")
        
        # Check if values are in reasonable ADC range
        # Typical 10-bit: 0-1023, 12-bit: 0-4095
        max_value = max(metrics['left_max'], metrics['right_max'])
        min_value = min(metrics['left_min'], metrics['right_min'])
        
        # Pass if values are within expected range and not constant
        in_range = (0 <= min_value) and (max_value <= 5000)
        has_variance = (metrics['left_std'] > 1.0) and (metrics['right_std'] > 1.0)
        
        passed = in_range and has_variance
        
        if passed:
            logger.info("✅ PASS: ADC scaling valid")
        else:
            if not in_range:
                logger.error(f"❌ FAIL: Values out of range (0-5000)")
            if not has_variance:
                logger.error(f"❌ FAIL: Insufficient variance (signal may be constant)")
        
        self.results.add_result("adc_scaling", passed, metrics, "0 <= ADC <= 5000, std > 1.0")
        
        return passed
    
    def validate_serial_integrity(self, duration: float = 60.0) -> bool:
        """
        1.3 Serial Integrity Test
        
        Monitors corruption rate and connection stability.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            bool: True if corruption < 1%
        """
        logger.info("="*70)
        logger.info("TEST 1.3: Serial Integrity Test")
        logger.info("="*70)
        
        if not self.acquisition.is_connected:
            logger.error("Cannot validate serial integrity - no connection")
            self.results.add_result(
                "serial_integrity",
                False,
                {'error': 'no_connection'},
                "Connection required"
            )
            return False
        
        # Reset counters
        initial_received = self.acquisition.packets_received
        initial_corrupted = self.acquisition.packets_corrupted
        
        start_time = time.time()
        logger.info(f"Monitoring serial integrity for {duration} seconds...")
        
        # Just let it run
        while time.time() - start_time < duration:
            self.acquisition.read_sample()
            time.sleep(0.001)
        
        # Get final stats
        stats = self.acquisition.get_statistics()
        
        total_received = self.acquisition.packets_received - initial_received
        total_corrupted = self.acquisition.packets_corrupted - initial_corrupted
        
        corruption_rate = (total_corrupted / (total_received + total_corrupted) * 100) if (total_received + total_corrupted) > 0 else 0
        
        metrics = {
            'packets_received': total_received,
            'packets_corrupted': total_corrupted,
            'corruption_rate_percent': corruption_rate,
            'time_since_last_packet': stats['time_since_last_packet']
        }
        
        logger.info(f"Packets received: {total_received}")
        logger.info(f"Packets corrupted: {total_corrupted}")
        logger.info(f"Corruption rate: {corruption_rate:.2f}%")
        
        passed = corruption_rate < 1.0
        
        if passed:
            logger.info("✅ PASS: Serial integrity good")
        else:
            logger.error(f"❌ FAIL: Corruption rate ({corruption_rate:.2f}%) exceeds 1%")
        
        self.results.add_result("serial_integrity", passed, metrics, "Corruption < 1%")
        
        return passed
    
    # =============================================================================
    # SECTION 2: FILTER VALIDATION
    # =============================================================================
    
    def validate_frequency_response(self) -> bool:
        """
        2.1 Frequency Response Verification
        
        Plots and validates filter frequency responses.
        
        Returns:
            bool: True if filters meet specifications
        """
        logger.info("="*70)
        logger.info("TEST 2.1: Frequency Response Verification")
        logger.info("="*70)
        
        fs = self.config.signal.sampling_rate
        
        # Test notch filter
        w_notch, h_notch = scipy_signal.freqz(
            self.filter.notch_b,
            self.filter.notch_a,
            worN=8000,
            fs=fs
        )
        
        # Test bandpass filter
        w_bp, h_bp = scipy_signal.sosfreqz(
            self.filter.bandpass_sos,
            worN=8000,
            fs=fs
        )
        
        # Test alpha filter
        w_alpha, h_alpha = scipy_signal.sosfreqz(
            self.filter.alpha_sos,
            worN=8000,
            fs=fs
        )
        
        # Check notch attenuation at 50 Hz
        idx_50hz = np.argmin(np.abs(w_notch - 50.0))
        notch_attenuation_db = 20 * np.log10(np.abs(h_notch[idx_50hz]))
        
        # Check bandpass passband (10 Hz)
        idx_10hz = np.argmin(np.abs(w_bp - 10.0))
        bp_passband_db = 20 * np.log10(np.abs(h_bp[idx_10hz]))
        
        # Check alpha center (10 Hz)
        idx_alpha = np.argmin(np.abs(w_alpha - 10.0))
        alpha_passband_db = 20 * np.log10(np.abs(h_alpha[idx_alpha]))
        
        metrics = {
            'notch_attenuation_50hz_db': float(notch_attenuation_db),
            'bandpass_passband_10hz_db': float(bp_passband_db),
            'alpha_passband_10hz_db': float(alpha_passband_db)
        }
        
        logger.info(f"Notch filter attenuation @ 50 Hz: {notch_attenuation_db:.2f} dB")
        logger.info(f"Bandpass gain @ 10 Hz: {bp_passband_db:.2f} dB")
        logger.info(f"Alpha filter gain @ 10 Hz: {alpha_passband_db:.2f} dB")
        
        # Criteria
        notch_ok = notch_attenuation_db < -20  # At least 20 dB attenuation
        bp_ok = bp_passband_db > -3  # Less than 3 dB loss in passband
        alpha_ok = alpha_passband_db > -3
        
        passed = notch_ok and bp_ok and alpha_ok
        
        # Create frequency response plot
        try:
            fig, axes = plt.subplots(3, 1, figsize=(10, 12))
            
            # Notch filter
            axes[0].plot(w_notch, 20 * np.log10(np.abs(h_notch)))
            axes[0].axvline(50, color='r', linestyle='--', label='50 Hz')
            axes[0].set_title('Notch Filter Frequency Response')
            axes[0].set_xlabel('Frequency (Hz)')
            axes[0].set_ylabel('Magnitude (dB)')
            axes[0].grid(True)
            axes[0].legend()
            axes[0].set_xlim([0, 100])
            
            # Bandpass filter
            axes[1].plot(w_bp, 20 * np.log10(np.abs(h_bp)))
            axes[1].axvline(1, color='g', linestyle='--', label='Low cutoff')
            axes[1].axvline(40, color='r', linestyle='--', label='High cutoff')
            axes[1].set_title('Bandpass Filter (1-40 Hz) Frequency Response')
            axes[1].set_xlabel('Frequency (Hz)')
            axes[1].set_ylabel('Magnitude (dB)')
            axes[1].grid(True)
            axes[1].legend()
            
            # Alpha filter
            axes[2].plot(w_alpha, 20 * np.log10(np.abs(h_alpha)))
            axes[2].axvline(8, color='g', linestyle='--', label='Alpha low')
            axes[2].axvline(12, color='r', linestyle='--', label='Alpha high')
            axes[2].set_title('Alpha Band Filter (8-12 Hz) Frequency Response')
            axes[2].set_xlabel('Frequency (Hz)')
            axes[2].set_ylabel('Magnitude (dB)')
            axes[2].grid(True)
            axes[2].legend()
            axes[2].set_xlim([0, 30])
            
            plt.tight_layout()
            plt.savefig('filter_frequency_response.png', dpi=150)
            plt.close()
            
            logger.info("Frequency response plot saved to filter_frequency_response.png")
            
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
        
        if passed:
            logger.info("✅ PASS: Filter frequency responses valid")
        else:
            logger.error("❌ FAIL: Filter specifications not met")
        
        self.results.add_result(
            "frequency_response",
            passed,
            metrics,
            "Notch < -20dB @ 50Hz, Passband > -3dB"
        )
        
        return passed
    
    def validate_filtering_effect(self, duration: float = 30.0) -> bool:
        """
        2.2 Real Signal Before/After Filtering
        
        Records real EEG and validates filtering effects.
        
        Args:
            duration: Recording duration in seconds
        
        Returns:
            bool: True if filtering improves signal quality
        """
        logger.info("="*70)
        logger.info("TEST 2.2: Real Signal Filtering Effect")
        logger.info("="*70)
        
        if not self.acquisition.is_connected:
            logger.error("Cannot validate filtering - no connection")
            self.results.add_result(
                "filtering_effect",
                False,
                {'error': 'no_connection'},
                "Connection required"
            )
            return False
        
        # Collect raw data
        logger.info(f"Recording {duration} seconds of raw EEG...")
        left_raw = []
        right_raw = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_raw.append(left)
                right_raw.append(right)
            time.sleep(0.001)
        
        if len(left_raw) < 100:
            logger.error("Insufficient data collected")
            self.results.add_result(
                "filtering_effect",
                False,
                {'samples': len(left_raw)},
                "Need at least 100 samples"
            )
            return False
        
        left_raw = np.array(left_raw)
        right_raw = np.array(right_raw)
        
        logger.info(f"Collected {len(left_raw)} samples")
        
        # Apply filtering
        left_filtered = self.filter.preprocess_signal(left_raw)
        right_filtered = self.filter.preprocess_signal(right_raw)
        
        # Extract alpha
        left_alpha = self.filter.extract_alpha_band(left_filtered)
        right_alpha = self.filter.extract_alpha_band(right_filtered)
        
        # Compute metrics
        # 1. Check 50 Hz removal
        line_noise_raw_left = self.filter.detect_line_noise(left_raw)
        line_noise_filtered_left = self.filter.detect_line_noise(left_filtered)
        
        # 2. DC removal
        dc_raw = abs(np.mean(left_raw))
        dc_filtered = abs(np.mean(left_filtered))
        
        # 3. Alpha visibility
        alpha_power = np.mean(left_alpha ** 2)
        
        metrics = {
            'samples_collected': len(left_raw),
            'line_noise_raw': float(line_noise_raw_left),
            'line_noise_filtered': float(line_noise_filtered_left),
            'noise_reduction_ratio': float(line_noise_raw_left / (line_noise_filtered_left + 1e-10)),
            'dc_raw': float(dc_raw),
            'dc_filtered': float(dc_filtered),
            'alpha_power': float(alpha_power)
        }
        
        logger.info(f"Line noise (raw): {line_noise_raw_left:.2f}")
        logger.info(f"Line noise (filtered): {line_noise_filtered_left:.2f}")
        logger.info(f"Noise reduction: {metrics['noise_reduction_ratio']:.2f}x")
        logger.info(f"DC offset (raw): {dc_raw:.2f}")
        logger.info(f"DC offset (filtered): {dc_filtered:.2f}")
        logger.info(f"Alpha power: {alpha_power:.2f}")
        
        # Pass if noise reduced and alpha power is reasonable
        noise_reduced = metrics['noise_reduction_ratio'] > 2.0
        dc_reduced = dc_filtered < dc_raw * 0.5
        alpha_present = alpha_power > 0.1
        
        passed = noise_reduced and alpha_present
        
        # Create visualization
        try:
            fig, axes = plt.subplots(4, 1, figsize=(12, 10))
            
            t = np.arange(len(left_raw)) / self.config.signal.sampling_rate
            plot_samples = min(1000, len(left_raw))
            
            axes[0].plot(t[:plot_samples], left_raw[:plot_samples])
            axes[0].set_title('Raw EEG Signal (Left Channel)')
            axes[0].set_ylabel('ADC Value')
            axes[0].grid(True)
            
            axes[1].plot(t[:plot_samples], left_filtered[:plot_samples])
            axes[1].set_title('After 1-40 Hz Bandpass Filter')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True)
            
            axes[2].plot(t[:plot_samples], left_alpha[:plot_samples])
            axes[2].set_title('Alpha Band (8-12 Hz) Extraction')
            axes[2].set_ylabel('Amplitude')
            axes[2].grid(True)
            
            # Envelope
            envelope = self.filter.compute_envelope(left_alpha[:plot_samples])
            axes[3].plot(t[:plot_samples], envelope)
            axes[3].set_title('Alpha Envelope (Hilbert Transform)')
            axes[3].set_xlabel('Time (s)')
            axes[3].set_ylabel('Amplitude')
            axes[3].grid(True)
            
            plt.tight_layout()
            plt.savefig('filtering_effect.png', dpi=150)
            plt.close()
            
            logger.info("Filtering effect plot saved to filtering_effect.png")
            
        except Exception as e:
            logger.warning(f"Could not create plot: {e}")
        
        if passed:
            logger.info("✅ PASS: Filtering effective")
        else:
            logger.error("❌ FAIL: Filtering not effective enough")
        
        self.results.add_result(
            "filtering_effect",
            passed,
            metrics,
            "Noise reduction > 2x, Alpha power > 0.1"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 3: FEATURE EXTRACTION VALIDATION
    # =============================================================================
    
    def validate_alpha_power_stability(self, duration: float = 60.0) -> bool:
        """
        3.1 Alpha Power Stability Test
        
        Computes alpha power over sliding windows and checks stability.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            bool: True if alpha power is stable enough
        """
        logger.info("="*70)
        logger.info("TEST 3.1: Alpha Power Stability")
        logger.info("="*70)
        
        if not self.acquisition.is_connected:
            logger.error("Cannot validate alpha power - no connection")
            self.results.add_result(
                "alpha_stability",
                False,
                {'error': 'no_connection'},
                "Connection required"
            )
            return False
        
        left_powers = []
        right_powers = []
        
        window_samples = self.config.window_samples
        left_buffer = deque(maxlen=window_samples)
        right_buffer = deque(maxlen=window_samples)
        
        logger.info(f"Computing alpha power over {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_buffer.append(left)
                right_buffer.append(right)
                
                if len(left_buffer) == window_samples:
                    # Compute alpha power
                    left_sig = np.array(list(left_buffer))
                    right_sig = np.array(list(right_buffer))
                    
                    left_filt = self.filter.preprocess_signal(left_sig)
                    right_filt = self.filter.preprocess_signal(right_sig)
                    
                    left_power, right_power = self.feature_extractor.extract_alpha_power(
                        left_filt, right_filt
                    )
                    
                    left_powers.append(left_power)
                    right_powers.append(right_power)
            
            time.sleep(0.001)
        
        if len(left_powers) < 2:
            logger.error("Insufficient windows computed")
            self.results.add_result(
                "alpha_stability",
                False,
                {'windows': len(left_powers)},
                "Need at least 2 windows"
            )
            return False
        
        left_powers = np.array(left_powers)
        right_powers = np.array(right_powers)
        
        # Compute statistics
        left_mean = np.mean(left_powers)
        left_std = np.std(left_powers)
        left_cv = (left_std / left_mean) if left_mean > 0 else 0
        
        right_mean = np.mean(right_powers)
        right_std = np.std(right_powers)
        right_cv = (right_std / right_mean) if right_mean > 0 else 0
        
        metrics = {
            'windows_computed': len(left_powers),
            'left_mean_power': float(left_mean),
            'left_std_power': float(left_std),
            'left_cv': float(left_cv),
            'right_mean_power': float(right_mean),
            'right_std_power': float(right_std),
            'right_cv': float(right_cv)
        }
        
        logger.info(f"Windows computed: {len(left_powers)}")
        logger.info(f"Left:  Mean={left_mean:.2f}, Std={left_std:.2f}, CV={left_cv:.2f}")
        logger.info(f"Right: Mean={right_mean:.2f}, Std={right_std:.2f}, CV={right_cv:.2f}")
        
        # Pass if coefficient of variation < 1.5 (moderate stability)
        passed = left_cv < 1.5 and right_cv < 1.5 and left_mean > 0 and right_mean > 0
        
        if passed:
            logger.info("✅ PASS: Alpha power stable")
        else:
            logger.error("❌ FAIL: Alpha power too variable")
        
        self.results.add_result(
            "alpha_stability",
            passed,
            metrics,
            "CV < 1.5, mean > 0"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 4: ARTIFACT REJECTION VALIDATION
    # =============================================================================
    
    def validate_artifact_detection(self, duration: float = 30.0) -> bool:
        """
        4.1-4.2 Artifact Detection Validation
        
        Tests artifact detection on real data.
        
        Args:
            duration: Test duration in seconds
        
        Returns:
            bool: True if artifact detection works
        """
        logger.info("="*70)
        logger.info("TEST 4.1-4.2: Artifact Detection")
        logger.info("="*70)
        logger.info("Please perform various movements to test artifact detection:")
        logger.info("  - Normal state (10s)")
        logger.info("  - Jaw clench (5s)")
        logger.info("  - Eye blinks (5s)")
        logger.info("  - Normal state (10s)")
        
        if not self.acquisition.is_connected:
            logger.error("Cannot validate artifacts - no connection")
            self.results.add_result(
                "artifact_detection",
                False,
                {'error': 'no_connection'},
                "Connection required"
            )
            return False
        
        artifact_count = 0
        clean_count = 0
        
        window_samples = self.config.window_samples
        left_buffer = deque(maxlen=window_samples)
        right_buffer = deque(maxlen=window_samples)
        
        logger.info(f"Testing artifact detection for {duration} seconds...")
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_buffer.append(left)
                right_buffer.append(right)
                
                if len(left_buffer) == window_samples:
                    left_sig = np.array(list(left_buffer))
                    right_sig = np.array(list(right_buffer))
                    
                    left_filt = self.filter.preprocess_signal(left_sig)
                    right_filt = self.filter.preprocess_signal(right_sig)
                    
                    left_artifact, left_type = self.quality_metrics.detect_artifacts(left_filt)
                    right_artifact, right_type = self.quality_metrics.detect_artifacts(right_filt)
                    
                    if left_artifact or right_artifact:
                        artifact_count += 1
                        logger.debug(f"Artifact: L={left_type.value}, R={right_type.value}")
                    else:
                        clean_count += 1
            
            time.sleep(0.001)
        
        total_windows = artifact_count + clean_count
        artifact_rate = (artifact_count / total_windows * 100) if total_windows > 0 else 0
        
        metrics = {
            'total_windows': total_windows,
            'artifact_windows': artifact_count,
            'clean_windows': clean_count,
            'artifact_rate_percent': artifact_rate
        }
        
        logger.info(f"Total windows: {total_windows}")
        logger.info(f"Artifact windows: {artifact_count}")
        logger.info(f"Clean windows: {clean_count}")
        logger.info(f"Artifact rate: {artifact_rate:.1f}%")
        
        # Pass if system can detect artifacts (at least some should be found if user moved)
        # But not too many (should not be constant false positives)
        passed = artifact_rate > 5.0 and artifact_rate < 50.0
        
        if passed:
            logger.info("✅ PASS: Artifact detection working")
        else:
            logger.warning(f"⚠️  REVIEW: Artifact rate {artifact_rate:.1f}% (expected 5-50%)")
        
        self.results.add_result(
            "artifact_detection",
            passed,
            metrics,
            "5% < artifact rate < 50%"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 5: LATERALIZATION INDEX VALIDATION
    # =============================================================================
    
    def run_controlled_attention_experiment(
        self,
        trials: int = 10,
        duration_per_trial: float = 20.0
    ) -> Tuple[List[float], List[float]]:
        """
        5.1 Controlled Attention Experiment
        
        User focuses left/right alternately. Returns LI distributions.
        
        Args:
            trials: Number of trials
            duration_per_trial: Duration of each trial (split left/right)
        
        Returns:
            Tuple of (li_during_left_focus, li_during_right_focus)
        """
        logger.info("="*70)
        logger.info("TEST 5.1: Controlled Attention Experiment")
        logger.info("="*70)
        logger.info(f"Running {trials} trials, {duration_per_trial}s each")
        logger.info("Follow the prompts to focus LEFT or RIGHT")
        
        if not self.acquisition.is_connected:
            if not self.acquisition.connect():
                logger.error("Cannot run experiment - no connection")
                return [], []
        
        li_left_focus = []
        li_right_focus = []
        
        window_samples = self.config.window_samples
        
        for trial in range(trials):
            logger.info(f"\n{'='*50}")
            logger.info(f"TRIAL {trial + 1}/{trials}")
            logger.info(f"{'='*50}")
            
            # Left focus phase
            logger.info(">>> FOCUS LEFT for 10 seconds <<<")
            input("Press Enter when ready...")
            
            left_buffer = deque(maxlen=window_samples)
            right_buffer = deque(maxlen=window_samples)
            
            start_time = time.time()
            while time.time() - start_time < duration_per_trial / 2:
                data = self.acquisition.read_sample()
                if data is not None:
                    _, left, right = data
                    left_buffer.append(left)
                    right_buffer.append(right)
                    
                    if len(left_buffer) == window_samples:
                        left_sig = np.array(list(left_buffer))
                        right_sig = np.array(list(right_buffer))
                        
                        left_filt = self.filter.preprocess_signal(left_sig)
                        right_filt = self.filter.preprocess_signal(right_sig)
                        
                        left_power, right_power = self.feature_extractor.extract_alpha_power(
                            left_filt, right_filt
                        )
                        
                        li = self.decision_engine.compute_lateralization_index(
                            left_power, right_power
                        )
                        li_left_focus.append(li)
                
                time.sleep(0.001)
            
            # Right focus phase
            logger.info(">>> FOCUS RIGHT for 10 seconds <<<")
            input("Press Enter when ready...")
            
            left_buffer.clear()
            right_buffer.clear()
            
            start_time = time.time()
            while time.time() - start_time < duration_per_trial / 2:
                data = self.acquisition.read_sample()
                if data is not None:
                    _, left, right = data
                    left_buffer.append(left)
                    right_buffer.append(right)
                    
                    if len(left_buffer) == window_samples:
                        left_sig = np.array(list(left_buffer))
                        right_sig = np.array(list(right_buffer))
                        
                        left_filt = self.filter.preprocess_signal(left_sig)
                        right_filt = self.filter.preprocess_signal(right_sig)
                        
                        left_power, right_power = self.feature_extractor.extract_alpha_power(
                            left_filt, right_filt
                        )
                        
                        li = self.decision_engine.compute_lateralization_index(
                            left_power, right_power
                        )
                        li_right_focus.append(li)
                
                time.sleep(0.001)
        
        logger.info(f"\nExperiment complete!")
        logger.info(f"Left focus samples: {len(li_left_focus)}")
        logger.info(f"Right focus samples: {len(li_right_focus)}")
        
        return li_left_focus, li_right_focus
    
    def validate_lateralization_index(
        self,
        trials: int = 5,
        duration_per_trial: float = 20.0
    ) -> bool:
        """
        5.1-5.3 Complete Lateralization Index Validation
        
        Runs controlled experiment and statistical analysis.
        
        Args:
            trials: Number of trials
            duration_per_trial: Duration per trial
        
        Returns:
            bool: True if LI is statistically significant
        """
        logger.info("="*70)
        logger.info("TEST 5: Lateralization Index Validation")
        logger.info("="*70)
        
        # Run experiment
        li_left, li_right = self.run_controlled_attention_experiment(
            trials, duration_per_trial
        )
        
        if len(li_left) < 2 or len(li_right) < 2:
            logger.error("Insufficient data collected")
            self.results.add_result(
                "lateralization_index",
                False,
                {'samples_left': len(li_left), 'samples_right': len(li_right)},
                "Need at least 2 samples per condition"
            )
            return False
        
        li_left = np.array(li_left)
        li_right = np.array(li_right)
        
        # 5.2 Distribution Analysis
        mean_li_left = np.mean(li_left)
        std_li_left = np.std(li_left)
        mean_li_right = np.mean(li_right)
        std_li_right = np.std(li_right)
        
        logger.info("\n" + "="*50)
        logger.info("DISTRIBUTION ANALYSIS")
        logger.info("="*50)
        logger.info(f"LEFT focus:  Mean LI = {mean_li_left:+.3f}, Std = {std_li_left:.3f}")
        logger.info(f"RIGHT focus: Mean LI = {mean_li_right:+.3f}, Std = {std_li_right:.3f}")
        
        # 5.3 Statistical Significance
        t_stat, p_value = ttest_ind(li_left, li_right)
        
        logger.info("\n" + "="*50)
        logger.info("STATISTICAL SIGNIFICANCE (t-test)")
        logger.info("="*50)
        logger.info(f"t-statistic: {t_stat:.3f}")
        logger.info(f"p-value: {p_value:.6f}")
        
        # Check separation
        separation = abs(mean_li_right - mean_li_left)
        logger.info(f"Mean separation: {separation:.3f}")
        
        # Compute overlap (simplified)
        # Distributions overlap if difference < 2 * combined std
        combined_std = (std_li_left + std_li_right) / 2
        overlap_ratio = combined_std / separation if separation > 0 else 999
        
        metrics = {
            'samples_left_focus': len(li_left),
            'samples_right_focus': len(li_right),
            'mean_li_left': float(mean_li_left),
            'std_li_left': float(std_li_left),
            'mean_li_right': float(mean_li_right),
            'std_li_right': float(std_li_right),
            'separation': float(separation),
            't_statistic': float(t_stat),
            'p_value': float(p_value)
        }
        
        # Pass criteria
        statistically_significant = p_value < 0.05
        well_separated = separation > 0.1
        
        passed = statistically_significant and well_separated
        
        # Create histogram
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(li_left, bins=20, alpha=0.5, label='LEFT focus', color='blue')
            plt.hist(li_right, bins=20, alpha=0.5, label='RIGHT focus', color='red')
            plt.axvline(mean_li_left, color='blue', linestyle='--', linewidth=2, label=f'Mean LEFT: {mean_li_left:.3f}')
            plt.axvline(mean_li_right, color='red', linestyle='--', linewidth=2, label=f'Mean RIGHT: {mean_li_right:.3f}')
            plt.xlabel('Lateralization Index')
            plt.ylabel('Frequency')
            plt.title(f'LI Distribution: LEFT vs RIGHT focus\n(p-value: {p_value:.4f})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig('lateralization_index_distribution.png', dpi=150)
            plt.close()
            
            logger.info("\nDistribution plot saved to lateralization_index_distribution.png")
            
        except Exception as e:
            logger.warning(f"Could not create histogram: {e}")
        
        if passed:
            logger.info(f"\n✅ PASS: LI statistically significant (p={p_value:.6f} < 0.05)")
        else:
            if not statistically_significant:
                logger.error(f"\n❌ FAIL: Not statistically significant (p={p_value:.6f} >= 0.05)")
            if not well_separated:
                logger.error(f"\n❌ FAIL: Distributions not well separated ({separation:.3f} < 0.1)")
        
        self.results.add_result(
            "lateralization_index",
            passed,
            metrics,
            "p < 0.05, separation > 0.1"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 6: DECISION ENGINE VALIDATION
    # =============================================================================
    
    def validate_decision_stability(self, duration: float = 300.0) -> bool:
        """
        6.2 Decision Stability Test
        
        Runs continuous processing and measures decision stability.
        
        Args:
            duration: Test duration in seconds (5 minutes)
        
        Returns:
            bool: True if decisions are stable
        """
        logger.info("="*70)
        logger.info("TEST 6: Decision Engine Stability")
        logger.info("="*70)
        logger.info(f"Running for {duration/60:.1f} minutes")
        logger.info("Perform natural attention shifts during this time")
        
        if not self.acquisition.is_connected:
            if not self.acquisition.connect():
                logger.error("Cannot run stability test - no connection")
                self.results.add_result(
                    "decision_stability",
                    False,
                    {'error': 'no_connection'},
                    "Connection required"
                )
                return False
        
        # Use real-time engine
        engine = RealtimeEEGEngine(self.config)
        engine.connect_and_start()
        
        decisions = []
        timestamps = []
        
        start_time = time.time()
        last_decision = None
        rapid_flips = 0
        
        while time.time() - start_time < duration:
            # Simplified processing loop
            ts, left_vals, right_vals = engine.acquisition.read_batch(max_samples=50)
            
            if len(left_vals) > 0:
                engine.left_buffer.extend(left_vals)
                engine.right_buffer.extend(right_vals)
                
                if len(engine.left_buffer) >= engine.config.window_samples:
                    left_win = np.array(list(engine.left_buffer)[-engine.config.window_samples:])
                    right_win = np.array(list(engine.right_buffer)[-engine.config.window_samples:])
                    
                    result = engine.process_window(left_win, right_win)
                    
                    if result is not None and result['smoothed_direction'] != 'UNKNOWN':
                        current_time = time.time()
                        current_decision = result['smoothed_direction']
                        
                        decisions.append(current_decision)
                        timestamps.append(current_time)
                        
                        # Check for rapid flip
                        if last_decision is not None and len(timestamps) > 1:
                            time_since_last = current_time - timestamps[-2]
                            if last_decision != current_decision and time_since_last < 1.0:
                                rapid_flips += 1
                        
                        last_decision = current_decision
            
            time.sleep(0.01)
        
        engine.stop()
        
        if len(decisions) < 2:
            logger.error("Insufficient decisions made")
            self.results.add_result(
                "decision_stability",
                False,
                {'decisions': len(decisions)},
                "Need at least 2 decisions"
            )
            return False
        
        # Compute metrics
        total_decisions = len(decisions)
        flip_rate = (rapid_flips / total_decisions * 100) if total_decisions > 0 else 0
        
        # Count decision types
        from collections import Counter
        decision_counts = Counter(decisions)
        
        metrics = {
            'total_decisions': total_decisions,
            'rapid_flips': rapid_flips,
            'flip_rate_percent': float(flip_rate),
            'left_count': decision_counts.get('LEFT', 0),
            'right_count': decision_counts.get('RIGHT', 0),
            'neutral_count': decision_counts.get('NEUTRAL', 0)
        }
        
        logger.info(f"\nTotal decisions: {total_decisions}")
        logger.info(f"Rapid flips (<1s): {rapid_flips}")
        logger.info(f"Flip rate: {flip_rate:.2f}%")
        logger.info(f"LEFT: {metrics['left_count']}, RIGHT: {metrics['right_count']}, NEUTRAL: {metrics['neutral_count']}")
        
        # Pass if flip rate < 5%
        passed = flip_rate < 5.0
        
        if passed:
            logger.info("✅ PASS: Decision engine stable")
        else:
            logger.error(f"❌ FAIL: Too many rapid flips ({flip_rate:.2f}% >= 5%)")
        
        self.results.add_result(
            "decision_stability",
            passed,
            metrics,
            "Flip rate < 5%"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 7: REAL-TIME PERFORMANCE TEST
    # =============================================================================
    
    def validate_realtime_performance(self, duration: float = 600.0) -> bool:
        """
        7.1-7.2 Real-time Performance and Memory Test
        
        Validates processing time and memory stability.
        
        Args:
            duration: Test duration (10 minutes)
        
        Returns:
            bool: True if performance meets requirements
        """
        logger.info("="*70)
        logger.info("TEST 7: Real-time Performance")
        logger.info("="*70)
        logger.info(f"Running for {duration/60:.0f} minutes")
        
        if not self.acquisition.is_connected:
            if not self.acquisition.connect():
                logger.error("Cannot run performance test - no connection")
                self.results.add_result(
                    "realtime_performance",
                    False,
                    {'error': 'no_connection'},
                    "Connection required"
                )
                return False
        
        processing_times = []
        
        engine = RealtimeEEGEngine(self.config)
        engine.connect_and_start()
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Measure processing time
            proc_start = time.time()
            
            ts, left_vals, right_vals = engine.acquisition.read_batch(max_samples=50)
            
            if len(left_vals) > 0:
                engine.left_buffer.extend(left_vals)
                engine.right_buffer.extend(right_vals)
                
                if len(engine.left_buffer) >= engine.config.window_samples:
                    left_win = np.array(list(engine.left_buffer)[-engine.config.window_samples:])
                    right_win = np.array(list(engine.right_buffer)[-engine.config.window_samples:])
                    
                    result = engine.process_window(left_win, right_win)
                    
                    proc_time = (time.time() - proc_start) * 1000  # ms
                    processing_times.append(proc_time)
            
            time.sleep(0.01)
        
        engine.stop()
        
        if len(processing_times) < 10:
            logger.error("Insufficient processing time samples")
            self.results.add_result(
                "realtime_performance",
                False,
                {'samples': len(processing_times)},
                "Need at least 10 samples"
            )
            return False
        
        processing_times = np.array(processing_times)
        
        mean_time = np.mean(processing_times)
        max_time = np.max(processing_times)
        p95_time = np.percentile(processing_times, 95)
        
        metrics = {
            'samples': len(processing_times),
            'mean_processing_time_ms': float(mean_time),
            'max_processing_time_ms': float(max_time),
            'p95_processing_time_ms': float(p95_time)
        }
        
        logger.info(f"\nProcessing time statistics:")
        logger.info(f"  Mean: {mean_time:.2f} ms")
        logger.info(f"  Max: {max_time:.2f} ms")
        logger.info(f"  95th percentile: {p95_time:.2f} ms")
        
        # Pass if mean < 50ms and p95 < 100ms
        passed = mean_time < 50.0 and p95_time < 100.0
        
        if passed:
            logger.info("✅ PASS: Real-time performance adequate")
        else:
            logger.error(f"❌ FAIL: Processing too slow (mean={mean_time:.1f}ms, p95={p95_time:.1f}ms)")
        
        self.results.add_result(
            "realtime_performance",
            passed,
            metrics,
            "Mean < 50ms, P95 < 100ms"
        )
        
        return passed
    
    # =============================================================================
    # SECTION 8: FULL SYSTEM REPORT
    # =============================================================================
    
    def generate_report(self):
        """Generate and export comprehensive validation report."""
        logger.info("\n" + "="*70)
        logger.info("FINAL VALIDATION REPORT")
        logger.info("="*70)
        
        summary = self.results.get_summary()
        
        logger.info(f"\nTotal tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['tests_passed']}")
        logger.info(f"Failed: {summary['tests_failed']}")
        logger.info(f"Pass rate: {summary['pass_rate_percent']:.1f}%")
        
        logger.info("\n" + "-"*70)
        logger.info("INDIVIDUAL TEST RESULTS")
        logger.info("-"*70)
        
        for test_name, data in summary['individual_results'].items():
            status = "✅ PASS" if data['passed'] else "❌ FAIL"
            logger.info(f"{test_name:30s} {status}")
        
        # Final pass criteria
        logger.info("\n" + "="*70)
        logger.info("STAGE 1 COMPLETION CRITERIA")
        logger.info("="*70)
        
        criteria_met = summary['pass_rate_percent'] >= 80.0
        
        if criteria_met:
            logger.info("✅ SYSTEM READY FOR DEPLOYMENT")
        else:
            logger.error("❌ SYSTEM REQUIRES FURTHER CALIBRATION")
        
        # Export reports
        self.results.export_json()
        self.results.export_csv()
        
        return criteria_met
    
    def run_full_validation(self, quick_mode: bool = False):
        """
        Run complete validation suite.
        
        Args:
            quick_mode: If True, uses shorter test durations
        """
        logger.info("\n" + "="*70)
        logger.info("STARTING FULL EEG SYSTEM VALIDATION")
        logger.info("="*70)
        
        if quick_mode:
            logger.info("QUICK MODE: Using reduced test durations")
        
        # Connect
        if not self.acquisition.connect():
            logger.error("Failed to establish connection. Cannot proceed.")
            return False
        
        # Section 1: Data Acquisition
        self.validate_sampling_rate(duration=60.0 if not quick_mode else 30.0)
        self.validate_adc_scaling(duration=10.0)
        self.validate_serial_integrity(duration=60.0 if not quick_mode else 30.0)
        
        # Section 2: Filters
        self.validate_frequency_response()
        self.validate_filtering_effect(duration=30.0 if not quick_mode else 15.0)
        
        # Section 3: Features
        self.validate_alpha_power_stability(duration=60.0 if not quick_mode else 30.0)
        
        # Section 4: Artifacts
        self.validate_artifact_detection(duration=30.0 if not quick_mode else 20.0)
        
        # Section 5: Lateralization (interactive)
        if not quick_mode:
            logger.info("\n⚠️  Next test requires user interaction")
            response = input("Run controlled attention experiment? (y/n): ")
            if response.lower() == 'y':
                self.validate_lateralization_index(trials=5 if not quick_mode else 2)
        
        # Section 6: Decision stability
        if not quick_mode:
            self.validate_decision_stability(duration=300.0 if not quick_mode else 60.0)
        
        # Section 7: Performance
        self.validate_realtime_performance(duration=600.0 if not quick_mode else 120.0)
        
        # Generate report
        all_passed = self.generate_report()
        
        # Cleanup
        self.acquisition.disconnect()
        
        return all_passed


def main():
    """Main entry point for validation."""
    import sys
    
    logger.info("="*70)
    logger.info("EEG SYSTEM VALIDATION & VERIFICATION")
    logger.info("Industry-Grade Testing Protocol")
    logger.info("="*70)
    
    quick_mode = '--quick' in sys.argv
    
    if quick_mode:
        logger.info("\n🚀 QUICK MODE: Reduced test durations")
    
    validator = EEGValidator()
    
    try:
        success = validator.run_full_validation(quick_mode=quick_mode)
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("✅ VALIDATION COMPLETE - SYSTEM PASSED")
            logger.info("="*70)
            return 0
        else:
            logger.info("\n" + "="*70)
            logger.error("❌ VALIDATION FAILED - REVIEW REQUIRED")
            logger.info("="*70)
            return 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Validation interrupted by user")
        validator.generate_report()
        return 2
    
    except Exception as e:
        logger.error(f"\n❌ Validation error: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    exit(main())
