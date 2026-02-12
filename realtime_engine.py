"""
Real-time EEG Processing Engine.

Main processing loop integrating all modules:
- Continuous data acquisition from serial port
- Signal preprocessing and filtering
- Feature extraction
- Quality assessment
- Attention direction detection
- Data logging
- Real-time visualization (optional)
"""

import numpy as np
import time
import csv
import logging
from pathlib import Path
from typing import Optional, Dict
from collections import deque
import sys

from config import SystemConfig, get_config
from acquisition import SerialAcquisition
from filters import SignalFilter
from features import FeatureExtractor
from metrics import SignalQualityMetrics
from decision import DecisionEngine, AttentionDirection


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeEEGEngine:
    """
    Main real-time EEG processing engine.
    
    Coordinates all processing components and implements the main processing loop.
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize real-time EEG processing engine.
        
        Args:
            config: System configuration. If None, uses global config.
        """
        if config is None:
            config = get_config()
        
        self.config = config
        
        # Initialize all components
        logger.info("Initializing EEG processing components...")
        
        self.acquisition = SerialAcquisition(config.serial)
        self.filter = SignalFilter(config.signal)
        self.feature_extractor = FeatureExtractor(config.signal)
        self.quality_metrics = SignalQualityMetrics(config.artifact, config.signal)
        self.decision_engine = DecisionEngine(config.decision)
        
        # Data buffers for sliding window processing
        self.left_buffer = deque(maxlen=config.window_samples * 2)
        self.right_buffer = deque(maxlen=config.window_samples * 2)
        
        # Processing state
        self.is_running = False
        self.samples_processed = 0
        self.windows_processed = 0
        
        # CSV logging
        self.csv_file = None
        self.csv_writer = None
        
        # Performance monitoring
        self.processing_times = deque(maxlen=100)
        self.last_status_time = time.time()
        
        logger.info("RealtimeEEGEngine initialized successfully")
    
    def connect_and_start(self) -> bool:
        """
        Connect to Arduino and start acquisition.
        
        Returns:
            bool: True if connection successful
        """
        logger.info("Connecting to Arduino...")
        
        if not self.acquisition.connect():
            logger.error("Failed to connect to Arduino")
            return False
        
        logger.info("Connection established successfully")
        
        # Initialize CSV logging if enabled
        if self.config.logging.enable_csv_logging:
            self._init_csv_logging()
        
        return True
    
    def _init_csv_logging(self) -> None:
        """Initialize CSV logging file."""
        try:
            csv_path = Path(self.config.logging.csv_filename)
            self.csv_file = open(csv_path, 'w', newline='')
            
            # Define CSV columns
            fieldnames = [
                'timestamp',
                'sample_count',
                'left_alpha_power',
                'right_alpha_power',
                'lateralization_index',
                'attention_direction',
                'confidence',
                'smoothed_direction',
                'quality_score',
                'left_snr_db',
                'right_snr_db',
                'left_artifact',
                'right_artifact'
            ]
            
            self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
            self.csv_writer.writeheader()
            self.csv_file.flush()
            
            logger.info(f"CSV logging initialized: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize CSV logging: {e}")
            self.csv_writer = None
    
    def _log_to_csv(self, data: Dict) -> None:
        """
        Log processing results to CSV file.
        
        Args:
            data: Dictionary containing data to log
        """
        if self.csv_writer is not None:
            try:
                self.csv_writer.writerow(data)
                
                # Flush periodically
                if self.samples_processed % 10 == 0:
                    self.csv_file.flush()
                    
            except Exception as e:
                logger.error(f"CSV logging error: {e}")
    
    def process_window(
        self,
        left_signal: np.ndarray,
        right_signal: np.ndarray
    ) -> Optional[Dict]:
        """
        Process one window of EEG data.
        
        Args:
            left_signal: Raw EEG data from left hemisphere
            right_signal: Raw EEG data from right hemisphere
        
        Returns:
            dict: Processing results or None if processing failed
        """
        try:
            processing_start = time.time()
            
            # Step 1: Preprocessing (notch + bandpass filtering)
            left_filtered = self.filter.preprocess_signal(left_signal)
            right_filtered = self.filter.preprocess_signal(right_signal)
            
            # Step 2: Quality assessment
            quality_score = self.quality_metrics.compute_quality_score(
                left_filtered, right_filtered
            )
            
            left_quality = self.quality_metrics.compute_channel_quality(left_filtered)
            right_quality = self.quality_metrics.compute_channel_quality(right_filtered)
            
            # Step 3: Feature extraction
            features = self.feature_extractor.extract_minimal_features(
                left_filtered, right_filtered
            )
            
            left_alpha_power = features['left_alpha_power']
            right_alpha_power = features['right_alpha_power']
            
            # Step 4: Artifact rejection
            if left_quality['has_artifact'] or right_quality['has_artifact']:
                logger.debug(
                    f"Artifact detected: "
                    f"L={left_quality['artifact_type']}, "
                    f"R={right_quality['artifact_type']}"
                )
                
                # Return result with artifact flag
                result = {
                    'timestamp': time.time(),
                    'sample_count': self.samples_processed,
                    'left_alpha_power': left_alpha_power,
                    'right_alpha_power': right_alpha_power,
                    'lateralization_index': 0.0,
                    'attention_direction': AttentionDirection.UNKNOWN.value,
                    'confidence': 0.0,
                    'smoothed_direction': AttentionDirection.UNKNOWN.value,
                    'quality_score': quality_score,
                    'left_snr_db': left_quality['snr_db'],
                    'right_snr_db': right_quality['snr_db'],
                    'left_artifact': True,
                    'right_artifact': True,
                    'processing_time_ms': 0.0
                }
                
                return result
            
            # Step 5: Decision making
            attention, li, confidence = self.decision_engine.make_decision(
                left_alpha_power, right_alpha_power
            )
            
            # Step 6: Smoothed decision
            smoothed_attention, smoothed_confidence = self.decision_engine.get_smoothed_decision()
            
            # Calculate processing time
            processing_time = (time.time() - processing_start) * 1000  # ms
            self.processing_times.append(processing_time)
            
            # Compile results
            result = {
                'timestamp': time.time(),
                'sample_count': self.samples_processed,
                'left_alpha_power': left_alpha_power,
                'right_alpha_power': right_alpha_power,
                'lateralization_index': li,
                'attention_direction': attention.value,
                'confidence': confidence,
                'smoothed_direction': smoothed_attention.value,
                'quality_score': quality_score,
                'left_snr_db': left_quality['snr_db'],
                'right_snr_db': right_quality['snr_db'],
                'left_artifact': left_quality['has_artifact'],
                'right_artifact': right_quality['has_artifact'],
                'processing_time_ms': processing_time
            }
            
            self.windows_processed += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Window processing error: {e}")
            return None
    
    def run(self, duration: Optional[float] = None) -> None:
        """
        Run real-time processing loop.
        
        Args:
            duration: Processing duration in seconds. If None, runs indefinitely.
        """
        logger.info("Starting real-time EEG processing...")
        
        if not self.acquisition.is_connected:
            if not self.connect_and_start():
                logger.error("Cannot start without connection")
                return
        
        self.is_running = True
        start_time = time.time()
        
        try:
            while self.is_running:
                # Check duration limit
                if duration is not None:
                    if time.time() - start_time >= duration:
                        logger.info(f"Duration limit ({duration}s) reached")
                        break
                
                # Read batch of samples
                timestamps, left_values, right_values = self.acquisition.read_batch(
                    max_samples=50
                )
                
                if len(left_values) == 0:
                    # No data available, check connection
                    if not self.acquisition.is_connected:
                        logger.warning("Connection lost, attempting reconnect...")
                        if self.acquisition.reconnect():
                            logger.info("Reconnected successfully")
                        else:
                            time.sleep(0.1)
                    else:
                        time.sleep(0.001)  # Brief sleep to avoid busy-waiting
                    continue
                
                # Add to buffers
                self.left_buffer.extend(left_values)
                self.right_buffer.extend(right_values)
                self.samples_processed += len(left_values)
                
                # Check if we have enough data for a window
                if len(self.left_buffer) >= self.config.window_samples:
                    # Extract window
                    left_window = np.array(list(self.left_buffer)[-self.config.window_samples:])
                    right_window = np.array(list(self.right_buffer)[-self.config.window_samples:])
                    
                    # Process window
                    result = self.process_window(left_window, right_window)
                    
                    if result is not None:
                        # Log to console
                        self._display_result(result)
                        
                        # Log to CSV
                        if self.config.logging.enable_csv_logging:
                            self._log_to_csv(result)
                    
                    # Remove processed samples (hop size)
                    for _ in range(self.config.hop_samples):
                        if len(self.left_buffer) > 0:
                            self.left_buffer.popleft()
                        if len(self.right_buffer) > 0:
                            self.right_buffer.popleft()
                
                # Display status periodically
                if time.time() - self.last_status_time >= 5.0:
                    self._display_status()
                    self.last_status_time = time.time()
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Runtime error: {e}", exc_info=True)
        
        finally:
            self.stop()
    
    def _display_result(self, result: Dict) -> None:
        """
        Display processing result to console.
        
        Args:
            result: Processing result dictionary
        """
        # Only display if smoothed direction is not UNKNOWN
        if result['smoothed_direction'] != AttentionDirection.UNKNOWN.value:
            direction_symbol = {
                'LEFT': '←',
                'RIGHT': '→',
                'NEUTRAL': '•'
            }.get(result['smoothed_direction'], '?')
            
            logger.info(
                f"Attention: {direction_symbol} {result['smoothed_direction']} | "
                f"LI: {result['lateralization_index']:+.3f} | "
                f"Conf: {result['confidence']:.2f} | "
                f"Quality: {result['quality_score']:.1f}/100 | "
                f"L-Alpha: {result['left_alpha_power']:.2f} | "
                f"R-Alpha: {result['right_alpha_power']:.2f}"
            )
    
    def _display_status(self) -> None:
        """Display system status."""
        # Acquisition statistics
        acq_stats = self.acquisition.get_statistics()
        
        # Decision statistics
        dec_stats = self.decision_engine.get_statistics()
        
        # Calibration status
        cal_status = self.decision_engine.get_calibration_status()
        
        # Processing performance
        if len(self.processing_times) > 0:
            avg_processing_time = np.mean(self.processing_times)
            max_processing_time = np.max(self.processing_times)
        else:
            avg_processing_time = 0.0
            max_processing_time = 0.0
        
        # Fix calibration status string to avoid nested f-string
        if cal_status['is_calibrated']:
            cal_status_str = '✓'
        else:
            cal_status_str = f"{cal_status['calibration_samples']}/{cal_status['required_samples']}"
        
        logger.info(
            f"\n{'='*70}\n"
            f"STATUS - Samples: {self.samples_processed} | Windows: {self.windows_processed}\n"
            f"Connection: {'✓' if acq_stats['connected'] else '✗'} | "
            f"Corruption: {acq_stats['corruption_rate_percent']:.1f}%\n"
            f"Calibration: {cal_status_str}\n"
            f"Processing: {avg_processing_time:.1f}ms avg, {max_processing_time:.1f}ms max\n"
            f"Decisions: L={dec_stats.get('left_decisions', 0)} | "
            f"R={dec_stats.get('right_decisions', 0)} | "
            f"N={dec_stats.get('neutral_decisions', 0)}\n"
            f"{'='*70}"
        )
    
    def stop(self) -> None:
        """Stop processing and cleanup."""
        logger.info("Stopping EEG processing engine...")
        
        self.is_running = False
        
        # Close acquisition
        self.acquisition.disconnect()
        
        # Close CSV file
        if self.csv_file is not None:
            try:
                self.csv_file.close()
                logger.info("CSV file closed")
            except Exception as e:
                logger.error(f"Error closing CSV file: {e}")
        
        # Display final statistics
        logger.info(
            f"\n{'='*70}\n"
            f"FINAL STATISTICS\n"
            f"Total samples processed: {self.samples_processed}\n"
            f"Total windows processed: {self.windows_processed}\n"
            f"{'='*70}"
        )
        
        logger.info("Engine stopped successfully")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect_and_start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def main():
    """
    Main entry point for standalone execution.
    """
    logger.info("="*70)
    logger.info("Real-time EEG Signal Processing & Attention Decoder")
    logger.info("="*70)
    
    # Load configuration
    config = get_config()
    
    # Allow command line configuration override
    if len(sys.argv) > 1:
        port = sys.argv[1]
        config.serial.port = port
        logger.info(f"Using serial port: {port}")
    
    # Create and run engine
    engine = RealtimeEEGEngine(config)
    
    try:
        # Run indefinitely (Ctrl+C to stop)
        engine.run(duration=None)
    
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    
    finally:
        engine.stop()
    
    logger.info("Program terminated")


if __name__ == "__main__":
    main()
