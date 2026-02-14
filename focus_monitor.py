"""
Single-Channel Focus Detection using Alpha Suppression.

Real-time focus monitoring based on left channel EEG only.
Uses alpha power suppression relative to baseline.
"""

import numpy as np
import time
import logging
from collections import deque

from config import get_config
from acquisition import SerialAcquisition
from filters import SignalFilter
from features import FeatureExtractor
from metrics import SignalQualityMetrics
from decision import FocusDetectionEngine, FocusState


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FocusMonitor:
    """Real-time focus monitoring system."""
    
    def __init__(self):
        """Initialize focus monitor."""
        self.config = get_config()
        
        # Initialize components
        self.acquisition = SerialAcquisition(self.config.serial)
        self.filter = SignalFilter(self.config.signal)
        self.feature_extractor = FeatureExtractor(self.config.signal)
        self.quality_metrics = SignalQualityMetrics(self.config.artifact, self.config.signal)
        self.focus_engine = FocusDetectionEngine(self.config.decision)
        
        # Data buffer (left channel only)
        window_samples = int(self.config.signal.window_size * self.config.signal.sampling_rate)
        self.left_buffer = deque(maxlen=window_samples * 2)
        
        logger.info("FocusMonitor initialized")
    
    def calibrate(self, duration=10.0):
        """
        Baseline calibration phase.
        
        User should relax with eyes closed.
        
        Args:
            duration: Calibration duration in seconds
        """
        print("\n" + "="*70)
        print("BASELINE CALIBRATION")
        print("="*70)
        print(f"\n📋 Instructions:")
        print(f"   • Sit comfortably")
        print(f"   • CLOSE YOUR EYES")
        print(f"   • Relax for {duration:.0f} seconds")
        print(f"   • Don't move or think intensely")
        print(f"\nStarting in 3 seconds...")
        time.sleep(3)
        print("\n🔴 Recording baseline... (eyes closed, relaxed)")
        
        alpha_values = []
        quality_scores = []
        
        start_time = time.time()
        window_size = int(self.config.signal.window_size * self.config.signal.sampling_rate)
        
        while time.time() - start_time < duration:
            # Read sample
            data = self.acquisition.read_sample()
            if data is None:
                time.sleep(0.001)
                continue
            
            timestamp, left_value, right_value = data
            self.left_buffer.append(left_value)
            
            # Process when buffer fills
            if len(self.left_buffer) >= window_size:
                # Get window
                left_signal = np.array(list(self.left_buffer)[-window_size:])
                
                # Quality check on RAW signal (before filtering)
                quality_score = self.quality_metrics.compute_quality_score(left_signal, left_signal)
                
                # Remove DC
                left_signal = left_signal - np.mean(left_signal)
                
                # Apply filters
                left_filtered = self.filter.apply_notch_filter(left_signal)
                left_filtered = self.filter.apply_bandpass_filter(left_filtered)
                
                # Extract alpha power
                alpha_power = self.filter.compute_alpha_power(left_filtered)
                
                alpha_values.append(alpha_power)
                quality_scores.append(quality_score)
                
                # Progress
                elapsed = time.time() - start_time
                progress = (elapsed / duration) * 100
                print(f"  Progress: {progress:.0f}% | Alpha: {alpha_power:.2f} | Quality: {quality_score:.0f}", end='\r')
        
        print("\n\n✓ Calibration complete!")
        
        # Calibrate engine
        success = self.focus_engine.calibrate_baseline(alpha_values, quality_scores)
        
        if success:
            print(f"✓ Baseline: {self.focus_engine.baseline_alpha:.2f} ± {self.focus_engine.baseline_std:.2f}")
            print(f"✓ Used {len(alpha_values)} windows")
        else:
            print("✗ Calibration failed!")
            return False
        
        print("\n" + "="*70)
        return True
    
    def run(self, duration=60.0):
        """
        Run focus monitoring.
        
        Args:
            duration: Monitoring duration in seconds (0 = infinite)
        """
        if not self.focus_engine.is_calibrated:
            print("⚠ Engine not calibrated! Run calibration first.")
            return
        
        print("\n" + "="*70)
        print("FOCUS MONITORING")
        print("="*70)
        print(f"\nMonitoring for {duration:.0f} seconds...")
        print("Try: eyes open → focus on mental task → relax\n")
        
        start_time = time.time()
        window_size = int(self.config.signal.window_size * self.config.signal.sampling_rate)
        window_count = 0
        
        while True:
            if duration> 0 and (time.time() - start_time) > duration:
                break
            
            # Read sample
            data = self.acquisition.read_sample()
            if data is None:
                time.sleep(0.001)
                continue
            
            timestamp, left_value, right_value = data
            self.left_buffer.append(left_value)
            
            # Process window
            if len(self.left_buffer) >= window_size:
                window_count += 1
                
                # Get window
                left_signal = np.array(list(self.left_buffer)[-window_size:])
                
                # Quality assessment on RAW signal (before filtering)
                quality_score = self.quality_metrics.compute_quality_score(left_signal, left_signal)
                
                # Remove DC
                left_signal = left_signal - np.mean(left_signal)
                
                # Apply filters
                left_filtered = self.filter.apply_notch_filter(left_signal)
                left_filtered = self.filter.apply_bandpass_filter(left_filtered)
                
                # Extract alpha power
                alpha_power = self.filter.compute_alpha_power(left_filtered)
                
                # Compute suppression ratio
                suppression_ratio = self.focus_engine.compute_suppression_ratio(alpha_power)
                
                # Additional quality metrics on filtered signal
                snr_db = self.quality_metrics.compute_snr(left_filtered)
                has_artifact = False  # Disable artifact detection for testing
                
                # Classify focus state
                state = self.focus_engine.classify_focus(
                    suppression_ratio,
                    quality_score,
                    snr_db,
                    has_artifact
                )
                
                # Compute confidence
                confidence = self.focus_engine.compute_confidence(suppression_ratio)
                
                # Smooth state
                smoothed_state, agreement = self.focus_engine.get_smoothed_state()
                
                # Display (only every window)
                if window_count % 1 == 0:  # Display every window
                    self._print_status(
                        alpha_power,
                        self.focus_engine.baseline_alpha,
                        suppression_ratio,
                        state,
                        confidence,
                        quality_score,
                        snr_db
                    )
        
        print("\n" + "="*70)
        print("MONITORING COMPLETE")
        print("="*70)
        
        # Print statistics
        stats = self.focus_engine.get_statistics()
        print(f"\nTotal Decisions: {stats.get('total_decisions', 0)}")
        print(f"  Focused: {stats.get('focused_count', 0)} ({stats.get('focused_percent', 0):.1f}%)")
        print(f"  Relaxed: {stats.get('relaxed_count', 0)} ({stats.get('relaxed_percent', 0):.1f}%)")
        print(f"  Neutral: {stats.get('neutral_count', 0)} ({stats.get('neutral_percent', 0):.1f}%)")
        
        if 'avg_suppression' in stats:
            print(f"\nAverage Suppression Ratio: {stats['avg_suppression']:.3f}")
            print(f"Range: {stats['min_suppression']:.3f} - {stats['max_suppression']:.3f}")
    
    def _print_status(self, alpha, baseline, ratio, state, confidence, quality, snr):
        """Print current status."""
        # State symbol
        if state == FocusState.FOCUSED:
            symbol = "🎯 FOCUSED"
        elif state == FocusState.RELAXED:
            symbol = "😌 RELAXED"
        elif state == FocusState.NEUTRAL:
            symbol = "➖ NEUTRAL"
        elif state == FocusState.UNRELIABLE:
            symbol = "⚠️  UNRELIABLE"
        else:
            symbol = "❓ UNCALIBRATED"
        
        print(
            f"Alpha: {alpha:6.2f} | "
            f"Baseline: {baseline:6.2f} | "
            f"Ratio: {ratio:5.2f} | "
            f"State: {symbol:15s} | "
            f"Conf: {confidence*100:3.0f}% | "
            f"Quality: {quality:3.0f} | "
            f"SNR: {snr:5.1f}dB"
        )
    
    def verify_focus(self):
        """Run focus verification protocol."""
        print("\n" + "="*70)
        print("FOCUS VERIFICATION PROTOCOL")
        print("="*70)
        
        # Phase 1: Relaxed
        print("\nPhase 1: RELAXED (eyes closed)")
        print("Starting in 3 seconds...")
        time.sleep(3)
        print("🔴 Recording... (10 seconds)")
        
        relaxed_alpha = []
        start_time = time.time()
        window_size = int(self.config.signal.window_size * self.config.signal.sampling_rate)
        
        while time.time() - start_time < 10.0:
            data = self.acquisition.read_sample()
            if data is None:
                time.sleep(0.001)
                continue
            
            timestamp, left_value, right_value = data
            self.left_buffer.append(left_value)
            
            if len(self.left_buffer) >= window_size:
                left_signal = np.array(list(self.left_buffer)[-window_size:])
                left_signal = left_signal - np.mean(left_signal)
                left_filtered = self.filter.apply_notch_filter(left_signal)
                left_filtered = self.filter.apply_bandpass_filter(left_filtered)
                alpha_power = self.filter.compute_alpha_power(left_filtered)
                relaxed_alpha.append(alpha_power)
        
        print("✓ Phase 1 complete")
        
        # Phase 2: Focused
        print("\nPhase 2: FOCUSED (mental math: 17 x 23 = ?)")
        print("Starting in 3 seconds...")
        time.sleep(3)
        print("🔴 Recording... (10 seconds)")
        
        focused_alpha = []
        start_time = time.time()
        
        while time.time() - start_time < 10.0:
            data = self.acquisition.read_sample()
            if data is None:
                time.sleep(0.001)
                continue
            
            timestamp, left_value, right_value = data
            self.left_buffer.append(left_value)
            
            if len(self.left_buffer) >= window_size:
                left_signal = np.array(list(self.left_buffer)[-window_size:])
                left_signal = left_signal - np.mean(left_signal)
                left_filtered = self.filter.apply_notch_filter(left_signal)
                left_filtered = self.filter.apply_bandpass_filter(left_filtered)
                alpha_power = self.filter.compute_alpha_power(left_filtered)
                focused_alpha.append(alpha_power)
        
        print("✓ Phase 2 complete")
        
        # Verification
        from decision import verify_focus_protocol
        results = verify_focus_protocol(self.focus_engine, relaxed_alpha, focused_alpha)
        
        print("\n" + "="*70)
        print("VERIFICATION RESULTS")
        print("="*70)
        
        if 'error' in results:
            print(f"✗ Error: {results['error']}")
        else:
            print(f"\nRelaxed Phase:")
            print(f"  Mean Ratio: {results['relaxed_phase']['mean_ratio']:.3f}")
            print(f"  Samples: {results['relaxed_phase']['n_samples']}")
            
            print(f"\nFocused Phase:")
            print(f"  Mean Ratio: {results['focused_phase']['mean_ratio']:.3f}")
            print(f"  Samples: {results['focused_phase']['n_samples']}")
            
            print(f"\nDifference: {results['difference']:.3f}")
            print(f"Suppression: {results['percent_suppression']:.1f}%")
            
            if results['detection_valid']:
                print("\n✓ Focus detection VALID (>15% difference)")
            else:
                print("\n⚠ Focus detection questionable (<15% difference)")


def main():
    """Main entry point."""
    monitor = FocusMonitor()
    
    # Connect
    print("\n🔌 Connecting to Arduino...")
    if not monitor.acquisition.connect():
        print("✗ Connection failed!")
        return
    
    print("✓ Connected")
    
    try:
        # Calibrate
        if not monitor.calibrate(duration=10.0):
            return
        
        # Run monitoring
        monitor.run(duration=60.0)
        
        # Optional: Verification
        print("\n\nRun verification protocol? (y/n): ", end='')
        # Skip input for now, comment out if needed
        # response = input().strip().lower()
        # if response == 'y':
        #     monitor.verify_focus()
        
    except KeyboardInterrupt:
        print("\n\n⚠ Stopped by user")
    finally:
        print("\n✓ Shutting down...")


if __name__ == "__main__":
    main()
