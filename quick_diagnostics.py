"""
Quick Diagnostics and Benchmarking Script.

Performs rapid system checks before running full validation.
Useful for identifying obvious issues quickly.
"""

import numpy as np
import time
import logging
from typing import Dict

from config import get_config
from acquisition import SerialAcquisition
from filters import SignalFilter
from features import FeatureExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuickDiagnostics:
    """Quick diagnostic checks for EEG system."""
    
    def __init__(self):
        """Initialize diagnostics."""
        self.config = get_config()
        self.acquisition = SerialAcquisition()
        self.filter = SignalFilter()
        self.feature_extractor = FeatureExtractor()
    
    def check_connection(self) -> bool:
        """Quick connection check."""
        logger.info("="*70)
        logger.info("CHECK 1: Connection")
        logger.info("="*70)
        
        if self.acquisition.connect():
            logger.info("✅ Connected successfully")
            
            # Try to read a few samples
            logger.info("Testing data reception...")
            samples_received = 0
            start_time = time.time()
            
            while time.time() - start_time < 3.0:
                if self.acquisition.read_sample() is not None:
                    samples_received += 1
                time.sleep(0.01)
            
            logger.info(f"Received {samples_received} samples in 3 seconds")
            
            if samples_received > 0:
                logger.info(f"✅ Data reception OK ({samples_received/3:.0f} samples/sec)")
                return True
            else:
                logger.error("❌ No data received")
                return False
        else:
            logger.error("❌ Connection failed")
            return False
    
    def check_signal_quality(self, duration: float = 10.0) -> Dict:
        """Quick signal quality check."""
        logger.info("\n" + "="*70)
        logger.info("CHECK 2: Signal Quality")
        logger.info("="*70)
        logger.info(f"Sampling for {duration} seconds...")
        
        if not self.acquisition.is_connected:
            logger.error("Not connected")
            return {'error': 'no_connection'}
        
        left_values = []
        right_values = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_values.append(left)
                right_values.append(right)
            time.sleep(0.001)
        
        if len(left_values) < 10:
            logger.error("Insufficient data")
            return {'error': 'insufficient_data'}
        
        left_arr = np.array(left_values)
        right_arr = np.array(right_values)
        
        results = {
            'samples': len(left_values),
            'effective_fs': len(left_values) / duration,
            'left_mean': float(np.mean(left_arr)),
            'left_std': float(np.std(left_arr)),
            'left_range': float(np.ptp(left_arr)),
            'right_mean': float(np.mean(right_arr)),
            'right_std': float(np.std(right_arr)),
            'right_range': float(np.ptp(right_arr))
        }
        
        logger.info(f"Samples collected: {results['samples']}")
        logger.info(f"Effective Fs: {results['effective_fs']:.1f} Hz")
        logger.info(f"\nLeft Channel:")
        logger.info(f"  Mean: {results['left_mean']:.2f}")
        logger.info(f"  Std:  {results['left_std']:.2f}")
        logger.info(f"  Range: {results['left_range']:.2f}")
        logger.info(f"\nRight Channel:")
        logger.info(f"  Mean: {results['right_mean']:.2f}")
        logger.info(f"  Std:  {results['right_std']:.2f}")
        logger.info(f"  Range: {results['right_range']:.2f}")
        
        # Check if signals look reasonable
        fs_ok = abs(results['effective_fs'] - self.config.signal.sampling_rate) < 50
        variance_ok = results['left_std'] > 1.0 and results['right_std'] > 1.0
        range_ok = results['left_range'] > 10 and results['right_range'] > 10
        
        if fs_ok and variance_ok and range_ok:
            logger.info("\n✅ Signal quality appears good")
        else:
            logger.warning("\n⚠️  Signal quality issues detected:")
            if not fs_ok:
                logger.warning(f"  - Sampling rate discrepancy: {results['effective_fs']:.1f} Hz")
            if not variance_ok:
                logger.warning("  - Low variance (signal may be constant)")
            if not range_ok:
                logger.warning("  - Low dynamic range")
        
        return results
    
    def benchmark_processing(self, iterations: int = 100) -> Dict:
        """Benchmark processing performance."""
        logger.info("\n" + "="*70)
        logger.info("CHECK 3: Processing Performance")
        logger.info("="*70)
        logger.info(f"Running {iterations} iterations...")
        
        # Generate test data
        window_samples = self.config.window_samples
        test_left = np.random.randn(window_samples) * 50 + 512
        test_right = np.random.randn(window_samples) * 50 + 512
        
        # Benchmark filtering
        filter_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.filter.preprocess_signal(test_left)
            filter_times.append((time.perf_counter() - start) * 1000)
        
        # Benchmark feature extraction
        left_filt = self.filter.preprocess_signal(test_left)
        right_filt = self.filter.preprocess_signal(test_right)
        
        feature_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            self.feature_extractor.extract_minimal_features(left_filt, right_filt)
            feature_times.append((time.perf_counter() - start) * 1000)
        
        # Complete pipeline
        pipeline_times = []
        for _ in range(iterations):
            start = time.perf_counter()
            left_f = self.filter.preprocess_signal(test_left)
            right_f = self.filter.preprocess_signal(test_right)
            self.feature_extractor.extract_minimal_features(left_f, right_f)
            pipeline_times.append((time.perf_counter() - start) * 1000)
        
        results = {
            'filter_mean_ms': float(np.mean(filter_times)),
            'filter_max_ms': float(np.max(filter_times)),
            'feature_mean_ms': float(np.mean(feature_times)),
            'feature_max_ms': float(np.max(feature_times)),
            'pipeline_mean_ms': float(np.mean(pipeline_times)),
            'pipeline_max_ms': float(np.max(pipeline_times)),
            'pipeline_p95_ms': float(np.percentile(pipeline_times, 95))
        }
        
        logger.info(f"\nFiltering:")
        logger.info(f"  Mean: {results['filter_mean_ms']:.2f} ms")
        logger.info(f"  Max:  {results['filter_max_ms']:.2f} ms")
        logger.info(f"\nFeature Extraction:")
        logger.info(f"  Mean: {results['feature_mean_ms']:.2f} ms")
        logger.info(f"  Max:  {results['feature_max_ms']:.2f} ms")
        logger.info(f"\nComplete Pipeline:")
        logger.info(f"  Mean: {results['pipeline_mean_ms']:.2f} ms")
        logger.info(f"  Max:  {results['pipeline_max_ms']:.2f} ms")
        logger.info(f"  P95:  {results['pipeline_p95_ms']:.2f} ms")
        
        # Check if fast enough for real-time
        realtime_ok = results['pipeline_mean_ms'] < 50.0 and results['pipeline_p95_ms'] < 100.0
        
        if realtime_ok:
            logger.info("\n✅ Processing fast enough for real-time")
        else:
            logger.warning("\n⚠️  Processing may be too slow for real-time")
        
        return results
    
    def test_alpha_extraction(self, duration: float = 20.0) -> bool:
        """Test alpha band extraction on real data."""
        logger.info("\n" + "="*70)
        logger.info("CHECK 4: Alpha Band Extraction")
        logger.info("="*70)
        logger.info(f"Recording {duration} seconds...")
        
        if not self.acquisition.is_connected:
            logger.error("Not connected")
            return False
        
        # Collect data
        left_values = []
        right_values = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            data = self.acquisition.read_sample()
            if data is not None:
                _, left, right = data
                left_values.append(left)
                right_values.append(right)
            time.sleep(0.001)
        
        if len(left_values) < 100:
            logger.error("Insufficient data")
            return False
        
        left_arr = np.array(left_values)
        right_arr = np.array(right_values)
        
        # Process
        logger.info("Processing...")
        left_filt = self.filter.preprocess_signal(left_arr)
        right_filt = self.filter.preprocess_signal(right_arr)
        
        left_alpha_power, right_alpha_power = self.feature_extractor.extract_alpha_power(
            left_filt, right_filt
        )
        
        logger.info(f"\nAlpha Power:")
        logger.info(f"  Left:  {left_alpha_power:.2f}")
        logger.info(f"  Right: {right_alpha_power:.2f}")
        logger.info(f"  Ratio: {left_alpha_power/right_alpha_power:.2f}" if right_alpha_power > 0 else "  Ratio: N/A")
        
        # Check if reasonable
        powers_reasonable = left_alpha_power > 0.1 and right_alpha_power > 0.1
        powers_balanced = 0.1 < (left_alpha_power/right_alpha_power) < 10.0 if right_alpha_power > 0 else False
        
        if powers_reasonable and powers_balanced:
            logger.info("\n✅ Alpha extraction working")
            return True
        else:
            logger.warning("\n⚠️  Alpha extraction issues:")
            if not powers_reasonable:
                logger.warning("  - Powers too low or zero")
            if not powers_balanced:
                logger.warning("  - Extreme power imbalance")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all quick diagnostic checks."""
        logger.info("\n" + "="*70)
        logger.info("QUICK DIAGNOSTICS - EEG SYSTEM")
        logger.info("="*70)
        
        results = {}
        
        # 1. Connection
        results['connection'] = self.check_connection()
        
        if not results['connection']:
            logger.error("\n❌ Connection failed. Cannot continue.")
            return False
        
        # 2. Signal quality
        signal_results = self.check_signal_quality(10.0)
        results['signal_quality'] = 'error' not in signal_results
        
        # 3. Performance
        perf_results = self.benchmark_processing(100)
        results['performance'] = perf_results['pipeline_mean_ms'] < 50.0
        
        # 4. Alpha extraction
        results['alpha_extraction'] = self.test_alpha_extraction(20.0)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("DIAGNOSTIC SUMMARY")
        logger.info("="*70)
        
        for check, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            logger.info(f"{check:20s} {status}")
        
        all_passed = all(results.values())
        
        if all_passed:
            logger.info("\n✅ All quick checks PASSED")
            logger.info("System ready for full validation")
        else:
            logger.error("\n❌ Some checks FAILED")
            logger.info("Address issues before running full validation")
        
        # Cleanup
        self.acquisition.disconnect()
        
        return all_passed


def main():
    """Main entry point."""
    logger.info("Starting quick diagnostics...\n")
    
    diagnostics = QuickDiagnostics()
    
    try:
        success = diagnostics.run_all_checks()
        
        if success:
            logger.info("\n" + "="*70)
            logger.info("READY FOR FULL VALIDATION")
            logger.info("Run: python validation.py")
            logger.info("Or:  python run_validation.py")
            logger.info("="*70)
            return 0
        else:
            logger.info("\n" + "="*70)
            logger.info("SYSTEM NOT READY")
            logger.info("Fix identified issues before validation")
            logger.info("="*70)
            return 1
    
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Interrupted by user")
        diagnostics.acquisition.disconnect()
        return 2
    
    except Exception as e:
        logger.error(f"\n❌ Diagnostic error: {e}", exc_info=True)
        return 3


if __name__ == "__main__":
    exit(main())
