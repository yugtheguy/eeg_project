"""
Example usage scripts for the EEG Processing System.

Demonstrates various ways to use the real-time EEG processing engine.
"""

import time
import logging
from config import SystemConfig, get_config, update_config
from realtime_engine import RealtimeEEGEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def example_basic():
    """
    Example 1: Basic usage with default configuration.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 1: Basic Usage")
    logger.info("="*70)
    
    # Create engine with default settings
    engine = RealtimeEEGEngine()
    
    # Run for 30 seconds
    engine.run(duration=30)


def example_custom_config():
    """
    Example 2: Custom configuration.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 2: Custom Configuration")
    logger.info("="*70)
    
    # Create custom configuration
    config = SystemConfig()
    
    # Customize serial settings
    config.serial.port = "COM3"  # Specify your port
    config.serial.baudrate = 115200
    
    # Customize signal processing
    config.signal.sampling_rate = 250.0
    config.signal.window_size = 2.0  # 2-second windows
    config.signal.window_overlap = 0.5  # 50% overlap
    
    # Customize decision thresholds
    config.decision.li_left_threshold = -0.20  # More sensitive to left
    config.decision.li_right_threshold = 0.20  # More sensitive to right
    config.decision.decision_smoothing_window = 7  # More smoothing
    
    # Customize logging
    config.logging.csv_filename = "my_eeg_data.csv"
    
    # Create engine with custom config
    engine = RealtimeEEGEngine(config)
    
    # Run for 60 seconds
    engine.run(duration=60)


def example_context_manager():
    """
    Example 3: Using context manager pattern.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 3: Context Manager Pattern")
    logger.info("="*70)
    
    # Automatic connection and cleanup
    with RealtimeEEGEngine() as engine:
        engine.run(duration=45)
    
    # Engine is automatically stopped and cleaned up


def example_adaptive_thresholds():
    """
    Example 4: Using adaptive threshold calibration.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 4: Adaptive Threshold Calibration")
    logger.info("="*70)
    
    config = SystemConfig()
    
    # Enable adaptive thresholds
    config.decision.adaptive_threshold = True
    config.decision.auto_calibration_samples = 50  # Calibrate with 50 samples
    
    engine = RealtimeEEGEngine(config)
    
    logger.info("Starting calibration phase...")
    logger.info("Please maintain neutral gaze for initial calibration")
    
    # Run indefinitely (Ctrl+C to stop)
    try:
        engine.run()
    except KeyboardInterrupt:
        logger.info("Stopping...")


def example_monitoring():
    """
    Example 5: Monitoring with periodic status checks.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 5: Real-time Monitoring")
    logger.info("="*70)
    
    engine = RealtimeEEGEngine()
    
    if not engine.connect_and_start():
        logger.error("Failed to connect")
        return
    
    # Run with manual monitoring
    engine.is_running = True
    start_time = time.time()
    
    try:
        while time.time() - start_time < 60:
            # Process in background
            time.sleep(5)
            
            # Get statistics
            acq_stats = engine.acquisition.get_statistics()
            dec_stats = engine.decision_engine.get_statistics()
            cal_status = engine.decision_engine.get_calibration_status()
            
            # Custom logging
            logger.info(
                f"\nCustom Status Report:\n"
                f"  Samples: {engine.samples_processed}\n"
                f"  Windows: {engine.windows_processed}\n"
                f"  Corruption: {acq_stats['corruption_rate_percent']:.1f}%\n"
                f"  Calibrated: {cal_status['is_calibrated']}\n"
                f"  LI Mean: {dec_stats.get('li_mean', 0):.3f}\n"
                f"  LEFT decisions: {dec_stats.get('left_decisions', 0)}\n"
                f"  RIGHT decisions: {dec_stats.get('right_decisions', 0)}\n"
                f"  NEUTRAL decisions: {dec_stats.get('neutral_decisions', 0)}"
            )
    
    except KeyboardInterrupt:
        logger.info("Interrupted")
    
    finally:
        engine.stop()


def example_update_config():
    """
    Example 6: Runtime configuration updates.
    """
    logger.info("="*70)
    logger.info("EXAMPLE 6: Runtime Configuration Updates")
    logger.info("="*70)
    
    # Update global configuration
    update_config(
        serial__port="COM4",
        signal__sampling_rate=250.0,
        signal__window_size=1.5,
        decision__li_left_threshold=-0.25,
        decision__li_right_threshold=0.25,
        decision__decision_smoothing_window=10,
        logging__csv_filename="updated_eeg_log.csv"
    )
    
    # Use updated config
    config = get_config()
    engine = RealtimeEEGEngine(config)
    
    engine.run(duration=30)


def example_high_sensitivity():
    """
    Example 7: High sensitivity mode (tight thresholds).
    """
    logger.info("="*70)
    logger.info("EXAMPLE 7: High Sensitivity Mode")
    logger.info("="*70)
    
    config = SystemConfig()
    
    # Tight thresholds for more sensitive detection
    config.decision.li_left_threshold = -0.10
    config.decision.li_right_threshold = 0.10
    config.decision.min_confidence = 0.5
    
    # Shorter window for faster response
    config.signal.window_size = 1.5
    
    # More aggressive artifact rejection
    config.artifact.variance_threshold_multiplier = 2.5
    config.artifact.beta_power_threshold = 80.0
    
    engine = RealtimeEEGEngine(config)
    engine.run(duration=45)


def example_conservative_mode():
    """
    Example 8: Conservative mode (relaxed thresholds).
    """
    logger.info("="*70)
    logger.info("EXAMPLE 8: Conservative Mode")
    logger.info("="*70)
    
    config = SystemConfig()
    
    # Relaxed thresholds for higher confidence
    config.decision.li_left_threshold = -0.30
    config.decision.li_right_threshold = 0.30
    config.decision.min_confidence = 0.75
    
    # Longer window for more stable estimates
    config.signal.window_size = 3.0
    
    # More smoothing
    config.decision.decision_smoothing_window = 10
    
    engine = RealtimeEEGEngine(config)
    engine.run(duration=45)


def main():
    """
    Main function to run examples.
    """
    print("\n" + "="*70)
    print("EEG Processing System - Usage Examples")
    print("="*70)
    print("\nAvailable examples:")
    print("  1. Basic usage")
    print("  2. Custom configuration")
    print("  3. Context manager pattern")
    print("  4. Adaptive threshold calibration")
    print("  5. Real-time monitoring")
    print("  6. Runtime configuration updates")
    print("  7. High sensitivity mode")
    print("  8. Conservative mode")
    print("\nPress Ctrl+C to stop any example\n")
    print("="*70)
    
    # Choose which example to run
    choice = input("\nSelect example (1-8) or 'all' to run all: ").strip()
    
    examples = {
        '1': example_basic,
        '2': example_custom_config,
        '3': example_context_manager,
        '4': example_adaptive_thresholds,
        '5': example_monitoring,
        '6': example_update_config,
        '7': example_high_sensitivity,
        '8': example_conservative_mode
    }
    
    if choice == 'all':
        for func in examples.values():
            try:
                func()
                time.sleep(2)
            except KeyboardInterrupt:
                logger.info("Skipping to next example...")
                time.sleep(1)
    elif choice in examples:
        examples[choice]()
    else:
        logger.error("Invalid choice")


if __name__ == "__main__":
    main()
