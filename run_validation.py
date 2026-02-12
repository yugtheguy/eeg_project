"""
Quick Validation Runner

Allows running individual validation tests or the full suite.
Useful for targeted testing during development and troubleshooting.
"""

import sys
import logging
from validation import EEGValidator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def print_menu():
    """Display test menu."""
    print("\n" + "="*70)
    print("EEG SYSTEM VALIDATION - QUICK TEST RUNNER")
    print("="*70)
    print("\nAvailable Tests:")
    print("  1. Sampling Rate Verification (60s)")
    print("  2. ADC Scaling Validation (10s)")
    print("  3. Serial Integrity Test (60s)")
    print("  4. Filter Frequency Response")
    print("  5. Filtering Effect (30s)")
    print("  6. Alpha Power Stability (60s)")
    print("  7. Artifact Detection (30s)")
    print("  8. Lateralization Index (Interactive)")
    print("  9. Decision Stability (5 min)")
    print(" 10. Real-time Performance (10 min)")
    print("\n  F. Full Validation Suite")
    print("  Q. Quick Validation (reduced durations)")
    print("  R. Generate Report from existing results")
    print("  X. Exit")
    print("="*70)


def run_individual_test(validator: EEGValidator, test_num: str):
    """Run a single test."""
    tests = {
        '1': ('Sampling Rate', lambda: validator.validate_sampling_rate(60.0)),
        '2': ('ADC Scaling', lambda: validator.validate_adc_scaling(10.0)),
        '3': ('Serial Integrity', lambda: validator.validate_serial_integrity(60.0)),
        '4': ('Frequency Response', lambda: validator.validate_frequency_response()),
        '5': ('Filtering Effect', lambda: validator.validate_filtering_effect(30.0)),
        '6': ('Alpha Power Stability', lambda: validator.validate_alpha_power_stability(60.0)),
        '7': ('Artifact Detection', lambda: validator.validate_artifact_detection(30.0)),
        '8': ('Lateralization Index', lambda: validator.validate_lateralization_index(5, 20.0)),
        '9': ('Decision Stability', lambda: validator.validate_decision_stability(300.0)),
        '10': ('Real-time Performance', lambda: validator.validate_realtime_performance(600.0))
    }
    
    if test_num not in tests:
        logger.error(f"Invalid test number: {test_num}")
        return False
    
    test_name, test_func = tests[test_num]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Running: {test_name}")
    logger.info(f"{'='*70}")
    
    try:
        # Connect if not already connected
        if not validator.acquisition.is_connected:
            if not validator.acquisition.connect():
                logger.error("Failed to connect. Cannot run test.")
                return False
        
        result = test_func()
        
        if result:
            logger.info(f"\n✅ {test_name} PASSED")
        else:
            logger.error(f"\n❌ {test_name} FAILED")
        
        return result
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Test interrupted")
        return False
    
    except Exception as e:
        logger.error(f"\n❌ Test error: {e}", exc_info=True)
        return False


def main():
    """Main interactive menu."""
    validator = EEGValidator()
    
    while True:
        print_menu()
        choice = input("\nSelect test or option: ").strip().upper()
        
        if choice == 'X':
            logger.info("Exiting...")
            if validator.acquisition.is_connected:
                validator.acquisition.disconnect()
            break
        
        elif choice == 'F':
            logger.info("Running FULL validation suite...")
            try:
                validator.run_full_validation(quick_mode=False)
            except KeyboardInterrupt:
                logger.info("\n⏹️  Validation interrupted")
        
        elif choice == 'Q':
            logger.info("Running QUICK validation...")
            try:
                validator.run_full_validation(quick_mode=True)
            except KeyboardInterrupt:
                logger.info("\n⏹️  Validation interrupted")
        
        elif choice == 'R':
            logger.info("Generating report from existing results...")
            validator.generate_report()
        
        elif choice.isdigit():
            run_individual_test(validator, choice)
        
        else:
            logger.warning("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")
    
    logger.info("Goodbye!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
