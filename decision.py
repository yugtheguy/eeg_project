"""
Focus Detection Engine using Alpha Suppression.

Single-channel focus detection based on alpha power suppression:
- Baseline calibration (relaxed state)
- Real-time suppression ratio computation
- Focus state classification (FOCUSED/RELAXED/NEUTRAL)
- Decision smoothing and confidence estimation
- Signal quality gating
"""

import numpy as np
from collections import deque
import logging
import time
from typing import Optional, Tuple, Dict
from enum import Enum

from config import DecisionConfig, get_config


# Configure logging
logger = logging.getLogger(__name__)


class FocusState(Enum):
    """Focus state classifications."""
    FOCUSED = "FOCUSED"
    RELAXED = "RELAXED"
    NEUTRAL = "NEUTRAL"
    UNRELIABLE = "UNRELIABLE"
    UNCALIBRATED = "UNCALIBRATED"


class FocusDetectionEngine:
    """
    Single-channel focus detection based on alpha power suppression.
    
    Alpha suppression principle:
    - Relaxed (eyes closed): High alpha power (baseline)
    - Focused (mental task): Low alpha power (suppressed)
    - Suppression ratio = current_alpha / baseline_alpha
    """
    
    def __init__(self, decision_config: Optional[DecisionConfig] = None):
        """
        Initialize focus detection engine.
        
        Args:
            decision_config: Decision configuration. If None, uses global config.
        """
        if decision_config is None:
            decision_config = get_config().decision
        
        self.config = decision_config
        
        # Baseline calibration
        self.baseline_alpha = None
        self.baseline_std = None
        self.is_calibrated = False
        self.calibration_alpha_values = []
        
        # Suppression ratio history
        self.suppression_history: deque = deque(
            maxlen=self.config.suppression_history_size
        )
        
        # Decision history for smoothing
        self.decision_history: deque = deque(
            maxlen=self.config.decision_smoothing_window
        )
        
        # Statistics
        self.total_decisions = 0
        self.focused_count = 0
        self.relaxed_count = 0
        self.neutral_count = 0
        
        logger.info("FocusDetectionEngine initialized (single-channel alpha suppression)")
    
    def calibrate_baseline(
        self,
        alpha_power_values: list,
        quality_scores: list = None
    ) -> bool:
        """
        Calibrate baseline alpha power from relaxed state data.
        
        User should be relaxed with eyes closed during calibration.
        
        Args:
            alpha_power_values: List of alpha power measurements
            quality_scores: Optional list of quality scores for filtering
        
        Returns:
            bool: True if calibration successful
        """
        try:
            if len(alpha_power_values) < 10:
                logger.error("Insufficient data for calibration (need at least 10 samples)")
                return False
            
            # Filter by quality if provided
            if quality_scores is not None:
                filtered_values = [
                    alpha for alpha, quality in zip(alpha_power_values, quality_scores)
                    if quality >= self.config.min_quality_score
                ]
                
                if len(filtered_values) < 5:
                    logger.warning("Too few high-quality samples, using all data")
                    filtered_values = alpha_power_values
            else:
                filtered_values = alpha_power_values
            
            # Remove outliers (beyond 3 standard deviations)
            values_array = np.array(filtered_values)
            mean = np.mean(values_array)
            std = np.std(values_array)
            
            if std > 0:
                z_scores = np.abs((values_array - mean) / std)
                clean_values = values_array[z_scores < 3]
                
                if len(clean_values) < 5:
                    clean_values = values_array
            else:
                clean_values = values_array
            
            # Compute baseline statistics
            self.baseline_alpha = float(np.mean(clean_values))
            self.baseline_std = float(np.std(clean_values))
            self.is_calibrated = True
            self.calibration_alpha_values = clean_values.tolist()
            
            logger.info(
                f"✓ Baseline calibration complete: "
                f"mean={self.baseline_alpha:.2f}, "
                f"std={self.baseline_std:.2f}, "
                f"n={len(clean_values)}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Calibration error: {e}")
            return False
    
    def compute_suppression_ratio(self, alpha_power: float) -> float:
        """
        Compute suppression ratio relative to baseline.
        
        Args:
            alpha_power: Current alpha power
        
        Returns:
            float: Suppression ratio (current / baseline)
        """
        if not self.is_calibrated or self.baseline_alpha is None:
            logger.warning("Engine not calibrated, cannot compute suppression ratio")
            return 1.0
        
        # Avoid division by zero
        ratio = alpha_power / (self.baseline_alpha + 1e-8)
        
        # Store in history
        self.suppression_history.append(ratio)
        
        return ratio
    
    def classify_focus(
        self,
        suppression_ratio: float,
        quality_score: float = 100.0,
        snr_db: float = 20.0,
        has_artifact: bool = False
    ) -> FocusState:
        """
        Classify focus state based on suppression ratio.
        
        Args:
            suppression_ratio: Current suppression ratio
            quality_score: Signal quality score (0-100)
            snr_db: Signal-to-noise ratio in dB
            has_artifact: Whether artifact detected
        
        Returns:
            FocusState: Detected focus state
        """
        try:
            # Check if calibrated
            if not self.is_calibrated:
                return FocusState.UNCALIBRATED
            
            # Quality gating
            if (quality_score < self.config.min_quality_score or
                snr_db < self.config.min_snr_threshold or
                has_artifact):
                logger.debug(
                    f"Low quality data: quality={quality_score:.1f}, "
                    f"SNR={snr_db:.1f}dB, artifact={has_artifact}"
                )
                return FocusState.UNRELIABLE
            
            # Classify based on thresholds
            if suppression_ratio < self.config.focus_threshold:
                state = FocusState.FOCUSED
                self.focused_count += 1
            elif suppression_ratio > self.config.relax_threshold:
                state = FocusState.RELAXED
                self.relaxed_count += 1
            else:
                state = FocusState.NEUTRAL
                self.neutral_count += 1
            
            self.total_decisions += 1
            
            # Store in history
            self.decision_history.append(state)
            
            return state
            
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return FocusState.UNRELIABLE
    
    def compute_confidence(self, suppression_ratio: float) -> float:
        """
        Compute confidence score based on distance from baseline.
        
        Args:
            suppression_ratio: Current suppression ratio
        
        Returns:
            float: Confidence score (0-1)
        """
        try:
            # Distance from neutral baseline (ratio = 1.0)
            distance = abs(1.0 - suppression_ratio)
            
            # Normalize: max distance of 0.5 gives confidence of 1.0
            confidence = min(distance / 0.5, 1.0)
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence computation error: {e}")
            return 0.0
    
    def get_smoothed_state(self) -> Tuple[FocusState, float]:
        """
        Get smoothed focus state using majority voting.
        
        Returns:
            tuple: (smoothed_state, agreement_fraction)
        """
        try:
            if len(self.decision_history) == 0:
                return FocusState.UNCALIBRATED, 0.0
            
            # Count votes for each state (ignore UNRELIABLE)
            focused_votes = sum(1 for s in self.decision_history if s == FocusState.FOCUSED)
            relaxed_votes = sum(1 for s in self.decision_history if s == FocusState.RELAXED)
            neutral_votes = sum(1 for s in self.decision_history if s == FocusState.NEUTRAL)
            
            # Find majority
            max_votes = max(focused_votes, relaxed_votes, neutral_votes)
            total_valid = focused_votes + relaxed_votes + neutral_votes
            
            if total_valid == 0:
                return FocusState.UNRELIABLE, 0.0
            
            # Determine state
            if focused_votes == max_votes:
                state = FocusState.FOCUSED
            elif relaxed_votes == max_votes:
                state = FocusState.RELAXED
            else:
                state = FocusState.NEUTRAL
            
            # Agreement metric
            agreement = max_votes / len(self.decision_history)
            
            return state, agreement
            
        except Exception as e:
            logger.error(f"Smoothed state error: {e}")
            return FocusState.UNRELIABLE, 0.0
    
    def get_average_suppression(self) -> float:
        """
        Get average suppression ratio from history.
        
        Returns:
            float: Mean suppression ratio
        """
        if len(self.suppression_history) == 0:
            return 1.0
        
        return float(np.mean(list(self.suppression_history)))
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get engine statistics.
        
        Returns:
            dict: Statistics including counts, ratios, baseline info
        """
        try:
            stats = {
                'is_calibrated': self.is_calibrated,
                'baseline_alpha': self.baseline_alpha,
                'baseline_std': self.baseline_std,
                'total_decisions': self.total_decisions,
                'focused_count': self.focused_count,
                'relaxed_count': self.relaxed_count,
                'neutral_count': self.neutral_count,
                'decision_history_size': len(self.decision_history),
                'suppression_history_size': len(self.suppression_history)
            }
            
            # Percentages
            if self.total_decisions > 0:
                stats['focused_percent'] = (self.focused_count / self.total_decisions) * 100
                stats['relaxed_percent'] = (self.relaxed_count / self.total_decisions) * 100
                stats['neutral_percent'] = (self.neutral_count / self.total_decisions) * 100
            
            # Suppression ratio statistics
            if len(self.suppression_history) > 0:
                ratios = np.array(list(self.suppression_history))
                stats['avg_suppression'] = float(np.mean(ratios))
                stats['std_suppression'] = float(np.std(ratios))
                stats['min_suppression'] = float(np.min(ratios))
                stats['max_suppression'] = float(np.max(ratios))
            
            return stats
            
        except Exception as e:
            logger.error(f"Statistics computation error: {e}")
            return {}
    
    def reset_calibration(self) -> None:
        """Reset calibration to uncalibrated state."""
        self.baseline_alpha = None
        self.baseline_std = None
        self.is_calibrated = False
        self.calibration_alpha_values = []
        self.suppression_history.clear()
        self.decision_history.clear()
        
        logger.info("Calibration reset")
    
    def reset_history(self) -> None:
        """Reset decision and suppression history (keep calibration)."""
        self.suppression_history.clear()
        self.decision_history.clear()
        self.total_decisions = 0
        self.focused_count = 0
        self.relaxed_count = 0
        self.neutral_count = 0
        
        logger.info("History reset (calibration preserved)")


def create_focus_engine(
    decision_config: Optional[DecisionConfig] = None
) -> FocusDetectionEngine:
    """
    Factory function to create a FocusDetectionEngine instance.
    
    Args:
        decision_config: Decision configuration. If None, uses global config.
    
    Returns:
        FocusDetectionEngine: Configured focus detection engine instance
    """
    return FocusDetectionEngine(decision_config)


# Verification protocol
def verify_focus_protocol(
    engine: FocusDetectionEngine,
    relaxed_alpha_values: list,
    focused_alpha_values: list
) -> Dict[str, any]:
    """
    Verify focus detection with controlled protocol.
    
    Protocol:
    - Phase 1: Relaxed with eyes closed (10 seconds)
    - Phase 2: Mental math or focused task (10 seconds)
    - Compare mean suppression ratios
    
    Args:
        engine: Calibrated FocusDetectionEngine
        relaxed_alpha_values: Alpha power during relaxed phase
        focused_alpha_values: Alpha power during focused phase
    
    Returns:
        dict: Verification results with statistics
    """
    try:
        if not engine.is_calibrated:
            logger.error("Engine must be calibrated before verification")
            return {'error': 'not_calibrated'}
        
        # Compute suppression ratios
        relaxed_ratios = [
            engine.compute_suppression_ratio(alpha)
            for alpha in relaxed_alpha_values
        ]
        
        focused_ratios = [
            engine.compute_suppression_ratio(alpha)
            for alpha in focused_alpha_values
        ]
        
        # Statistics
        relaxed_mean = float(np.mean(relaxed_ratios))
        relaxed_std = float(np.std(relaxed_ratios))
        
        focused_mean = float(np.mean(focused_ratios))
        focused_std = float(np.std(focused_ratios))
        
        # Difference
        difference = relaxed_mean - focused_mean
        percent_change = (difference / relaxed_mean) * 100 if relaxed_mean > 0 else 0
        
        results = {
            'relaxed_phase': {
                'mean_ratio': relaxed_mean,
                'std_ratio': relaxed_std,
                'n_samples': len(relaxed_ratios)
            },
            'focused_phase': {
                'mean_ratio': focused_mean,
                'std_ratio': focused_std,
                'n_samples': len(focused_ratios)
            },
            'difference': difference,
            'percent_suppression': abs(percent_change),
            'detection_valid': difference > 0.15  # At least 15% difference
        }
        
        logger.info(
            f"Verification: Relaxed={relaxed_mean:.3f}, "
            f"Focused={focused_mean:.3f}, "
            f"Difference={difference:.3f} ({percent_change:.1f}%)"
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Verification protocol error: {e}")
        return {'error': str(e)}
