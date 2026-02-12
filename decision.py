"""
Decision Engine Module.

Implements attention direction detection based on hemispheric lateralization:
- Lateralization Index (LI) computation
- Attention direction classification (LEFT/RIGHT/NEUTRAL)
- Decision smoothing and confidence estimation
- Adaptive threshold calibration
"""

import numpy as np
from collections import deque
import logging
from typing import Optional, Tuple, Dict
from enum import Enum

from config import DecisionConfig, get_config


# Configure logging
logger = logging.getLogger(__name__)


class AttentionDirection(Enum):
    """Attention direction states."""
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class DecisionEngine:
    """
    Attention direction detection based on alpha power asymmetry.
    
    Implements lateralization index computation and decision logic with
    smoothing and adaptive threshold calibration.
    """
    
    def __init__(self, decision_config: Optional[DecisionConfig] = None):
        """
        Initialize decision engine.
        
        Args:
            decision_config: Decision configuration. If None, uses global config.
        """
        if decision_config is None:
            decision_config = get_config().decision
        
        self.config = decision_config
        
        # Decision history for smoothing
        self.decision_history: deque = deque(
            maxlen=self.config.decision_smoothing_window
        )
        
        # Lateralization index history for adaptive thresholds
        self.li_history: deque = deque(
            maxlen=self.config.auto_calibration_samples
        )
        
        # Adaptive thresholds
        self.adaptive_left_threshold = self.config.li_left_threshold
        self.adaptive_right_threshold = self.config.li_right_threshold
        
        # Calibration status
        self.is_calibrated = False
        self.calibration_samples = 0
        
        logger.info("DecisionEngine initialized")
    
    def compute_lateralization_index(
        self,
        left_alpha_power: float,
        right_alpha_power: float
    ) -> float:
        """
        Compute Lateralization Index (LI).
        
        LI = (Right - Left) / (Right + Left)
        
        Negative LI indicates stronger right hemisphere alpha (left attention)
        Positive LI indicates stronger left hemisphere alpha (right attention)
        
        Note: Higher alpha power indicates lower cortical activity, so:
        - High left alpha → low left activity → attention directed RIGHT
        - High right alpha → low right activity → attention directed LEFT
        
        Args:
            left_alpha_power: Alpha band power from left hemisphere
            right_alpha_power: Alpha band power from right hemisphere
        
        Returns:
            float: Lateralization index (-1 to +1)
        """
        try:
            # Avoid division by zero
            total_power = left_alpha_power + right_alpha_power
            
            if total_power == 0:
                return 0.0
            
            # Compute lateralization index
            li = (right_alpha_power - left_alpha_power) / total_power
            
            # Clip to valid range
            li = np.clip(li, -1.0, 1.0)
            
            return li
            
        except Exception as e:
            logger.error(f"LI computation error: {e}")
            return 0.0
    
    def classify_attention(
        self,
        lateralization_index: float
    ) -> AttentionDirection:
        """
        Classify attention direction based on lateralization index.
        
        Args:
            lateralization_index: Computed LI value
        
        Returns:
            AttentionDirection: Detected attention direction
        """
        try:
            # Use adaptive thresholds if available
            left_threshold = self.adaptive_left_threshold
            right_threshold = self.adaptive_right_threshold
            
            if lateralization_index < left_threshold:
                # Negative LI: stronger right alpha → LEFT attention
                return AttentionDirection.LEFT
            elif lateralization_index > right_threshold:
                # Positive LI: stronger left alpha → RIGHT attention
                return AttentionDirection.RIGHT
            else:
                # Within neutral zone
                return AttentionDirection.NEUTRAL
                
        except Exception as e:
            logger.error(f"Attention classification error: {e}")
            return AttentionDirection.UNKNOWN
    
    def compute_confidence(
        self,
        lateralization_index: float,
        attention: AttentionDirection
    ) -> float:
        """
        Compute confidence score for the decision.
        
        Args:
            lateralization_index: Computed LI value
            attention: Classified attention direction
        
        Returns:
            float: Confidence score (0-1)
        """
        try:
            if attention == AttentionDirection.NEUTRAL:
                # Distance from both thresholds
                dist_left = abs(lateralization_index - self.adaptive_left_threshold)
                dist_right = abs(lateralization_index - self.adaptive_right_threshold)
                
                # Confidence is lower in the middle of neutral zone
                neutral_zone_width = self.adaptive_right_threshold - self.adaptive_left_threshold
                if neutral_zone_width > 0:
                    min_dist = min(dist_left, dist_right)
                    confidence = 1.0 - (neutral_zone_width - 2 * min_dist) / neutral_zone_width
                    confidence = np.clip(confidence, 0.0, 1.0)
                else:
                    confidence = 0.5
                
            elif attention == AttentionDirection.LEFT:
                # How far below left threshold
                distance = abs(lateralization_index - self.adaptive_left_threshold)
                max_distance = 1.0 + abs(self.adaptive_left_threshold)
                confidence = min(distance / max_distance * 2, 1.0)
                
            elif attention == AttentionDirection.RIGHT:
                # How far above right threshold
                distance = abs(lateralization_index - self.adaptive_right_threshold)
                max_distance = 1.0 - self.adaptive_right_threshold
                confidence = min(distance / max_distance * 2, 1.0)
                
            else:  # UNKNOWN
                confidence = 0.0
            
            return confidence
            
        except Exception as e:
            logger.error(f"Confidence computation error: {e}")
            return 0.0
    
    def make_decision(
        self,
        left_alpha_power: float,
        right_alpha_power: float
    ) -> Tuple[AttentionDirection, float, float]:
        """
        Make attention direction decision with confidence.
        
        Args:
            left_alpha_power: Alpha power from left hemisphere
            right_alpha_power: Alpha power from right hemisphere
        
        Returns:
            tuple: (attention_direction, lateralization_index, confidence)
        """
        try:
            # Compute lateralization index
            li = self.compute_lateralization_index(left_alpha_power, right_alpha_power)
            
            # Store in history for calibration
            self.li_history.append(li)
            self.calibration_samples += 1
            
            # Update adaptive thresholds if enabled
            if self.config.adaptive_threshold:
                self._update_adaptive_thresholds()
            
            # Classify attention direction
            attention = self.classify_attention(li)
            
            # Compute confidence
            confidence = self.compute_confidence(li, attention)
            
            # Apply minimum confidence threshold
            if confidence < self.config.min_confidence:
                if attention != AttentionDirection.NEUTRAL:
                    logger.debug(f"Low confidence ({confidence:.2f}), setting to NEUTRAL")
                    attention = AttentionDirection.NEUTRAL
            
            # Store decision in history
            self.decision_history.append((attention, confidence))
            
            return attention, li, confidence
            
        except Exception as e:
            logger.error(f"Decision making error: {e}")
            return AttentionDirection.UNKNOWN, 0.0, 0.0
    
    def get_smoothed_decision(self) -> Tuple[AttentionDirection, float]:
        """
        Get smoothed decision based on decision history.
        
        Uses majority voting with confidence weighting.
        
        Returns:
            tuple: (smoothed_attention_direction, average_confidence)
        """
        try:
            if len(self.decision_history) == 0:
                return AttentionDirection.UNKNOWN, 0.0
            
            # Count weighted votes for each direction
            left_score = 0.0
            right_score = 0.0
            neutral_score = 0.0
            
            for attention, confidence in self.decision_history:
                if attention == AttentionDirection.LEFT:
                    left_score += confidence
                elif attention == AttentionDirection.RIGHT:
                    right_score += confidence
                elif attention == AttentionDirection.NEUTRAL:
                    neutral_score += confidence
            
            # Find winner
            max_score = max(left_score, right_score, neutral_score)
            
            if max_score == 0:
                return AttentionDirection.NEUTRAL, 0.0
            
            if left_score == max_score:
                final_attention = AttentionDirection.LEFT
            elif right_score == max_score:
                final_attention = AttentionDirection.RIGHT
            else:
                final_attention = AttentionDirection.NEUTRAL
            
            # Average confidence
            total_confidence = sum(conf for _, conf in self.decision_history)
            avg_confidence = total_confidence / len(self.decision_history)
            
            return final_attention, avg_confidence
            
        except Exception as e:
            logger.error(f"Smoothed decision error: {e}")
            return AttentionDirection.UNKNOWN, 0.0
    
    def _update_adaptive_thresholds(self) -> None:
        """
        Update adaptive thresholds based on LI history.
        
        Calibrates thresholds to account for individual differences.
        """
        try:
            # Need sufficient samples for calibration
            if len(self.li_history) < 20:
                return
            
            li_array = np.array(list(self.li_history))
            
            # Compute statistics
            mean_li = np.mean(li_array)
            std_li = np.std(li_array)
            
            # Adjust thresholds based on distribution
            # If mean is not centered at 0, shift thresholds
            if not self.is_calibrated:
                if len(self.li_history) >= self.config.auto_calibration_samples:
                    # Initial calibration
                    self.adaptive_left_threshold = mean_li - std_li
                    self.adaptive_right_threshold = mean_li + std_li
                    
                    self.is_calibrated = True
                    
                    logger.info(
                        f"Calibration complete: "
                        f"LEFT threshold={self.adaptive_left_threshold:.3f}, "
                        f"RIGHT threshold={self.adaptive_right_threshold:.3f}"
                    )
            else:
                # Ongoing adaptation (slower)
                alpha = 0.1  # Adaptation rate
                
                target_left = mean_li - std_li
                target_right = mean_li + std_li
                
                self.adaptive_left_threshold = (
                    (1 - alpha) * self.adaptive_left_threshold +
                    alpha * target_left
                )
                self.adaptive_right_threshold = (
                    (1 - alpha) * self.adaptive_right_threshold +
                    alpha * target_right
                )
                
                # Ensure minimum separation
                min_separation = 0.1
                if self.adaptive_right_threshold - self.adaptive_left_threshold < min_separation:
                    center = (self.adaptive_left_threshold + self.adaptive_right_threshold) / 2
                    self.adaptive_left_threshold = center - min_separation / 2
                    self.adaptive_right_threshold = center + min_separation / 2
                
        except Exception as e:
            logger.error(f"Adaptive threshold update error: {e}")
    
    def get_calibration_status(self) -> Dict[str, any]:
        """
        Get current calibration status and threshold values.
        
        Returns:
            dict: Calibration information
        """
        return {
            'is_calibrated': self.is_calibrated,
            'calibration_samples': self.calibration_samples,
            'required_samples': self.config.auto_calibration_samples,
            'left_threshold': self.adaptive_left_threshold,
            'right_threshold': self.adaptive_right_threshold,
            'li_history_size': len(self.li_history),
            'decision_history_size': len(self.decision_history)
        }
    
    def reset_calibration(self) -> None:
        """Reset calibration and return to default thresholds."""
        self.li_history.clear()
        self.decision_history.clear()
        self.adaptive_left_threshold = self.config.li_left_threshold
        self.adaptive_right_threshold = self.config.li_right_threshold
        self.is_calibrated = False
        self.calibration_samples = 0
        
        logger.info("Calibration reset to defaults")
    
    def get_statistics(self) -> Dict[str, any]:
        """
        Get decision engine statistics.
        
        Returns:
            dict: Statistics including decision counts, LI statistics, etc.
        """
        try:
            if len(self.li_history) == 0:
                return {
                    'li_mean': 0.0,
                    'li_std': 0.0,
                    'li_min': 0.0,
                    'li_max': 0.0,
                    'left_decisions': 0,
                    'right_decisions': 0,
                    'neutral_decisions': 0,
                    'avg_confidence': 0.0
                }
            
            li_array = np.array(list(self.li_history))
            
            # Count decisions
            left_count = sum(
                1 for att, _ in self.decision_history
                if att == AttentionDirection.LEFT
            )
            right_count = sum(
                1 for att, _ in self.decision_history
                if att == AttentionDirection.RIGHT
            )
            neutral_count = sum(
                1 for att, _ in self.decision_history
                if att == AttentionDirection.NEUTRAL
            )
            
            # Average confidence
            if len(self.decision_history) > 0:
                avg_conf = np.mean([conf for _, conf in self.decision_history])
            else:
                avg_conf = 0.0
            
            return {
                'li_mean': np.mean(li_array),
                'li_std': np.std(li_array),
                'li_min': np.min(li_array),
                'li_max': np.max(li_array),
                'left_decisions': left_count,
                'right_decisions': right_count,
                'neutral_decisions': neutral_count,
                'avg_confidence': avg_conf
            }
            
        except Exception as e:
            logger.error(f"Statistics computation error: {e}")
            return {}


def create_decision_engine(
    decision_config: Optional[DecisionConfig] = None
) -> DecisionEngine:
    """
    Factory function to create a DecisionEngine instance.
    
    Args:
        decision_config: Decision configuration. If None, uses global config.
    
    Returns:
        DecisionEngine: Configured decision engine instance
    """
    return DecisionEngine(decision_config)
