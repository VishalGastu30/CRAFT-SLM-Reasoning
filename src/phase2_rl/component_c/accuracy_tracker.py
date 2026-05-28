from collections import deque
from loguru import logger

class AccuracyTracker:
    """
    Tracks rolling accuracy of the active reinforcement learning model
    over a sliding window of questions. Used for curriculum difficulty shifting.
    """
    def __init__(self, window_size=50, stability_threshold=0.7):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.window = deque(maxlen=window_size)

    def add(self, is_correct: bool or float):
        """Adds a new observation (1.0 or True for correct, 0.0 or False for incorrect)."""
        val = 1.0 if is_correct else 0.0
        self.window.append(val)

    def get_accuracy(self) -> float:
        """Returns the average accuracy over the tracked sliding window."""
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

    def is_stable(self) -> bool:
        """
        Determines if training has stabilized at/above the target threshold.
        Requires the window to be at least half full to avoid premature scaling.
        """
        if len(self.window) < (self.window_size // 2):
            return False
            
        current_acc = self.get_accuracy()
        stable = current_acc >= self.stability_threshold
        
        if stable:
            logger.info(f"Accuracy stabilized at {current_acc:.2f} (threshold: {self.stability_threshold:.2f})")
            
        return stable

    def reset(self):
        """Resets the rolling window tracker."""
        self.window.clear()
        logger.info("Accuracy tracker reset.")
