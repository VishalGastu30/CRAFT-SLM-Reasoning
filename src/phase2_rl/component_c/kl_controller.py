"""
kl_controller.py — Component C: Dynamic KL Coefficient Controller

Controls how tightly the training policy is constrained to stay close
to the SFT reference model. Higher beta = tighter constraint = more stable
but slower learning. Lower beta = looser = faster but more risky.

KEY CHANGE in v3:
- Added warmup_steps parameter: beta is FROZEN for first N steps.
  This prevents beta from decaying before the model has had a chance
  to start actually learning (when KL is naturally very small).
- Reduced loosening rate from 10% to 5% per adjustment.
- Added hard minimum floor enforcement.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class KLController:
    """
    Adaptive KL penalty coefficient controller.
    
    Loosens beta (reduces penalty) when KL divergence is below target
    (model not diverging enough — too conservative).
    Tightens beta (increases penalty) when KL divergence is above target
    (model diverging too fast — needs more constraint).
    
    Args:
        initial_beta: Starting KL coefficient. Recommended: 0.04
        target_kl: Target KL divergence. Recommended: 0.1
        beta_min: Minimum beta floor. Recommended: 0.005
        beta_max: Maximum beta ceiling. Recommended: 0.5
        adjustment_factor: How much to adjust per step. Recommended: 0.05 (5%)
                          DO NOT use 0.10 — too aggressive for SLMs.
        warmup_steps: Steps to freeze beta before allowing adjustments. 
                      Recommended: 20. Prevents early decay when KL is
                      naturally near zero before learning begins.
    """
    
    def __init__(
        self,
        initial_beta: float = 0.04,
        target_kl: float = 0.1,
        beta_min: float = 0.005,
        beta_max: float = 0.5,
        adjustment_factor: float = 0.05,
        warmup_steps: int = 20,
    ):
        self.beta = initial_beta
        self._initial_beta = initial_beta
        self.target_kl = target_kl
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.adjustment_factor = adjustment_factor
        self.warmup_steps = warmup_steps
        self._step_count = 0
        self.history = []
    
    def reset(self):
        """
        Reset to initial state. Call this when starting a fresh training run
        (not resuming from checkpoint). Prevents carrying over a decayed beta
        from a previous run.
        """
        self.beta = self._initial_beta
        self._step_count = 0
        self.history = []
        logger.info(f"KL controller reset. beta={self.beta:.4f}")
    
    def step(self, kl_divergence: float) -> float:
        """
        Update beta based on observed KL divergence.
        
        Call once per training step, AFTER computing KL divergence.
        
        Args:
            kl_divergence: Observed KL divergence for this step. Must be >= 0.
        
        Returns:
            Current beta value (after any adjustment).
        """
        self._step_count += 1
        
        # Clamp kl_divergence to valid range
        kl_val = max(0.0, float(kl_divergence))
        self.history.append(kl_val)
        
        # WARMUP PERIOD: freeze beta for the first N steps.
        # Reason: in early training, KL is near 0 because the model hasn't
        # updated much yet. This would trigger loosening, but it's premature —
        # we want beta stable until the model starts actually learning.
        if self._step_count <= self.warmup_steps:
            logger.debug(
                f"KL warmup step {self._step_count}/{self.warmup_steps}. "
                f"Beta frozen at {self.beta:.4f}. KL={kl_val:.4f}"
            )
            return self.beta
        
        # After warmup: use rolling average of last 5 steps to reduce noise
        recent_kl_vals = self.history[-5:]
        avg_kl = sum(recent_kl_vals) / len(recent_kl_vals)
        
        # Adjust beta based on whether KL is above or below target
        if avg_kl < 0.5 * self.target_kl:
            # KL too low — model not diverging enough — loosen constraint
            new_beta = self.beta * (1.0 - self.adjustment_factor)
            new_beta = max(new_beta, self.beta_min)
            if new_beta != self.beta:
                logger.info(
                    f"KL avg ({avg_kl:.4f}) below 0.5x target ({self.target_kl}). "
                    f"Loosening: {self.beta:.4f} → {new_beta:.4f}"
                )
            self.beta = new_beta
            
        elif avg_kl > 1.5 * self.target_kl:
            # KL too high — model diverging too fast — tighten constraint
            new_beta = self.beta * (1.0 + self.adjustment_factor)
            new_beta = min(new_beta, self.beta_max)
            if new_beta != self.beta:
                logger.info(
                    f"KL avg ({avg_kl:.4f}) above 1.5x target ({self.target_kl}). "
                    f"Tightening: {self.beta:.4f} → {new_beta:.4f}"
                )
            self.beta = new_beta
        
        # else: KL is in the acceptable range — no change
        
        return self.beta
    
    def get_beta(self) -> float:
        """Return current beta value."""
        return self.beta
