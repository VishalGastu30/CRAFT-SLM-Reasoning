from loguru import logger

class KLController:
    """
    Dynamic KL divergence controller.
    Automatically scales the KL coefficient beta up or down
    based on whether the current KL divergence exceeds target bounds,
    preventing distribution collapse while allowing optimization.
    """
    def __init__(self, initial_beta: float = 0.04, target_kl: float = 0.1, kl_target: float = None, min_beta: float = 0.005, max_beta: float = 0.5, beta_min: float = None, beta_max: float = None):
        self.beta = initial_beta
        self.target_kl = kl_target if kl_target is not None else target_kl
        # support both beta_min / min_beta and beta_max / max_beta
        self.beta_min = beta_min if beta_min is not None else min_beta
        self.beta_max = beta_max if beta_max is not None else max_beta
        self.history = []
        logger.info(f"KLController initialized with initial_beta={self.beta}, target_kl={self.target_kl}")

    def step(self, kl_value: float) -> float:
        """
        Called after each training step with the measured KL.
        Adjusts beta to keep KL near target_kl.
        Returns current beta.
        """
        # Clamp: KL is always non-negative
        kl_value = max(0.0, kl_value)
        self.history.append(kl_value)
        
        # If KL is too low (model barely changing), loosen the penalty
        if kl_value < 0.5 * self.target_kl:
            old_beta = self.beta
            self.beta = max(self.beta_min, self.beta * 0.9)
            logger.info(f"KL divergence ({kl_value:.4f}) below 0.5x target ({self.target_kl}). Loosening penalty: {old_beta:.4f} -> {self.beta:.4f}")
        
        # If KL is too high (model changing too fast), tighten the penalty
        elif kl_value > 2.0 * self.target_kl:
            old_beta = self.beta
            self.beta = min(self.beta_max, self.beta * 1.5)
            logger.warning(f"KL divergence ({kl_value:.4f}) exceeds 2.0x target ({self.target_kl}). Tightening penalty: {old_beta:.4f} -> {self.beta:.4f}")
        
        # Otherwise: KL is in acceptable range, keep beta as is
        return self.beta

    def is_destabilizing(self) -> bool:
        """True if KL spiked suddenly — sign of training instability."""
        if len(self.history) < 10:
            return False
        recent = self.history[-5:]
        prev = self.history[-10:-5]
        return (sum(recent) / 5) > 3 * (sum(prev) / 5 + 1e-8)

    def get_beta(self) -> float:
        return self.beta
