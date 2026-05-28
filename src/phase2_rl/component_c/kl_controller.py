from loguru import logger

class KLController:
    """
    Dynamic KL divergence controller.
    Automatically scales the KL coefficient beta up or down
    based on whether the current KL divergence exceeds target bounds,
    preventing distribution collapse while allowing optimization.
    """
    def __init__(self, initial_beta=0.04, kl_target=0.1, min_beta=0.005, max_beta=0.5):
        self.beta = initial_beta
        self.kl_target = kl_target
        self.min_beta = min_beta
        self.max_beta = max_beta
        logger.info(f"KLController initialized with initial_beta={self.beta}, target_kl={self.kl_target}")

    def step(self, current_kl: float) -> float:
        """
        Dynamically adjusts beta depending on the difference between 
        current KL and target KL.
        Returns the updated beta value.
        """
        if current_kl > (self.kl_target * 1.5):
            # Diverging too fast: increase penalty to pull model closer to reference
            old_beta = self.beta
            self.beta = min(self.max_beta, self.beta * 1.5)
            logger.warning(f"KL divergence ({current_kl:.4f}) exceeds 1.5x target ({self.kl_target}). Tightening penalty: {old_beta:.4f} -> {self.beta:.4f}")
        elif current_kl < (self.kl_target * 0.5):
            # Learning is too conservative / slow: reduce penalty to permit exploration
            old_beta = self.beta
            self.beta = max(self.min_beta, self.beta * 0.9)
            logger.info(f"KL divergence ({current_kl:.4f}) below 0.5x target ({self.kl_target}). Loosening penalty: {old_beta:.4f} -> {self.beta:.4f}")
            
        return self.beta
