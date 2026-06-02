import logging
logger = logging.getLogger(__name__)

class KLController:
    def __init__(
        self,
        initial_beta: float = 0.04,
        target_kl: float = 0.1,
        beta_min: float = 0.005,
        beta_max: float = 0.5,
        adjustment_factor: float = 0.05,   # 5% per step (was 10%)
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
        self.beta = self._initial_beta
        self._step_count = 0
        self.history = []
        logger.info(f"KL controller reset. beta={self.beta:.4f}")

    def step(self, kl_divergence: float) -> float:
        self._step_count += 1
        kl_val = max(0.0, float(kl_divergence))
        self.history.append(kl_val)

        # Warmup: freeze beta
        if self._step_count <= self.warmup_steps:
            return self.beta

        # Rolling average of last 5 steps
        recent = self.history[-5:]
        avg_kl = sum(recent) / len(recent)

        if avg_kl < 0.5 * self.target_kl:
            # Loosen
            new_beta = self.beta * (1.0 - self.adjustment_factor)
            new_beta = max(self.beta_min, new_beta)
            if new_beta != self.beta:
                logger.debug(f"KL low ({avg_kl:.4f}) → loosen beta {self.beta:.4f} → {new_beta:.4f}")
            self.beta = new_beta
        elif avg_kl > 1.5 * self.target_kl:
            # Tighten
            new_beta = self.beta * (1.0 + self.adjustment_factor)
            new_beta = min(self.beta_max, new_beta)
            if new_beta != self.beta:
                logger.debug(f"KL high ({avg_kl:.4f}) → tighten beta {self.beta:.4f} → {new_beta:.4f}")
            self.beta = new_beta
        return self.beta

    def get_beta(self) -> float:
        return self.beta