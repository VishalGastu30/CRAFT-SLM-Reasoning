from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class SFTConfig:
    """Configuration class for SFT training."""
    # Model details
    model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct"
    max_seq_length: int = 1024

    # Dataset details – balanced mix
    gsm8k_fraction: float = 0.40
    aqua_fraction: float = 0.20
    strategyqa_fraction: float = 0.20
    mmlu_fraction: float = 0.20
    train_split_seed: int = 42

    # Hyperparameters
    epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"

    # QLoRA parameters
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_up_proj", "down_proj"
    ])

    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 50
    output_dir: str = "checkpoints/sft"
    logging_dir: str = "logs/sft"

    def update_from_profile(self, hardware_profile, local_overrides: bool = False):
        """Updates parameters based on the detected hardware profile."""
        if hardware_profile.is_kaggle:
            self.per_device_train_batch_size = 2
            self.gradient_accumulation_steps = 8
            self.save_steps = 50
        elif hardware_profile.is_local and local_overrides:
            self.per_device_train_batch_size = 4
            self.gradient_accumulation_steps = 4
            self.save_steps = 100

        # VRAM suggestions from hardware detector
        suggestions = hardware_profile.suggest_training_configs(self.model_name_or_path)
        self.per_device_train_batch_size = suggestions.get("per_device_train_batch_size", self.per_device_train_batch_size)
        self.gradient_accumulation_steps = suggestions.get("gradient_accumulation_steps", self.gradient_accumulation_steps)