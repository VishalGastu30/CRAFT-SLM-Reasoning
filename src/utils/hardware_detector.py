import os
import sys
import psutil
import torch
from dataclasses import dataclass
from loguru import logger

@dataclass
class HardwareProfile:
    gpu_count: int
    gpu_vram_gb: float
    gpu_name: str
    cpu_cores: int
    ram_gb: float
    is_kaggle: bool
    is_local: bool

    def suggest_training_configs(self, model_name_or_path="microsoft/Phi-3-mini-4k-instruct"):
        """
        Suggests batch size, gradient accumulation, and device maps
        based on the detected hardware profile.
        """
        # Default fallback (e.g. CPU or low VRAM)
        config = {
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,
            "use_gradient_checkpointing": True,
            "device_map": "auto"
        }
        
        # If we have GPU VRAM
        if self.gpu_vram_gb > 0:
            if "Phi-3-mini" in model_name_or_path or "3.8B" in model_name_or_path:
                if self.gpu_vram_gb >= 24: # 24GB+ GPU (local workstation / A100 / etc.)
                    config["per_device_train_batch_size"] = 4
                    config["gradient_accumulation_steps"] = 4
                elif self.gpu_vram_gb >= 14: # 15GB/16GB GPU (T4 / V100 / etc.)
                    config["per_device_train_batch_size"] = 2
                    config["gradient_accumulation_steps"] = 8
                else: # Low VRAM GPU (like RTX 4050 6GB)
                    config["per_device_train_batch_size"] = 1
                    config["gradient_accumulation_steps"] = 16
            else: # Larger models (like Qwen2.5-7B)
                if self.gpu_vram_gb >= 24:
                    config["per_device_train_batch_size"] = 2
                    config["gradient_accumulation_steps"] = 8
                else:
                    config["per_device_train_batch_size"] = 1
                    config["gradient_accumulation_steps"] = 32
                    
        return config

def detect_hardware() -> HardwareProfile:
    """
    Detects the current runtime hardware environment.
    Checks for Kaggle environments, CUDA availability, CPU counts, and system RAM.
    """
    # 1. Environment check (Kaggle detection)
    is_kaggle = os.path.exists("/kaggle/") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ
    is_local = not is_kaggle
    
    # 2. CPU and RAM detection
    cpu_cores = psutil.cpu_count(logical=False) or 1
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    
    # 3. GPU detection
    gpu_count = 0
    gpu_vram_gb = 0.0
    gpu_name = "None"
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        # Approximate VRAM from device properties
        total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
        gpu_vram_gb = round(total_memory_bytes / (1024 ** 3), 1)
        
    profile = HardwareProfile(
        gpu_count=gpu_count,
        gpu_vram_gb=gpu_vram_gb,
        gpu_name=gpu_name,
        cpu_cores=cpu_cores,
        ram_gb=ram_gb,
        is_kaggle=is_kaggle,
        is_local=is_local
    )
    
    logger.info(f"Detected hardware: {gpu_count}x {gpu_name} ({gpu_vram_gb} GB VRAM per GPU)")
    logger.info(f"System details: {cpu_cores} CPU cores, {ram_gb} GB RAM | environment: {'Kaggle' if is_kaggle else 'Local'}")
    
    return profile
