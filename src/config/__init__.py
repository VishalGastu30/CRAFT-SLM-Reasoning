import os
import yaml
from loguru import logger

def load_config(model_type="phi3_mini", hardware="kaggle"):
    """
    Loads and merges configuration YAML files.
    1. Loads default.yaml
    2. Loads model-specific configuration (e.g., phi3_mini.yaml)
    3. Merges hardware profile settings
    """
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Load default
    default_path = os.path.join(config_dir, "default.yaml")
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Default config not found at: {default_path}")
        
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)
        
    # 2. Load model-specific config
    model_path = os.path.join(config_dir, f"{model_type}.yaml")
    if os.path.exists(model_path):
        with open(model_path, "r") as f:
            model_config = yaml.safe_load(f)
            
        # Recursive merge of model_config into config
        for section, values in model_config.items():
            if section in config and isinstance(config[section], dict) and isinstance(values, dict):
                config[section].update(values)
            else:
                config[section] = values
    else:
        logger.warning(f"Model config file {model_type}.yaml not found. Using defaults.")
        
    # 3. Merge hardware profile settings
    hw_profile = config.get("hardware_profiles", {}).get(hardware, {})
    if hw_profile:
        # Override SFT and RL batch size / accumulation steps from hardware profile
        for target in ["sft", "rl"]:
            if target in config:
                if "per_device_train_batch_size" in hw_profile:
                    config[target]["per_device_train_batch_size"] = hw_profile["per_device_train_batch_size"]
                if "gradient_accumulation_steps" in hw_profile:
                    config[target]["gradient_accumulation_steps"] = hw_profile["gradient_accumulation_steps"]
                if "use_gradient_checkpointing" in hw_profile:
                    config[target]["use_gradient_checkpointing"] = hw_profile["use_gradient_checkpointing"]
                    
    logger.info(f"Loaded configuration for model={model_type}, hardware={hardware}")
    return config
