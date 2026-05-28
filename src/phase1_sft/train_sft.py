import os
import argparse
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch
from loguru import logger

from src.utils.logger import setup_logger
from src.utils.hardware_detector import detect_hardware
from src.utils.checkpoint_manager import CheckpointManager
from src.config import load_config
from src.phase1_sft.sft_config import SFTConfig
from src.phase1_sft.data_loader import create_training_dataset

class CheckpointCallback(TrainerCallback):
    """Callback to handle custom checkpointing during SFT training."""
    def __init__(self, manager, keep_n=3):
        self.manager = manager
        self.keep_n = keep_n

    def on_save(self, args, state, control, **kwargs):
        # We write metadata into the checkpoint directory
        step = state.global_step
        metadata = {
            "global_step": step,
            "loss": state.log_history[-1].get("loss", 0.0) if state.log_history else 0.0,
            "epoch": state.epoch
        }
        self.manager.save(
            model=kwargs.get("model"),
            tokenizer=kwargs.get("tokenizer"),
            step=step,
            metadata=metadata
        )
        self.manager.cleanup(keep_n=self.keep_n)

def train_sft(config_name="phi3_mini", hardware_name="kaggle", output_dir="checkpoints/sft", resume=False):
    """Main training function for SFT warmup."""
    setup_logger()
    logger.info("Initializing SFT Warmup (Phase 1)...")
    
    # 1. Load config and merge hardware profile
    raw_config = load_config(config_name, hardware_name)
    hw_profile = detect_hardware()
    
    sft_cfg = SFTConfig(
        model_name_or_path=raw_config["model"]["name"],
        max_seq_length=raw_config["model"]["max_seq_length"],
        gsm8k_fraction=raw_config["probe"]["sampling_fractions"].get("gsm8k", 0.6),
        aqua_fraction=raw_config["probe"]["sampling_fractions"].get("aqua_rat", 0.4),
        epochs=raw_config["sft"].get("epochs", 3),
        learning_rate=float(raw_config["sft"].get("learning_rate", 2e-4)),
        lora_rank=raw_config["sft"].get("lora_rank", 64),
        lora_alpha=raw_config["sft"].get("lora_alpha", 128),
        output_dir=output_dir,
    )
    sft_cfg.update_from_profile(hw_profile)
    
    # Initialize checkpoint manager
    manager = CheckpointManager(sft_cfg.output_dir)
    
    # 2. Configure QLoRA quantization settings
    logger.info(f"Loading tokenizer & base model: {sft_cfg.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_cfg.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            sft_cfg.model_name_or_path,
            quantization_config=bnb_config,
            device_map={"": 0},
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.warning("No CUDA device detected. Running model training on CPU (very slow!).")
        model = AutoModelForCausalLM.from_pretrained(
            sft_cfg.model_name_or_path,
            device_map="cpu",
            trust_remote_code=True
        )

    # 3. Configure PEFT (LoRA) settings
    peft_config = LoraConfig(
        r=sft_cfg.lora_rank,
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=sft_cfg.lora_dropout,
        target_modules=sft_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 4. Load training dataset
    train_dataset = create_training_dataset(
        tokenizer=tokenizer,
        gsm_fraction=sft_cfg.gsm8k_fraction,
        aqua_fraction=sft_cfg.aqua_fraction,
        seed=sft_cfg.train_split_seed
    )
    
    # 5. Set up HF Training Arguments
    training_args = TrainingArguments(
        output_dir=sft_cfg.output_dir,
        num_train_epochs=sft_cfg.epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        learning_rate=sft_cfg.learning_rate,
        weight_decay=sft_cfg.weight_decay,
        warmup_ratio=sft_cfg.warmup_ratio,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        logging_dir=sft_cfg.logging_dir,
        logging_steps=sft_cfg.logging_steps,
        save_steps=sft_cfg.save_steps,
        save_total_limit=3,
        fp16=(device == "cuda"),
        optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
        remove_unused_columns=True,
        report_to="none"
    )
    
    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=sft_cfg.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[CheckpointCallback(manager)]
    )
    
    # 7. Start SFT Training
    resume_from_checkpoint = None
    if resume:
        metadata, checkpoint_path = manager.get_latest()
        if checkpoint_path:
            logger.info(f"Resuming SFT training from latest checkpoint: {checkpoint_path}")
            resume_from_checkpoint = checkpoint_path
            
    logger.info("Starting SFT training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # 8. Save final model
    final_model_dir = os.path.join(sft_cfg.output_dir, "final")
    logger.info(f"Saving final SFT Warmup model weights to {final_model_dir}...")
    trainer.model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info("SFT training completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: SFT Warmup")
    parser.add_argument("--config", type=str, default="phi3_mini", help="Model config name")
    parser.add_argument("--hardware", type=str, default="kaggle", help="Hardware profile name")
    parser.add_argument("--output", type=str, default="checkpoints/sft", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if exists")
    args = parser.parse_args()

    train_sft(
        config_name=args.config,
        hardware_name=args.hardware,
        output_dir=args.output,
        resume=args.resume
    )
