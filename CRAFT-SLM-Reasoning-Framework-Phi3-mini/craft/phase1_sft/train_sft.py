import os
import argparse
import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer
from loguru import logger

from craft.utils.logger import setup_logger
from craft.utils.hardware_detector import detect_hardware
from craft.utils.checkpoint_manager import CheckpointManager
from craft.config import load_config
from .sft_config import SFTConfig
from .data_loader import create_training_dataset

torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])

_original_load = torch.load

def _patched_load(f, map_location=None, pickle_module=None, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(f, map_location=map_location, pickle_module=pickle_module, **kwargs)

torch.load = _patched_load


class CheckpointCallback(TrainerCallback):
    def __init__(self, manager, keep_n=3):
        self.manager = manager
        self.keep_n = keep_n

    def on_save(self, args, state, control, **kwargs):
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
    setup_logger()
    logger.info("Initializing SFT Warmup (Phase 1)...")

    raw_config = load_config(config_name, hardware_name)
    hw_profile = detect_hardware()

    sft_cfg = SFTConfig(
        model_name_or_path=raw_config["model"]["name"],
        max_seq_length=raw_config["model"]["max_seq_length"],
        gsm8k_fraction=raw_config.get("sft", {}).get("gsm8k_fraction", 0.40),
        aqua_fraction=raw_config.get("sft", {}).get("aqua_fraction", 0.20),
        strategyqa_fraction=raw_config.get("sft", {}).get("strategyqa_fraction", 0.20),
        mmlu_fraction=raw_config.get("sft", {}).get("mmlu_fraction", 0.20),
        epochs=raw_config.get("sft", {}).get("epochs", 3),
        learning_rate=float(raw_config.get("sft", {}).get("learning_rate", 2e-4)),
        lora_rank=raw_config.get("sft", {}).get("lora_rank", 64),
        lora_alpha=raw_config.get("sft", {}).get("lora_alpha", 128),
        output_dir=output_dir,
    )
    sft_cfg.update_from_profile(hw_profile)

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
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        logger.warning("No CUDA – training on CPU (very slow!).")
        model = AutoModelForCausalLM.from_pretrained(
            sft_cfg.model_name_or_path,
            device_map="cpu",
            trust_remote_code=True
        )

    peft_config = LoraConfig(
        r=sft_cfg.lora_rank,
        lora_alpha=sft_cfg.lora_alpha,
        lora_dropout=sft_cfg.lora_dropout,
        target_modules=sft_cfg.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )

    train_dataset = create_training_dataset(
        tokenizer=tokenizer,
        gsm8k_fraction=sft_cfg.gsm8k_fraction,
        aqua_fraction=sft_cfg.aqua_fraction,
        strategyqa_fraction=sft_cfg.strategyqa_fraction,
        mmlu_fraction=sft_cfg.mmlu_fraction,
        seed=sft_cfg.train_split_seed
    )

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

    manager = CheckpointManager(sft_cfg.output_dir)
    callback = CheckpointCallback(manager)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=sft_cfg.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        callbacks=[callback]
    )

    resume_from_checkpoint = None
    if resume:
        metadata, checkpoint_path = manager.get_latest()
        if checkpoint_path:
            logger.info(f"Resuming from checkpoint: {checkpoint_path}")
            resume_from_checkpoint = checkpoint_path

    logger.info("Starting SFT training (4‑dataset mix)...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    final_model_dir = os.path.join(sft_cfg.output_dir, "final")
    logger.info(f"Saving final SFT model to {final_model_dir}")
    trainer.model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    logger.info("SFT training completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 1: SFT Warmup (all 4 datasets)")
    parser.add_argument("--config", default="phi3_mini")
    parser.add_argument("--hardware", default="kaggle")
    parser.add_argument("--output", default="checkpoints/sft")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    train_sft(
        config_name=args.config,
        hardware_name=args.hardware,
        output_dir=args.output,
        resume=args.resume
    )