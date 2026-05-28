import os
import glob
import json
import shutil
from loguru import logger

class CheckpointManager:
    """
    Manages training checkpoints including model states, tokenizers,
    step counters, and custom metadata. Designed to survive Kaggle session limits.
    """
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"CheckpointManager initialized in: {self.checkpoint_dir}")

    def get_checkpoint_path(self, step):
        """Returns the directory path for a specific step checkpoint."""
        return os.path.join(self.checkpoint_dir, f"checkpoint-{step}")

    def save(self, model, tokenizer, step, metadata=None):
        """
        Saves the model, tokenizer, and metadata to a step directory.
        Works for both standard HuggingFace models and PEFT (LoRA) models.
        """
        step_dir = self.get_checkpoint_path(step)
        os.makedirs(step_dir, exist_ok=True)
        
        logger.info(f"Saving checkpoint for step {step} to {step_dir}...")
        
        # Save model (handles PEFT LoRA or standard HF model)
        if hasattr(model, "save_pretrained"):
            model.save_pretrained(step_dir)
        else:
            logger.warning("Model does not have save_pretrained method, skipping weights save.")
            
        # Save tokenizer
        if tokenizer is not None:
            tokenizer.save_pretrained(step_dir)
            
        # Save metadata
        metadata = metadata or {}
        metadata["step"] = step
        metadata_path = os.path.join(step_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Checkpoint for step {step} successfully saved.")

    def list_checkpoints(self):
        """
        Finds all checkpoint directories under checkpoint_dir and returns them
        sorted by step number in ascending order.
        Returns:
            list of dict: [{"step": step, "path": path, "metadata": metadata}]
        """
        checkpoint_dirs = glob.glob(os.path.join(self.checkpoint_dir, "checkpoint-*"))
        checkpoints = []
        
        for d in checkpoint_dirs:
            basename = os.path.basename(d)
            try:
                step = int(basename.split("-")[1])
                metadata_path = os.path.join(d, "metadata.json")
                metadata = {}
                if os.path.exists(metadata_path):
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                checkpoints.append({
                    "step": step,
                    "path": d,
                    "metadata": metadata
                })
            except (IndexError, ValueError) as e:
                logger.warning(f"Malformed checkpoint folder name: {basename}")
                
        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"])
        return checkpoints

    def get_latest(self):
        """
        Returns the metadata dictionary and path of the most recent checkpoint.
        Does not load the model weights.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None, None
        latest = checkpoints[-1]
        return latest["metadata"], latest["path"]

    def cleanup(self, keep_n=3):
        """
        Deletes older checkpoint directories, keeping only the latest N.
        """
        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= keep_n:
            return
            
        to_delete = checkpoints[:-keep_n]
        for cp in to_delete:
            logger.info(f"Cleaning up old checkpoint directory: {cp['path']}")
            try:
                shutil.rmtree(cp["path"])
            except Exception as e:
                logger.error(f"Failed to delete checkpoint at {cp['path']}: {e}")
