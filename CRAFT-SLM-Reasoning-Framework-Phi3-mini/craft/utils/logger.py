import os
import sys
import json
from loguru import logger


def setup_logger(log_dir="logs", log_level="INFO"):
    """
    Sets up the loguru logger to console and file.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    logger.add(
        os.path.join(log_dir, "training.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:7} | {name}:{function} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="10 days"
    )
    
    metrics_log_path = os.path.join(log_dir, "metrics.jsonl")
    logger.add(
        metrics_log_path,
        format="{message}",
        level="TRACE",
        filter=lambda record: "metric" in record["extra"],
        rotation="50 MB"
    )
    
    logger.info(f"Logger initialized. Logging to console and {log_dir}/")
    return logger


def log_metrics(step, metrics_dict, phase="training", component=""):
    """
    Logs metrics to the structured metrics log file in JSON lines format.
    """
    payload = {
        "step": step,
        "phase": phase,
        "component": component,
        **metrics_dict
    }
    logger.bind(metric=True).trace(json.dumps(payload))
    
    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics_dict.items()])
    logger.info(f"[Step {step}] {phase.upper()} {component} Metrics: {metrics_str}")