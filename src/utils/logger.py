import os
import sys
import json
from loguru import logger

def setup_logger(log_dir="logs", log_level="INFO"):
    """
    Sets up the loguru logger to log to console (using rich formatting if available)
    and to a file (in both human-readable and structured JSON format).
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Console handler (standard formatting)
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:7}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )
    
    # Human-readable file handler
    logger.add(
        os.path.join(log_dir, "training.log"),
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:7} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation="10 MB",
        retention="10 days"
    )
    
    # Structured JSON metrics handler
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
    # Log with extra flag 'metric' so it filters only to metrics.jsonl
    logger.bind(metric=True).trace(json.dumps(payload))
    
    # Also log a human-readable summary of metrics to standard log
    metrics_str = ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics_dict.items()])
    logger.info(f"[Step {step}] {phase.upper()} {component} Metrics: {metrics_str}")
