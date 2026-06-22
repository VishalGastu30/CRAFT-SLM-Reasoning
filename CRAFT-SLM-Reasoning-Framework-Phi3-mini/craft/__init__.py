"""CRAFT Framework for SLM Reasoning Enhancement."""

__version__ = "1.0.0"
__author__ = "Team Aurvion"

from .cli import main
from .config import load_config
from .utils import setup_logger, detect_hardware, CheckpointManager