#!/usr/bin/env python3
"""CRAFT - Curriculum-guided Reinforced Adaptive Fine-Tuning for SLMs.

A framework for improving reasoning capabilities in Small Language Models
using Reinforcement Learning with process-based rewards.
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())