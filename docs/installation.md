# Installation Guide

## Navigation
- [Main README](../README.md)
- [Technical Stack](technical_stack.md)
- [Architecture](architecture.md)
- [Implementation Details](implementation_details.md)
- [Installation Guide](installation.md)
- [User Guide](user_guide.md)
- [Features](features.md)
- [Benchmark Results](benchmark_results.md)
- [API Reference](api_reference.md)
- [Agentic AI Development (ax.md)](ax.md)

---

## Prerequisites

### 1. Hardware Requirements
* **Training Stage:**
  * **Minimal:** Dual GPU setup with at least 15GB VRAM per GPU (e.g., 2x NVIDIA T4 GPUs in Kaggle).
  * **Recommended:** Dual GPU setup with 24GB VRAM per GPU (e.g., 2x NVIDIA RTX 3090 / 4090 or A5000) for faster training iterations and Qwen-2.5-7B support.
* **Inference Server serving GGUF:**
  * Minimum 8GB system RAM (CPU execution) or any NVIDIA GPU supporting CUDA.

### 2. Runtime Environment
* Python version `python >= 3.9` (Recommended: `python 3.10`).
* Python package installer (`pip`).
* NVIDIA CUDA toolkit compatible with PyTorch (Recommended: `CUDA >= 11.8`).

---

## Step-by-Step Installation

### Step 1: Clone the Repository
Clone the CRAFT repository to your local machine:
```bash
git clone <YOUR_REPO_URL>
cd CRAFT-SLM-Reasoning
```

### Step 2: Install dependencies
Navigate to your desired model's framework folder (e.g., `CRAFT-SLM-Reasoning-Framework-Qwen25`) and install dependencies:
```bash
cd CRAFT-SLM-Reasoning-Framework-Qwen25
pip install -r requirements.txt
```
*Note: If no explicit `requirements.txt` file exists in the folder, install the packages directly using the `pyproject.toml` descriptor:*
```bash
pip install -e .
```
For local developer dependencies (such as `pytest`), run:
```bash
pip install -e .[dev]
```

### Step 3: Install Inference Serving Engine
To launch the quantized GGUF server, install `llama-cpp-python` with CUDA acceleration:
```bash
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python>=0.3.0
```

---

## Verification & Diagnostics

CRAFT includes built-in diagnostic commands to verify your setup, check library versions, and confirm GPU allocations.

### 1. Verify Framework Info
Run the info CLI command to print the workspace paths and current configuration status:
```bash
python -m craft info
```

### 2. Run the Hardware Doctor Suite
Run the doctor suite to scan CUDA devices, check VRAM parameters, and validate dependency alignments:
```bash
python -m craft doctor
```

Successful execution should produce a confirmation report verifying that CUDA is active and all core libraries are correctly configured.

![Successful installation output](screenshots/installation_success.png)
<!-- SCREENSHOT NEEDED: Terminal output showing successful dependency install
     and any verification script passing. -->
