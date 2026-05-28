#!/bin/bash
# CRAFT Environment Setup Script
set -e

echo "=== CRAFT Environment Setup ==="

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "Creating conda environment 'craft'..."
    conda env create -f environment.yml || conda env update -f environment.yml
    echo "Conda environment 'craft' created/updated."
    echo "To activate, run: conda activate craft"
else
    echo "Conda not found. Installing via pip..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Virtual environment 'venv' created and dependencies installed."
fi

# Detect CUDA and build llama-cpp-python accordingly
echo "Configuring llama-cpp-python..."
if [ -n "$CUDA_VISIBLE_DEVICES" ] || command -v nvcc &> /dev/null || nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing llama-cpp-python with CUDA support..."
    CMAKE_ARGS="-GGPU_SUPPORT=on -DLLAMA_CUDA=on" pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir
else
    echo "No CUDA detected. Installing llama-cpp-python for CPU only..."
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
fi

echo "=== Setup Completed Successfully ==="
