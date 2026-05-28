# CRAFT — SLM Reasoning Pipeline & Interactive Dashboard
### Samsung EnnovateX Hackathon Phase 2 Delivery | Team Aurvion

This directory contains the fully package-ready deployment bundle for the CRAFT (Curriculum-guided Reinforced Adaptive Fine-Tuning) small language model.

## Folder Directory
- `src/inference/` : FastAPI inference server and core llama-cpp-python interface
- `src/ui/`        : Premium dark mode interactive reasoning dashboard
- `docs/`          : Exhaustive design, installation, training, and architecture guides
- `craft_output/`  : Output directory containing the quantized GGUF model files
- `scripts/`       : Environment setups and model downloading automations

## Local Execution Instructions
To start the model locally on standard laptop hardware (CPU/RTX GPU):
1. Make sure you have python 3.10+ installed.
2. Run the launch shell script:
   ```bash
   ./launch.sh
   ```
3. Open `src/ui/index.html` in any web browser and explore the interactive multi-step reasoning capabilities.