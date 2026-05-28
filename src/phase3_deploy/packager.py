import os
import shutil
import tarfile
import argparse
from loguru import logger

class DeliveryPackager:
    """
    Packages the final CRAFT application into a single consolidated, 
    deployable delivery bundle.
    """
    def __init__(self, delivery_dir="craft_delivery"):
        self.delivery_dir = delivery_dir
        os.makedirs(self.delivery_dir, exist_ok=True)

    def package_all(self, gguf_path: str, output_archive="craft_delivery_bundle.tar.gz"):
        logger.info(f"Starting packaging process into directory: {self.delivery_dir}...")
        
        # 1. Recreate directory structure
        dirs = [
            "src/inference",
            "src/ui",
            "docs",
            "craft_output"
        ]
        for d in dirs:
            os.makedirs(os.path.join(self.delivery_dir, d), exist_ok=True)

        # 2. Copy source files
        # Copy inference server
        inference_src = "src/inference"
        if os.path.exists(inference_src):
            for file in os.listdir(inference_src):
                if file.endswith(".py"):
                    shutil.copy(os.path.join(inference_src, file), os.path.join(self.delivery_dir, "src/inference", file))
                    
        # Copy UI files
        ui_src = "src/ui"
        if os.path.exists(ui_src):
            for file in os.listdir(ui_src):
                if file.endswith((".html", ".css", ".js", ".png", ".jpg")):
                    shutil.copy(os.path.join(ui_src, file), os.path.join(self.delivery_dir, "src/ui", file))
                    
        # Copy Docs
        docs_src = "docs"
        if os.path.exists(docs_src):
            for file in os.listdir(docs_src):
                if file.endswith(".md"):
                    shutil.copy(os.path.join(docs_src, file), os.path.join(self.delivery_dir, "docs", file))

        # 3. Copy GGUF Model (if exists)
        if gguf_path and os.path.exists(gguf_path):
            logger.info(f"Copying GGUF model from {gguf_path} to deployment bundle...")
            dest_model_path = os.path.join(self.delivery_dir, "craft_output", os.path.basename(gguf_path))
            shutil.copy(gguf_path, dest_model_path)
        else:
            logger.warning(f"GGUF model not found at {gguf_path}. Bundle will be created with dummy/empty model placeholder.")
            # Touch placeholder
            with open(os.path.join(self.delivery_dir, "craft_output", "craft_model_Q4_K_M.gguf.placeholder"), "w") as f:
                f.write("Download model from HuggingFace Aurvion/CRAFT-Phi3-Mini and place it here.")

        # 4. Copy basic scripts
        scripts_to_copy = ["scripts/download_model.sh", "scripts/setup.sh"]
        os.makedirs(os.path.join(self.delivery_dir, "scripts"), exist_ok=True)
        for s in scripts_to_copy:
            if os.path.exists(s):
                shutil.copy(s, os.path.join(self.delivery_dir, s))
                os.chmod(os.path.join(self.delivery_dir, s), 0o755)

        # 5. Create launch scripts in root of bundle
        launch_sh_content = """#!/bin/bash
echo "=== Starting CRAFT Inference Server & Interactive Dashboard ==="
echo "Installing minimal python dependencies..."
pip install -r requirements_inference.txt

echo "Starting FastAPI Backend Server on port 8000..."
python -m uvicorn src.inference.inference_server:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

echo "FastAPI Server launched with PID: $SERVER_PID"
echo "--------------------------------------------------------"
echo "Dashboard UI is now active!"
echo "Please open src/ui/index.html in your web browser."
echo "--------------------------------------------------------"
echo "Press Ctrl+C to terminate the server."
wait $SERVER_PID
"""
        with open(os.path.join(self.delivery_dir, "launch.sh"), "w") as f:
            f.write(launch_sh_content.strip())
        os.chmod(os.path.join(self.delivery_dir, "launch.sh"), 0o755)

        # 6. Copy requirements
        if os.path.exists("requirements_inference.txt"):
            shutil.copy("requirements_inference.txt", os.path.join(self.delivery_dir, "requirements_inference.txt"))

        # 7. Create README in root of bundle
        readme_content = """# CRAFT — SLM Reasoning Pipeline & Interactive Dashboard
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
"""
        with open(os.path.join(self.delivery_dir, "README.md"), "w") as f:
            f.write(readme_content.strip())

        # 8. Compress to tar.gz
        logger.info(f"Compressing deployment bundle into archive {output_archive}...")
        with tarfile.open(output_archive, "w:gz") as tar:
            tar.add(self.delivery_dir, arcname=os.path.basename(self.delivery_dir))
            
        logger.info(f"Deployment packaging completed successfully! Bundle archive: {output_archive}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CRAFT Deployment Packager")
    parser.add_argument("--gguf", type=str, default="craft_output/craft_model_Q4_K_M.gguf", help="Path to quantized GGUF model")
    parser.add_argument("--dir", type=str, default="craft_delivery", help="Output directory to package files into")
    parser.add_argument("--archive", type=str, default="craft_delivery_bundle.tar.gz", help="Output compressed archive file path")
    args = parser.parse_args()
    
    packager = DeliveryPackager(delivery_dir=args.dir)
    packager.package_all(args.gguf, args.archive)

if __name__ == "__main__":
    main()
