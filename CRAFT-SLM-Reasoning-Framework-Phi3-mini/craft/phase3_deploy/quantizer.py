import os
import sys
import argparse
import subprocess
import platform
from loguru import logger

GGUF_FILENAME = "CRAFT_Q4_K_M.gguf"

def get_system_info():
    """Detect operating system and return appropriate configurations."""
    system = platform.system().lower()
    is_windows = system == "windows"
    
    # Binary extensions
    ext = ".exe" if is_windows else ""
    
    # Build commands
    if is_windows:
        build_cmd = ["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"]
        build_cmd2 = ["cmake", "--build", "build", "--config", "Release", "--parallel"]
    else:
        build_cmd = ["make", "-j"]
        build_cmd2 = None
    
    return {
        "is_windows": is_windows,
        "ext": ext,
        "build_cmd": build_cmd,
        "build_cmd2": build_cmd2,
        "python": sys.executable
    }


def run_command(cmd, cwd=None, check=True):
    """Run a command with proper Windows handling."""
    logger.info(f"Running command: {' '.join(cmd)}")
    
    result = subprocess.run(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        shell=False
    )

    if result.stdout:
        logger.info(result.stdout)
    if result.stderr:
        logger.warning(result.stderr)

    if result.returncode != 0 and check:
        logger.error(f"Command failed with code {result.returncode}!")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    
    logger.info("Command completed successfully.")
    return result.stdout


def merge_lora(base_model_name_or_path, lora_adapter_path, output_dir):
    """Merges LoRA adapter weights back into the base model."""
    logger.info(f"Merging LoRA adapter {lora_adapter_path} with base model {base_model_name_or_path}...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False
        )
    except Exception as e:
        logger.error(f"Failed to load base model: {e}")
        logger.info("Trying to load from local cache...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=False,
            local_files_only=True
        )
    
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    merged_model = model.merge_and_unload()
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path, trust_remote_code=False)
    
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Merged model saved to {output_dir}")


def setup_llama_cpp(llama_dir="llama.cpp"):
    """Clones and compiles llama.cpp (Windows compatible)."""
    sys_info = get_system_info()
    is_windows = sys_info["is_windows"]
    ext = sys_info["ext"]
    
    # Check if llama.cpp exists
    if not os.path.exists(llama_dir):
        logger.info(f"Cloning llama.cpp to {llama_dir}...")
        run_command(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp.git", llama_dir])
    
    # Check for pre-built binary (including Release folder)
    quantize_bin = os.path.join(llama_dir, f"llama-quantize{ext}")
    build_bin_release = os.path.join(llama_dir, "build", "bin", "Release", f"llama-quantize{ext}")
    build_bin = os.path.join(llama_dir, "build", "bin", f"llama-quantize{ext}")
    
    if not os.path.exists(quantize_bin) and not os.path.exists(build_bin_release) and not os.path.exists(build_bin):
        logger.info("Building llama.cpp...")
        
        if is_windows:
            # Windows: Use cmake
            try:
                os.makedirs(os.path.join(llama_dir, "build"), exist_ok=True)
                run_command([
                    "cmake",
                    "-B",
                    "build",
                    "-G",
                    "MinGW Makefiles",
                    "-DCMAKE_CXX_FLAGS=-D_WIN32_WINNT=0x0A00",
                    "-DCMAKE_C_FLAGS=-D_WIN32_WINNT=0x0A00"
                ], cwd=llama_dir)

                run_command([
                    "cmake",
                    "--build",
                    "build",
                    "--parallel"
                ], cwd=llama_dir)
            except Exception as e:
                logger.error(f"CMake build failed: {e}")
                logger.info("Please install Visual Studio Build Tools and CMake.")
                logger.info("Or download pre-built llama.cpp from: https://github.com/ggml-org/llama.cpp/releases")
                raise
        else:
            # Linux/Mac: Use make
            try:
                run_command(["make", "-j"], cwd=llama_dir)
            except Exception as e:
                logger.error(f"Make build failed: {e}")
                raise


def find_quantize_binary(llama_dir="llama.cpp"):
    ext = ".exe" if platform.system() == "Windows" else ""

    candidates = [
        os.path.join(llama_dir, "build", "bin", f"llama-quantize{ext}"),
        os.path.join(llama_dir, "build", "bin", "Release", f"llama-quantize{ext}"),
        os.path.join(llama_dir, "build", "bin", f"llama-quantize-cli{ext}"),      # NEW
        os.path.join(llama_dir, "build", "bin", "Release", f"llama-quantize-cli{ext}"),  # NEW
    ]

    for c in candidates:
        if os.path.exists(c):
            logger.info(f"Using quantizer: {c}")
            return c

    raise FileNotFoundError("Could not find llama quantizer.")


def convert_to_gguf(merged_hf_dir, gguf_output_path, llama_dir="llama.cpp"):
    """Converts Hugging Face model directory to GGUF format."""
    setup_llama_cpp(llama_dir)
    
    sys_info = get_system_info()
    python = sys_info["python"]
    
    # Try different convert script names
    convert_scripts = [
        os.path.join(llama_dir, "convert_hf_to_gguf.py"),
        os.path.join(llama_dir, "convert.py"),
        os.path.join(llama_dir, "convert-hf-to-gguf.py"),
    ]
    
    convert_script = None
    for script in convert_scripts:
        if os.path.exists(script):
            convert_script = script
            break
    
    if convert_script is None:
        raise FileNotFoundError(f"Could not find GGUF convert script in {llama_dir}")
    
    # Install dependencies
    req_file = os.path.join(llama_dir, "requirements.txt")
    if os.path.exists(req_file):
        try:
            run_command([python, "-m", "pip", "install", "-q", "-r", req_file])
        except Exception as e:
            logger.warning(f"Failed to install requirements: {e}")
    
    os.makedirs(os.path.dirname(gguf_output_path) or ".", exist_ok=True)
    
    # Run conversion
    cmd = [
        python,
        convert_script,
        merged_hf_dir,
        "--outfile",
        gguf_output_path,
        "--outtype",
        "f16"
    ]
    run_command(cmd)
    
    assert os.path.exists(gguf_output_path), f"GGUF file not created at {gguf_output_path}!"
    size_mb = os.path.getsize(gguf_output_path) / (1024 * 1024)
    logger.info(f"Converted to GGUF: {gguf_output_path} ({size_mb:.0f} MB)")


def quantize_gguf(input_gguf_path, quantized_output_path, target_quantization="Q4_K_M", llama_dir="llama.cpp"):
    """Quantizes a GGUF model to a lower bit precision."""
    quantize_bin = find_quantize_binary(llama_dir)
    
    os.makedirs(os.path.dirname(quantized_output_path) or ".", exist_ok=True)
    
    cmd = [quantize_bin, input_gguf_path, quantized_output_path, target_quantization]
    run_command(cmd)
    
    assert os.path.exists(quantized_output_path), f"Quantized GGUF not created at {quantized_output_path}!"
    size_mb = os.path.getsize(quantized_output_path) / (1024 * 1024)
    logger.info(f"Quantized: {quantized_output_path} ({size_mb:.0f} MB)")


def quantize_model(base_model, lora_adapter, output_path, quantization="Q4_K_M"):
    """Main quantization function."""
    merged_dir = "craft_output/merged_temp"
    f16_gguf = "craft_output/temp_f16.gguf"
    
    try:
        merge_lora(base_model, lora_adapter, merged_dir)
        convert_to_gguf(merged_dir, f16_gguf)
        quantize_gguf(f16_gguf, output_path, quantization)
    finally:
        # Cleanup temp files
        import shutil
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        if os.path.exists(f16_gguf):
            os.remove(f16_gguf)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: CRAFT Model Quantizer")
    parser.add_argument("--base-model", type=str, required=True, help="Base model name or path")
    parser.add_argument("--lora-adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--merged-output", type=str, default="craft_output/merged", help="Output directory for merged model")
    parser.add_argument("--gguf-f16", type=str, default="craft_output/craft_f16.gguf", help="Output path for FP16 GGUF")
    parser.add_argument("--gguf-quantized", type=str, default=f"craft_output/{GGUF_FILENAME}", help="Output path for quantized GGUF")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", help="Quantization type (e.g., Q4_K_M, Q5_K_M, Q6_K)")
    parser.add_argument("--llama-dir", type=str, default="llama.cpp", help="Directory of llama.cpp repository")
    args = parser.parse_args()
    
    os.makedirs("craft_output", exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting CRAFT Model Quantization Pipeline")
    logger.info(f"System: {platform.system()}")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"LoRA Adapter: {args.lora_adapter}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("=" * 60)
    
    try:
        merge_lora(args.base_model, args.lora_adapter, args.merged_output)
        convert_to_gguf(args.merged_output, args.gguf_f16, args.llama_dir)
        quantize_gguf(args.gguf_f16, args.gguf_quantized, args.quantization, args.llama_dir)
        
        logger.info("=" * 60)
        logger.info("✅ Quantization completed successfully!")
        logger.info(f"Final model: {args.gguf_quantized}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"❌ Quantization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()