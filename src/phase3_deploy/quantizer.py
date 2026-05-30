import os
import sys
import argparse
import subprocess
from loguru import logger

# ── Consistent GGUF filename used across the entire pipeline ──
GGUF_FILENAME = "craft_phi3_Q4_K_M.gguf"

def run_command(cmd, cwd=None):
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    logger.info(f"Command completed successfully.")
    return result.stdout

def merge_lora(base_model_name_or_path, lora_adapter_path, output_dir):
    """Merges LoRA adapter weights back into the base model."""
    logger.info(f"Merging LoRA adapter {lora_adapter_path} with base model {base_model_name_or_path}...")
    
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=False
    )
    
    # Load LoRA adapter on top
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge weights
    logger.info("Merging LoRA weights into base model...")
    merged_model = model.merge_and_unload()
    
    # Load tokenizer from the BASE model (not the adapter dir)
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path,
        trust_remote_code=False
    )
    
    # Save merged model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Saving merged model to {output_dir}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # FIX: Explicitly ensure tokenizer.model exists (llama.cpp needs it for Phi-3)
    tokenizer_model_path = os.path.join(output_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_model_path):
        logger.info("tokenizer.model not found after save. Downloading explicitly from HuggingFace...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=base_model_name_or_path,
                filename="tokenizer.model",
                local_dir=output_dir
            )
            logger.info("tokenizer.model downloaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to download tokenizer.model: {e}")
    
    # Verify the merged output directory has all needed files
    files = os.listdir(output_dir)
    logger.info(f"Merged model directory contains: {files}")
    assert "model.safetensors" in files or any(f.startswith("model") and f.endswith(".safetensors") for f in files), \
        f"FATAL: No model weights found in {output_dir}! Files: {files}"
    
    logger.info(f"Merged model successfully saved to {output_dir}")

def setup_llama_cpp(llama_dir="llama.cpp"):
    """Clones and compiles llama.cpp if not present."""
    if not os.path.exists(llama_dir):
        logger.info(f"Cloning llama.cpp to {llama_dir}...")
        run_command(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", llama_dir])
    
    # Check if compiled binaries exist
    quantize_bin = os.path.join(llama_dir, "llama-quantize")
    build_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")
    
    if not os.path.exists(quantize_bin) and not os.path.exists(build_bin):
        logger.info("Building llama.cpp...")
        try:
            # Try cmake first (more reliable on Kaggle)
            os.makedirs(os.path.join(llama_dir, "build"), exist_ok=True)
            run_command(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"], cwd=llama_dir)
            run_command(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=llama_dir)
        except Exception as cmake_err:
            logger.warning(f"cmake build failed: {cmake_err}. Trying make...")
            try:
                run_command(["make", "-j"], cwd=llama_dir)
            except Exception as make_err:
                logger.error(f"Both cmake and make failed. cmake: {cmake_err}, make: {make_err}")
                raise RuntimeError("Could not compile llama.cpp. Install cmake or make.")

def find_quantize_binary(llama_dir="llama.cpp"):
    """Find the llama-quantize binary, checking multiple possible locations."""
    candidates = [
        os.path.join(llama_dir, "llama-quantize"),
        os.path.join(llama_dir, "build", "bin", "llama-quantize"),
        os.path.join(llama_dir, "build", "llama-quantize"),
        os.path.join(llama_dir, "quantize"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find llama-quantize binary. Searched: {candidates}"
    )

def convert_to_gguf(merged_hf_dir, gguf_output_path, llama_dir="llama.cpp"):
    """Converts Hugging Face model directory to GGUF format."""
    setup_llama_cpp(llama_dir)
    
    # Locate convert script
    convert_script = os.path.join(llama_dir, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        convert_script = os.path.join(llama_dir, "convert.py")
        if not os.path.exists(convert_script):
            raise FileNotFoundError(f"Could not find GGUF convert script in {llama_dir}")
    
    # Install llama.cpp requirements first
    req_file = os.path.join(llama_dir, "requirements.txt")
    if os.path.exists(req_file):
        logger.info("Installing llama.cpp python dependencies...")
        run_command([sys.executable, "-m", "pip", "install", "-q", "-r", req_file])
    
    # Make sure output directory exists
    os.makedirs(os.path.dirname(gguf_output_path) or ".", exist_ok=True)
    
    logger.info(f"Converting HF model {merged_hf_dir} to GGUF...")
    run_command([sys.executable, convert_script, merged_hf_dir, "--outfile", gguf_output_path])
    
    # Verify the file was created
    assert os.path.exists(gguf_output_path), f"FATAL: GGUF file was not created at {gguf_output_path}!"
    size_mb = os.path.getsize(gguf_output_path) / (1024 * 1024)
    logger.info(f"HF model successfully converted to: {gguf_output_path} ({size_mb:.0f} MB)")

def quantize_gguf(input_gguf_path, quantized_output_path, target_quantization="Q4_K_M", llama_dir="llama.cpp"):
    """Quantizes a GGUF model to a lower bit precision (e.g. Q4_K_M)."""
    quantize_bin = find_quantize_binary(llama_dir)
    
    os.makedirs(os.path.dirname(quantized_output_path) or ".", exist_ok=True)
    
    logger.info(f"Quantizing GGUF model {input_gguf_path} to {target_quantization}...")
    run_command([quantize_bin, input_gguf_path, quantized_output_path, target_quantization])
    
    # Verify
    assert os.path.exists(quantized_output_path), f"FATAL: Quantized GGUF not created at {quantized_output_path}!"
    size_mb = os.path.getsize(quantized_output_path) / (1024 * 1024)
    logger.info(f"Model quantized successfully: {quantized_output_path} ({size_mb:.0f} MB)")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CRAFT Model Quantizer via llama.cpp")
    parser.add_argument("--base-model", type=str, required=True, help="Path/name of base model")
    parser.add_argument("--lora-adapter", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--merged-output", type=str, default="craft_output/merged", help="Directory to save merged HF model")
    parser.add_argument("--gguf-f16", type=str, default="craft_output/craft_phi3_f16.gguf", help="Output path for FP16 GGUF")
    parser.add_argument("--gguf-quantized", type=str, default=f"craft_output/{GGUF_FILENAME}", help="Output path for quantized GGUF")
    parser.add_argument("--quantization", type=str, default="Q4_K_M", help="Quantization format (default: Q4_K_M)")
    parser.add_argument("--llama-dir", type=str, default="llama.cpp", help="Directory of llama.cpp repository")
    args = parser.parse_args()
    
    os.makedirs("craft_output", exist_ok=True)
    
    # 1. Merge LoRA weights into base model
    merge_lora(args.base_model, args.lora_adapter, args.merged_output)
    
    # 2. Convert HF to F16 GGUF
    convert_to_gguf(args.merged_output, args.gguf_f16, args.llama_dir)
    
    # 3. Quantize to Q4_K_M
    quantize_gguf(args.gguf_f16, args.gguf_quantized, args.quantization, args.llama_dir)
    
    logger.info("=" * 60)
    logger.info("CRAFT Model Quantization Pipeline Completed Successfully!")
    logger.info(f"Final quantized model: {args.gguf_quantized}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
