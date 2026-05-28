import os
import sys
import argparse
import subprocess
from loguru import logger

def run_command(cmd, cwd=None):
    logger.info(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with code {result.returncode}!")
        logger.error(f"STDOUT:\n{result.stdout}")
        logger.error(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    logger.info(f"Command completed successfully.")
    return result.stdout

def merge_lora(base_model_name_or_path, lora_adapter_path, output_dir):
    """Merges LoRA adapter weights back into the base model."""
    logger.info(f"Merging LoRA adapter {lora_adapter_path} with base model {base_model_name_or_path}...")
    
    # Peft and transformers imports are deferred to function call
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load base model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else "cpu",
        trust_remote_code=True
    )
    
    # Load LoRA model
    model = PeftModel.from_pretrained(base_model, lora_adapter_path)
    
    # Merge weights
    merged_model = model.merge_and_unload()
    
    # Save merged model and tokenizer
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Merged model successfully saved to {output_dir}")

def setup_llama_cpp(llama_dir="llama.cpp"):
    """Clones and compiles llama.cpp if not present."""
    if not os.path.exists(llama_dir):
        logger.info(f"Cloning llama.cpp to {llama_dir}...")
        run_command(["git", "clone", "https://github.com/ggerganov/llama.cpp.git", llama_dir])
        
    # Check if compiled binaries exist
    quantize_bin = os.path.join(llama_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        logger.info("Building llama.cpp using make...")
        try:
            run_command(["make"], cwd=llama_dir)
        except Exception:
            logger.info("make failed, trying cmake...")
            run_command(["cmake", "-B", "build"], cwd=llama_dir)
            run_command(["cmake", "--build", "build", "--config", "Release"], cwd=llama_dir)

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
    logger.info("Installing llama.cpp python dependencies...")
    run_command(["pip", "install", "-r", os.path.join(llama_dir, "requirements.txt")])
    
    logger.info(f"Converting HF model {merged_hf_dir} to GGUF...")
    run_command([sys.executable, convert_script, merged_hf_dir, "--outfile", gguf_output_path])
    logger.info(f"HF model successfully converted to: {gguf_output_path}")

def quantize_gguf(input_gguf_path, quantized_output_path, target_quantization="Q4_K_M", llama_dir="llama.cpp"):
    """Quantizes a GGUF model to a lower bit precision (e.g. Q4_K_M)."""
    # Locate quantize binary
    quantize_bin = os.path.join(llama_dir, "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_dir, "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_bin):
            quantize_bin = os.path.join(llama_dir, "quantize")
            if not os.path.exists(quantize_bin):
                raise FileNotFoundError(f"Could not find quantize binary in {llama_dir}")
                
    logger.info(f"Quantizing GGUF model {input_gguf_path} to {target_quantization}...")
    run_command([quantize_bin, input_gguf_path, quantized_output_path, target_quantization])
    logger.info(f"Model quantized successfully saved to: {quantized_output_path}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CRAFT Model Quantizer via llama.cpp")
    parser.add_argument("--base-model", type=str, required=True, help="Path/name of base model")
    parser.add_argument("--lora-adapter", type=str, required=True, help="Path to LoRA adapter weights")
    parser.add_argument("--merged-output", type=str, default="craft_output/merged_hf", help="Directory to save merged HF model")
    parser.add_argument("--gguf-f16", type=str, default="craft_output/craft_model_f16.gguf", help="Output path for FP16 GGUF")
    parser.add_argument("--gguf-quantized", type=str, default="craft_output/craft_model_Q4_K_M.gguf", help="Output path for quantized GGUF")
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
    
    logger.info("CRAFT Model Quantization Pipeline Completed Successfully!")

if __name__ == "__main__":
    main()
