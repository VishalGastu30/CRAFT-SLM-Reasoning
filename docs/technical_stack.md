# Technical Stack

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

## Language & Runtime
* **Programming Language:** Python
* **Runtime Version Requirement:** `python >= 3.9`

---

## Core Dependencies (from `pyproject.toml`)

| Library | Version | Purpose in CRAFT | Official OSS Link |
| :--- | :--- | :--- | :--- |
| **torch** | `>=2.0.0` | Core tensor math library, backpropagation engine, and GPU computation. | [GitHub - pytorch/pytorch](https://github.com/pytorch/pytorch) |
| **transformers** | `>=4.35.0` | Model loading, configuration interfaces, tokenizer processing, and token generation. | [GitHub - huggingface/transformers](https://github.com/huggingface/transformers) |
| **datasets** | `>=2.14.0` | Loading, shuffling, preprocessing, and structuring training and evaluation datasets from HuggingFace. | [GitHub - huggingface/datasets](https://github.com/huggingface/datasets) |
| **accelerate** | `>=0.24.0` | Multi-GPU training orchestration, gradient accumulation, and device-placement management. | [GitHub - huggingface/accelerate](https://github.com/huggingface/accelerate) |
| **peft** | `>=0.6.0` | Parameter-Efficient Fine-Tuning. Orchestrates LoRA adapter injection and adapter parameters management. | [GitHub - huggingface/peft](https://github.com/huggingface/peft) |
| **trl** | `>=0.7.0` | Transformer Reinforcement Learning library. Utilized for SFT baseline warm-up trainers. | [GitHub - huggingface/trl](https://github.com/huggingface/trl) |
| **bitsandbytes** | `>=0.41.0` | 4-bit and 8-bit optimizer and model weights quantization (QLoRA) to enable training large SLMs on consumer GPUs. | [GitHub - TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes) |
| **fastapi** | `>=0.104.0` | Asynchronous REST API framework powering the deployed inference server. | [GitHub - fastapi/fastapi](https://github.com/fastapi/fastapi) |
| **uvicorn** | `>=0.24.0` | Lightweight, high-performance ASGI server to run the FastAPI deployment endpoint. | [GitHub - encode/uvicorn](https://github.com/encode/uvicorn) |
| **llama-cpp-python**| `>=0.3.0` | Python bindings for llama.cpp, allowing memory-efficient CPU/GPU loading and execution of GGUF-quantized models. | [GitHub - abetlen/llama-cpp-python](https://github.com/abetlen/llama-cpp-python) |
| **loguru** | `>=0.7.0` | Thread-safe, colorized, structured logging mechanism for training loop monitoring and diagnostics. | [GitHub - Delgan/loguru](https://github.com/Delgan/loguru) |
| **pyyaml** | `>=6.0` | Configuration loading and parsing of project YAML config files. | [GitHub - yaml/pyyaml](https://github.com/yaml/pyyaml) |

---

## Framework Architecture Stack

### 1. Training Framework
* **Warmup Stage (SFT):** Powered by HuggingFace `trl.SFTTrainer` to train a base model adapter on step-by-step reasoning structures.
* **Reinforcement Learning Loop (GRPO + DPO):** Built as a custom PyTorch-based training loop. Group Relative Policy Optimization (GRPO) advantages are computed mathematically without a critic model. Step-level preferences are trained using a custom `StepDPOTrainer` with cross-entropy alignment losses.

### 2. Quantization & Deployment Stack
* **Quantization Format:** GGUF (`Q4_K_M` 4-bit quantization layout).
* **Merging Pipeline:** Merges PEFT/LoRA adapter weights back into base model configurations using HuggingFace `peft` and saves FP16 model checkpoints.
* **GGUF Converter:** Integrates with local `llama.cpp` CLI scripts (e.g., `convert_hf_to_gguf.py` and `llama-quantize`) to transform and compress models.

### 3. Serving & User Interface Stack
* **Web Server:** FastAPI serving backend. Re-parses reasoning chains and formats step-by-step breakdowns into structured JSON.
* **User Interface:** Vanilla HTML5, CSS3 (featuring Sleek Dark Glassmorphism, tailored gradients, and responsive layouts), and Client-Side Javascript (AJAX fetch client with real-time stream simulation and side-by-side model comparison view).

### 4. Evaluation Suite
* **Evaluator Script:** A custom Python script (`evaluator.py`) validating model accuracy and format compliance (detecting `<thought>`, `<answer>`, and correct step-wise formatting) across MMLU, GSM8K, and StrategyQA datasets.
