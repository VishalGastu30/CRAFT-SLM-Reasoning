# Implementation Details

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

## Code Module Index

The primary codebase is divided into clear directories within each model's framework folder (e.g., `CRAFT-SLM-Reasoning-Framework-Qwen25/craft/`):

| File/Directory Path | Purpose | Key Implementation Choice |
| :--- | :--- | :--- |
| [`craft/cli.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/cli.py) | Central CLI controller exposing commands. | Integrates all sub-phases (doctor, probe, sft, rl, quantize, etc.) into a unified executable. |
| [`craft/phase0_probe/probe.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase0_probe/probe.py) | Executes Phase 0 capability probing. | Employs pass@k sampling to estimate capability instead of a single greedy response. |
| [`craft/phase1_sft/train_sft.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase1_sft/train_sft.py) | Phase 1 SFT format training warmup. | Loads model in 4-bit and uses parameter-efficient LoRA adapters (QLoRA) to save VRAM. |
| [`craft/phase2_rl/craft_rl_loop.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase2_rl/craft_rl_loop.py) | Main RL alignment trainer. | Implements a dual-objective reinforcement optimization step combining GRPO and Step DPO. |
| [`craft/phase2_rl/component_a/execution_verifier.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase2_rl/component_a/execution_verifier.py) | Programmatic math/step verifier. | Evaluates mathematical equations using safe Python AST parsing (`ast.literal_eval`) to prevent code injection. |
| [`craft/phase2_rl/component_b/dpo_trainer.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase2_rl/component_b/dpo_trainer.py) | Computes step DPO loss on preference pairs. | Slices logits selectively for targeted steps to compute precise step-level policy updates. |
| [`craft/phase2_rl/component_c/curriculum_engine.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase2_rl/component_c/curriculum_engine.py) | Dynamic difficulty curriculum filtering. | Filters the question list dynamically based on accuracy-driven expansion limits. |
| [`craft/phase3_deploy/quantizer.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/phase3_deploy/quantizer.py) | Weight merger and GGUF quantizer. | Merges PEFT adapters at 16-bit precision, then calls local llama.cpp tools to output GGUF binaries. |
| [`craft/utils/hardware_detector.py`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/craft/utils/hardware_detector.py) | GPU availability detector. | Automatically overrides batch sizes and gradient accumulation based on local VRAM profile. |

---

## Hyperparameters Configuration

The training hyperparameters are declared in model-specific YAML configurations. The values used during optimization are detailed below:

### 1. Supervised Fine-Tuning (SFT) Stage

| Parameter | Default (`default.yaml`) | Phi-3-Mini (`phi3_mini.yaml`) | Qwen-2.5-7B (`qwen25_7b.yaml`) |
| :--- | :--- | :--- | :--- |
| **Base Model Name** | *None* | `microsoft/Phi-3-mini-4k-instruct` | `Qwen/Qwen2.5-7B-Instruct` |
| **Max Sequence Length**| `1024` | `1024` | `2048` |
| **Learning Rate** | `2e-4` | `2e-4` | `1e-4` |
| **Training Epochs** | `3` | `1` | `3` |
| **LoRA Rank ($r$)** | `64` | `64` | `64` |
| **LoRA Alpha ($\alpha$)** | `128` | `128` | `128` |
| **LoRA Dropout** | `0.05` | `0.05` | `0.05` |
| **Batch Size (Kaggle 2xT4)**| `2` | `2` (accumulation = 8) | `8` (accumulation = 2) |
| **Batch Size (Local 2x24GB)**| `4` | `4` (accumulation = 4) | `8` (accumulation = 2) |
| **LoRA Target Modules** | *None* | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_up_proj`, `down_proj` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

### 2. Reinforcement Learning (RL) Alignment Stage

| Parameter | Default (`default.yaml`) | Phi-3-Mini (`phi3_mini.yaml`) | Qwen-2.5-7B (`qwen25_7b.yaml`) |
| :--- | :--- | :--- | :--- |
| **Total Steps** | `500` | `500` | `500` |
| **Learning Rate** | `5e-6` | `5e-6` | `3e-6` |
| **GRPO Group Size ($G$)**| `8` | `4` | `4` |
| **Max New Tokens** | `256` | `256` | `512` |
| **Temperature ($T$)** | `0.8` | `0.9` | `0.8` |
| **KL Divergence Beta** | `0.04` | `0.04` (warmup = 20) | `0.04` (warmup = 20) |
| **KL Target Divergence** | `0.1` | `0.1` | `0.1` |
| **DPO Loss Beta** | `0.1` | `0.1` | `0.1` |
| **DPO Step Activation** | `50` | `50` | `50` |
| **Batch Size (Kaggle 2xT4)**| `2` | `2` (accumulation = 8) | `1` (accumulation = 16) |
| **Batch Size (Local 2x24GB)**| `4` | `4` (accumulation = 4) | `2` (accumulation = 8) |

### 3. Curriculum parameters (`default.yaml`)
* **Initial Range:** `[0.4, 0.7]`
* **Window Size:** `50`
* **Stability Threshold:** `0.7`
* **Update Frequency:** `100` (Step frequency to re-measure difficulty map)

---

## Known Constraints & Edge Cases Handled

### 1. DPO Logprob Slicing Offset Correctness
Step DPO calculation evaluates conditional probabilities of target reasoning steps. The token boundaries in `StepDPOTrainer.compute_step_logps` were corrected to slide from `prompt_len - 1` rather than `prompt_len`. Slicing at `prompt_len - 1` includes the first token of the target step (as PyTorch causal language modeling shifting labels offsets target indices by 1), resolving an off-by-one mismatch that degraded gradient signals.

### 2. MMLU Baseline Numeric Formatting Correction
During baseline checks, MMLU prompts evaluate choices marked as `'A'..'D'`. Base models occasionally output numeric strings (`'1'`, `'2'`, etc.) instead of letters. The evaluator implements dataset-specific mapping within the validation loop:
```python
# Mapping baseline numeric outputs to matching letters
if dataset == "mmlu" and pred_answer in ["1", "2", "3", "4"]:
    pred_answer = ["A", "B", "C", "D"][int(pred_answer) - 1]
```
This post-processing prevents false-negatives on baseline tests, providing a scientifically sound baseline.

### 3. StrategyQA True/False Mapping Correction
StrategyQA answers are binary `yes`/`no` values. Baseline model responses frequently yield `true`/`false` or `1`/`0`. The evaluator intercepts and maps these outputs:
```python
# Normalizing boolean values for StrategyQA
if dataset == "strategyqa":
    if pred_answer in ["true", "1"]:
        pred_answer = "yes"
    elif pred_answer in ["false", "0"]:
        pred_answer = "no"
```

### 4. Curriculum "Death Spiral" Collapse Safeguard
To prevent the curriculum window from collapsing to 0.0 when training on hard tasks, the following constraints were implemented:
* Set `MIN_DIFFICULTY_FLOOR_COLLAPSE = 0.35` to ensure the curriculum never drops below this limit.
* No-oped `collapse_temporarily` to ensure the curriculum expands rather than shrinks.
* Added a low-accuracy trigger: if success rates stay below $0.5$ for 100 steps, difficulty bounds expand by $0.05$ automatically to introduce simpler questions and restart optimization.
