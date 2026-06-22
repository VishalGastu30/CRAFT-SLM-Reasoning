# Salient Features

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

## 1. Deterministic, Noise-Free Reward Signal
Traditional reinforcement learning for language models often relies on large, computationally heavy "critic" models or API-based LLM-as-a-judge scorers. These approaches introduce noise, delay, and bias.

CRAFT solves this by using a fully programmatic, deterministic scoring module (**Component A**):
* **AST parsing:** Mathematics equations are extracted and evaluated safely using Python's Abstract Syntax Trees (`ast`), yielding exact truth values.
* **Logical Progress Verifier:** Measures the overlap of numerical entities across consecutive steps to score rational progression.
* **Formatting Incentives:** Auto-scans response boundaries to reward XML block configurations (`<thought>` and `<answer>`).

This guarantees consistent, bias-free rewards with zero scoring latency.

---

## 2. Self-Supervised Step-Level Learning
Most policy optimization pipelines (such as standard DPO or PPO) align outputs at the full-sequence level. However, a model might construct a long, correct reasoning chain and fail only at the final step, leading to the entire sequence being penalized.

CRAFT introduces targeted step-level contrastive alignment (**Component B**):
* **Deviational Mining:** Automatically compares candidates within a GRPO generation group to isolate the exact step index where correctness diverges.
* **Step-Level DPO:** Constructs precise preference pairs (e.g., chosen vs. rejected steps under the same prefix context) and updates model policy weights specifically for the faulty reasoning steps.

This isolates reasoning errors, preventing the model from forgetting valid intermediate reasoning paths.

---

## 3. Live Adaptive Curriculum
Training small models on hard mathematical tasks from the beginning often leads to gradient collapse because the model fails to generate any correct completions.

CRAFT implements a dynamic curriculum engine (**Component C**):
* **Capability Probing:** Pre-maps question difficulties using pass@k evaluations during Phase 0.
* **Learning Zone Filtering:** Constrains training to questions within the active difficulty range `[min_difficulty, max_difficulty]`, avoiding both trivially simple tasks and impossibly complex tasks.
* **Accuracy-Driven Expansion:** Expands difficulty bounds by checking rolling model accuracy, pushing the SLM's reasoning boundary outwards in response to training success.
* **Safety Floor Safeguards:** Ensures that the active training difficulty range remains within stable bounds, preventing death-spiral collapses.

---

## 4. Model-Agnostic Training Framework
CRAFT is not bound to a specific Small Language Model architecture. The codebase features modular configurations and hardware-aware adaptors allowing easy scaling:
* **Phi-3-Mini (3.8B):** High-efficiency deployment target optimized for mobile and edge platforms.
* **Qwen-2.5-7B-Instruct:** Deeper parameter budget yielding stronger logical checks and complex multi-step reasoning capabilities.

System overrides automatically load model-specific target modules, sequence lengths, and templates.

---

## 5. Fully On-Device Deployment
CRAFT includes a built-in Phase 3 compression pipeline:
* **Adapter Merging:** FP16 PEFT/LoRA weights are fused directly back into the base layers.
* **GGUF Quantization:** Outputs high-efficiency, compressed `Q4_K_M` binaries.
* **Local Serving Ready:** Deploys as a lightweight ASGI service (`FastAPI` + `llama-cpp-python`) with zero cloud dependencies, making it suitable for secure, on-device enterprise applications.
