# CRAFT: Curriculum-guided Reinforced Adaptive Fine-Tuning

CRAFT is a framework designed to optimize step-by-step reasoning capabilities in Small Language Models (SLMs). Developed for Samsung EnnovateX 2026, it aligns architectures like Phi-3-Mini (3.8B) and Qwen-2.5-7B on complex mathematical, logical, and multi-domain benchmarks.

CRAFT leverages three core components:
1. **Component A (Deterministic Rewards):** Structural and execution-level math checkers (using Python AST evaluation) to score reasoning paths with zero critic latency.
2. **Component B (Step-level Learning):** Deviances in reasoning steps are isolated within candidate groups, allowing direct preference optimization (DPO) targeted specifically to the faulty reasoning step.
3. **Component C (Live Curriculum):** Filters questions dynamically based on accuracy-driven thresholds, expanding the model's reasoning boundary step by step.

---

## Technical Documentation Guide

Below is the structured technical documentation for the CRAFT framework:

* 📚 **[Technical Stack](docs/technical_stack.md)**
  * A detailed index of the core libraries, versions, and frameworks (e.g. `torch`, `peft`, `trl`, `llama-cpp-python`) powering training and serving.
* 🏗️ **[Architecture Overview](docs/architecture.md)**
  * A comprehensive breakdown of the four execution phases (Probing, SFT warmup, Reinforcement Learning, Deployment) and the mathematical formulations behind Components A, B, and C.
* ⚙️ **[Implementation Details](docs/implementation_details.md)**
  * Design decisions, directory structures, hyperparameter configuration tables (SFT & RL), and specific edge cases resolved in code.
* 💻 **[Installation Guide](docs/installation.md)**
  * Hardware prerequisites, system dependencies setup, and command-line verification scripts (`python -m craft doctor`).
* 📖 **[User Guide](docs/user_guide.md)**
  * How to run the framework CLI commands, host the FastAPI inference server, launch the local Web UI, and format API requests.
* ⚡ **[Salient Features](docs/features.md)**
  * The five core value-propositions defining CRAFT (Deterministic reward signals, Step-level preference pairs, Adaptive curriculum, Model-agnostic design, and On-device GGUF deployment).
* 📊 **[Benchmark Results](docs/benchmark_results.md)**
  * Verified accuracy comparison sweeps comparing baseline models against SFT and CRAFT adapters on GSM8K, StrategyQA, and MMLU.
* 🔌 **[API Reference](docs/api_reference.md)**
  * OpenAPI reference schemas, requests, response layouts, and cURL commands for the FastAPI deployment endpoints.
* 🤖 **[Agentic AI Development (ax.md)](docs/ax.md)**
  * Detailed logs of the agentic AI workflow, iterative code modifications, and optimizations implemented by the coding assistant during framework development.

---

## Quick Start Sequence

### 1. Diagnose Setup suitability
```bash
python -m craft doctor
```

### 2. Run the Full Alignment Pipeline
```bash
python -m craft run-all
```
This executes Phase 0 difficulty mapping, Phase 1 SFT warmup training, Phase 2 Reinforcement Learning optimization, and Phase 3 FP16 weight merging, outputting GGUF binaries.

### 3. Deploy the serving endpoints
```bash
python -m craft deploy
```
This starts the local FastAPI server. Open the Web Dashboard by visiting `craft_delivery/src/ui/index.html` in your web browser.
