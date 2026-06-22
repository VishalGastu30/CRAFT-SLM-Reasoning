# CRAFT: Curriculum-guided Reinforced Adaptive Fine-Tuning

- **Problem Statement Number** - 06
- **Problem Statement Title** - Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning
- **Team name** - Aurvion
- **Team members (Names)** - G.Vishal, E.Kevin Godfrey
- **Institute/College Name** - SRM Institute of Science and Technology, Kattankulathur, Chennai - 603203
- **Final Presentation Google Drive Link** - [https://drive.google.com/file/d/1uG1i7ydhIY5qdk8W0W_jfjacMrjtujzM/view?usp=sharing](https://drive.google.com/file/d/1uG1i7ydhIY5qdk8W0W_jfjacMrjtujzM/view?usp=sharing)
- **Full Submission Demo Video Link** - [https://youtu.be/E09OrWdO9Wk](https://youtu.be/E09OrWdO9Wk)
- **Setup & Result Reproducibility Video Link** - [https://youtu.be/pMNLqnw3hqs](https://youtu.be/pMNLqnw3hqs)
- **Models Used** - [Microsoft Phi-3-Mini-4k-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct), [Qwen-2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
- **Models Published** - [OmnipotentFool/Aurvion](https://huggingface.co/OmnipotentFool/Aurvion)

---

## Project Overview

CRAFT (Curriculum-guided Reinforced Adaptive Fine-Tuning) is an advanced reinforcement learning training framework designed specifically to optimize step-by-step reasoning capabilities in Small Language Models (SLMs) like Microsoft's Phi-3-Mini (3.8B) and Qwen-2.5-7B. SLMs are highly attractive for edge deployment because of their low compute and memory requirements. However, applying RL to them often fails due to three major limitations:
1. **Sparse/Misaligned Step Rewards**: Outcome-only rewards result in correct answers via flawed reasoning steps.
2. **Preference Signal Mismatch**: Coarse step comparisons fail to differentiate between alternative reasoning paths.
3. **KL-Divergence Collapse / Distribution Shift**: Standard RL training destabilizes and collapses quickly under distribution shift.

CRAFT solves these issues through a unified four-phase pipeline: Capability Probing, SFT Warmup, the adaptive CRAFT RL Loop (integrating math step verification, step-level DPO, and a live difficulty curriculum), and GGUF Edge Deployment.

---

## Innovation & Novelty

The novelty of CRAFT lies in three core components:
1. **Component A: Step-Level Symbolic & NLI Execution Verifier**: Instead of checking only the final answer, CRAFT extracts math expressions and evaluates them using a safe Python interpreter. For logical questions, it runs a local NLI model (`DeBERTa-v3-small`) to score the entailment between successive steps, ensuring the reasoning chain is sound with zero critic latency.
2. **Component B: Contrastive Step-Level Reinforcement**: We extract step boundaries from reasoning traces and pair correct steps with common errors to compute a step-level DPO preference loss. Deviances in reasoning steps are isolated within candidate groups, allowing direct preference optimization targeted specifically to the faulty reasoning step.
3. **Component C: Live Adaptive Curriculum**: We track rolling model accuracy. The framework actively samples training questions within the model's current "learning zone" (40-70% difficulty) using a pre-computed capability map, expanding this reasoning boundary dynamically as training stabilizes.

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

## System Architecture

```text
                  ┌──────────────────────────────────────────┐
                  │      Phase 0: Offline Capability Probe   │
                  │   - Establishes Question Difficulty Map  │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │        Phase 1: SFT Warmup (QLoRA)       │
                  │   - Formats traces: "Step 1...", "Step 2"│
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │        Phase 2: CRAFT RL Loop (GRPO)      │
                  │  ┌────────────────────────────────────┐  │
                  │  │ [Component C] Adaptive Curriculum  │  │
                  │  └─────────────────┬──────────────────┘  │
                  │                    ▼                     │
                  │  ┌────────────────────────────────────┐  │
                  │  │       Active SLM Model                │  │
                  │  └─────────────────┬──────────────────┘  │
                  │                    ▼                     │
                  │  ┌────────────────────────────────────┐  │
                  │  │ [Component A] Step Verifiers       │  │
                  │  │  - Symbolic Math Safe Eval         │  │
                  │  │  - DeBERTa NLI Logical Entailment │  │
                  │  └─────────────────┬──────────────────┘  │
                  │                    ▼                     │
                  │  ┌────────────────────────────────────┐  │
                  │  │ [Component B] Step DPO Preference  │  │
                  │  └────────────────────────────────────┘  │
                  └────────────────────┬─────────────────────┘
                                       │
                                       ▼
                  ┌──────────────────────────────────────────┐
                  │      Phase 3: Edge Deployment (GGUF)     │
                  │   - Llama.cpp Quantization (Q4_K_M)      │
                  │   - Interactive Desktop UI (FastAPI)     │
                  └──────────────────────────────────────────┘
```

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

---

## Project Artefacts

- **Technical Documentation** - Located in the [`docs/`](docs/) directory. (See Technical Documentation Guide above)
- **Source Code** - Located in the [`CRAFT-SLM-Reasoning-Framework-Phi3-mini/craft`](CRAFT-SLM-Reasoning-Framework-Phi3-mini/craft) directory for Phi-3-Mini, and the [`CRAFT-SLM-Reasoning-Framework-Qwen25/craft`](CRAFT-SLM-Reasoning-Framework-Qwen25/craft) directory for Qwen-2.5.
- **Datasets Used** - [GSM8K](https://huggingface.co/datasets/gsm8k), [AQuA-RAT](https://huggingface.co/datasets/aqua_rat), [StrategyQA](https://huggingface.co/datasets/wza/strategyqa), [MMLU](https://huggingface.co/datasets/cais/mmlu)

---

## Attribution

This project utilizes code templates from the [Samsung EnnovateX Phase 2 Solution Template](https://github.com/ennovatex-io/ax-hackathon-2026-full-solution-template).
All training, curriculum, and RL logic have been custom-developed by Team Aurvion.
