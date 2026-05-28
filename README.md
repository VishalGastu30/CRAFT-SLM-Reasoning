# CRAFT: Curriculum-guided Reinforced Adaptive Fine-Tuning

- **Problem Statement Number** - 06
- **Problem Statement Title** - Enhancing Reasoning in Small Language Models (SLMs) using Reinforcement Learning
- **Team name** - Aurvion
- **Team members (Names)** - G.Vishal, E.Kevin Godfrey
- **Institute/College Name** - SRM Institute of Science and Technology, Kattankulathur, Chennai - 603203
- **Final Presentation Google Drive Link** - *[Pending upload during Week 4]*
- **Full Submission Demo Video Link** - *[Pending upload during Week 4]*
- **Setup & Result Reproducibility Video Link** - *[Pending upload during Week 4]*

---

## Project Overview

CRAFT (Curriculum-guided Reinforced Adaptive Fine-Tuning) is an advanced reinforcement learning training framework designed specifically to enhance reasoning capabilities in Small Language Models (SLMs) like Microsoft's Phi-3-Mini (3.8B). SLMs are highly attractive for edge deployment because of their low compute and memory requirements. However, applying RL to them often fails due to three major limitations:
1. **Sparse/Misaligned Step Rewards**: Outcome-only rewards result in correct answers via flawed reasoning steps.
2. **Preference Signal Mismatch**: Coarse step comparisons fail to differentiate between alternative reasoning paths.
3. **KL-Divergence Collapse / Distribution Shift**: Standard RL training destabilizes and collapses quickly under distribution shift.

CRAFT solves these issues through a unified four-phase pipeline: Capability Probing, SFT Warmup, the adaptive CRAFT RL Loop (integrating math step verification, step-level DPO, and a live difficulty curriculum), and GGUF Edge Deployment.

---

## Project Artefacts

- **Technical Documentation** - Located in the [docs/](file:///home/vishal/Projects/CRAFT/docs/) directory. Contains:
  - [ax.md](file:///home/vishal/Projects/CRAFT/docs/ax.md) (Agentic AI design document, Samsung required)
  - [architecture.md](file:///home/vishal/Projects/CRAFT/docs/architecture.md) (Full system architecture & diagrams)
  - [installation.md](file:///home/vishal/Projects/CRAFT/docs/installation.md) (Kaggle & local setup guide)
  - [user_guide.md](file:///home/vishal/Projects/CRAFT/docs/user_guide.md) (CLI & UI usage instructions)
  - [training_guide.md](file:///home/vishal/Projects/CRAFT/docs/training_guide.md) (Phase-by-phase training steps)
  - [benchmark_results.md](file:///home/vishal/Projects/CRAFT/docs/benchmark_results.md) (Evaluation & ablation tables)
  - [technical_stack.md](file:///home/vishal/Projects/CRAFT/docs/technical_stack.md) (OSS libraries & licenses)
- **Source Code** - Located in the [src/](file:///home/vishal/Projects/CRAFT/src/) directory.
- **Models Used** - [Microsoft Phi-3-Mini-4k-Instruct](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) (Base & SFT)
- **Models Published** - *[Pending upload to HuggingFace under Aurvion/CRAFT-Phi3-Mini]*
- **Datasets Used** - [GSM8K](https://huggingface.co/datasets/gsm8k), [AQuA-RAT](https://huggingface.co/datasets/aqua_rat), [StrategyQA](https://huggingface.co/datasets/wza/strategyqa), [MMLU](https://huggingface.co/datasets/cais/mmlu)

---

## Innovation & Novelty

The novelty of CRAFT lies in three core components:
1. **Component A: Step-Level Symbolic & NLI Execution Verifier**: Instead of checking only the final answer, CRAFT extracts math expressions and evaluates them using a safe Python interpreter. For logical questions, it runs a local NLI model (`DeBERTa-v3-small`) to score the entailment between successive steps, ensuring the reasoning chain is sound.
2. **Component B: Contrastive Step-Level Reinforcement**: We extract step boundaries from reasoning traces and pair correct steps with common errors to compute a step-level DPO preference loss, forcing the model to explicitly learn step-by-step corrections.
3. **Component C: Live Adaptive Curriculum**: We track rolling model accuracy. The framework actively samples training questions within the model's current "learning zone" (40-70% difficulty) using a pre-computed capability map, expanding this zone dynamically as training stabilizes.

---

## System Architecture

```
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
                  │  │       Active Phi-3-Mini Model      │  │
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

## Setup & Reproducibility

### Local CLI & UI Inference
To run local inference using our pre-trained model, execute:
```bash
# Set up environment
bash scripts/setup.sh
# Download the GGUF model
bash scripts/download_model.sh
# Run CLI inference
python src/inference/inference.py --question "If John has 3 apples and eats 1, how many does he have left?"
# Start the Web UI
uvicorn src.inference.inference_server:app --port 8000
```
Open `src/ui/index.html` in your browser to interact with the dashboard.

---

## Attribution

This project utilizes code templates from the [Samsung EnnovateX Phase 2 Solution Template](https://github.com/ennovatex-io/ax-hackathon-2026-full-solution-template).
All training, curriculum, and RL logic have been custom-developed by Team Aurvion.
