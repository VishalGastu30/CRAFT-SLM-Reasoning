# Documentation Generation Plan — docs/ Folder
## Instructions for the Coding Agent

You are tasked with creating complete technical documentation for the CRAFT project
(Curriculum-guided Reinforced Adaptive Fine-Tuning for SLMs), built for Samsung
EnnovateX 2026 Phase 2 submission. Follow this plan exactly, in order.

---

## STEP 0 — Before Writing Anything: Explore the Project

Do not write any documentation file until you have done the following:

1. **List the full repository structure.** Run a recursive directory listing
   (excluding `.git`, `node_modules`, `__pycache__`, `checkpoints/`, `craft_output/`
   large binaries) and read it carefully.

2. **Read every source file in `src/`.** This includes:
   - `src/phase0_probe/` — all files
   - `src/phase1_sft/` — all files
   - `src/phase2_rl/` — including `component_a/`, `component_b/`, `component_c/`
   - `src/phase3_deploy/`
   - `src/inference/`
   - `src/ui/`
   - `src/craft_pipeline.py`
   - `src/utils/`

3. **Read every config file** in `configs/` or `config/` to understand actual
   hyperparameters used (don't guess — use the real values present in the YAML files).

4. **Read `requirements.txt`** and any `environment.yml` to get exact library
   versions actually pinned in this project.

5. **Read existing files** if present: `README.md`, any existing `docs/*.md`,
   `LICENSE`, `.gitignore` — to avoid contradicting or duplicating existing content.

6. **Check for a `tests/` folder** and read test files if present — these reveal
   intended behavior and edge cases worth documenting.

7. **Check for benchmark/results files** — `results/*.json`, `craft_output/benchmark_report.md`,
   any ablation study output — and extract the REAL numbers from them. Never
   invent or estimate a benchmark number. If a number cannot be found in an
   actual results file, write `[NEEDS REAL VALUE — see results/*.json]` instead
   of guessing.

8. **Check git log / commit history** if accessible, to understand chronological
   development order — this helps write an accurate architecture narrative rather
   than a generic one.

**Do not proceed to Step 1 until you have actually read these files. Documentation
written without reading the real code will contain incorrect function names,
incorrect file paths, and incorrect parameter values — all of which actively hurt
the project during evaluation.**

---

## STEP 1 — Create the docs/ Folder Structure

Create exactly this structure:

```
docs/
├── ax.md                          ← Skip if this already exists (see note below)
├── technical_stack.md
├── architecture.md
├── implementation_details.md
├── installation.md
├── user_guide.md
├── features.md
├── benchmark_results.md
├── api_reference.md
└── screenshots/
    └── .gitkeep
```

**Note on ax.md:** If `docs/ax.md` already exists in the repository, do NOT
overwrite it — it documents the agentic development workflow and may already be
finalized. Only create it if it does not exist, and if you must create it, base
its content strictly on actual tool usage found in commit history, comments, or
explicit instructions provided separately — never fabricate agentic workflow claims.

---

## STEP 2 — File-by-File Content Instructions

### 2.1 `docs/technical_stack.md`

Document every technology actually used, derived from `requirements.txt` and
actual import statements in the code — not a generic ML stack list.

Required sections:
- **Languages & Runtime** (Python version from environment files)
- **Core ML Libraries** — for each, include: name, version (from requirements.txt),
  one-sentence purpose in this project, and link to the official OSS repository/page
- **Training Framework** (e.g., HuggingFace TRL for GRPO — confirm by checking
  actual import in `craft_rl_loop.py`)
- **Quantization & Deployment Tools** (llama.cpp, bitsandbytes — confirm via
  actual usage in `quantizer.py`)
- **Backend/UI Stack** (check `src/inference/` and `src/ui/` for actual
  frameworks used — e.g., FastAPI, vanilla HTML/CSS/JS)
- **Evaluation Tools** (check `evaluator.py` for actual library — lm-evaluation-harness
  or custom)

Format as a table:
```markdown
| Library | Version | Purpose | Link |
|---|---|---|---|
| transformers | X.X.X | Model loading and generation | https://github.com/huggingface/transformers |
```

Every single row must be backed by an actual line in `requirements.txt` or an
actual `import` statement you found while reading the code. Do not list a
library that isn't actually imported anywhere in `src/`.

---

### 2.2 `docs/architecture.md`

This is the most important file. Structure:

1. **System Overview** — one paragraph, plain language, what CRAFT does end to end
2. **Pipeline Diagram** — insert this placeholder:
   ```
   ![CRAFT Architecture Diagram](screenshots/architecture_diagram.png)
   <!-- SCREENSHOT NEEDED: Full Phase 0→1→2→3 pipeline diagram.
        Use the existing architecture diagram generated earlier in the project,
        or regenerate from src/craft_pipeline.py structure. -->
   ```
3. **Phase-by-phase breakdown** — for EACH phase (0, 1, 2, 3), write:
   - What it does (derived from actually reading the phase's source file)
   - Input → Output (exact file paths used in the actual code, e.g.
     `difficulty_map.json`, `checkpoints/sft/final`)
   - Key functions/classes involved, with their REAL names from the source code
     (do not invent function names — copy them exactly as they appear)
4. **Component A/B/C Deep Dive** — one subsection each, explaining:
   - The exact formula/logic as implemented (read `reward_scorer.py`,
     `contrastive_builder.py`, `curriculum_engine.py` and describe what they
     ACTUALLY do, including any quirks or fixes that were applied)
   - Reference the actual class names and method names found in the code
5. **Data Flow Diagram** — placeholder:
   ```
   ![Data Flow](screenshots/data_flow.png)
   <!-- SCREENSHOT NEEDED: Diagram showing difficulty_map.json → SFT checkpoint
        → RL checkpoint → GGUF → benchmark results, with arrows. -->
   ```

---

### 2.3 `docs/implementation_details.md`

For each major module, document:

- **File path** (exact, from the repo)
- **Purpose** (one sentence)
- **Key implementation choices and why** — e.g., "QLoRA with rank=64 was chosen
  over full fine-tuning because..." — pull the actual rank/alpha/config values
  from the real config files, never placeholder numbers
- **Known constraints or edge cases handled** — check for any comments in code
  that explain "why" a particular approach was taken (e.g., handling for GSM8K's
  `#### N` answer format, or a fix for a specific bug) — document these as
  deliberate design decisions

Include a subsection: **Hyperparameters Used**, as a table, with values read
directly from the actual YAML config files in `configs/`:

```markdown
| Parameter | Value | File |
|---|---|---|
| LoRA rank | [READ FROM ACTUAL CONFIG] | configs/phi3_mini.yaml |
| Learning rate | [READ FROM ACTUAL CONFIG] | configs/phi3_mini.yaml |
```

Never write a hyperparameter value without confirming it against the actual
config file content.

---

### 2.4 `docs/installation.md`

Write exact, copy-pasteable, tested commands. Structure:

```markdown
## Prerequisites
- Python [exact version from environment files]
- [GPU requirements — derived from actual VRAM needs you find documented
  in the codebase or config comments]

## Step 1: Clone the repository
```bash
git clone [actual repo URL if known, otherwise placeholder: <YOUR_REPO_URL>]
cd CRAFT-SLM-Reasoning
```

## Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

## Step 3: Verify installation
[Include an actual smoke-test command if one exists in scripts/ or tests/]
```

Add a screenshot placeholder after the verification step:
```
![Successful installation output](screenshots/installation_success.png)
<!-- SCREENSHOT NEEDED: Terminal output showing successful dependency install
     and any verification script passing. -->
```

If a `scripts/setup.sh` file exists, document its exact contents rather than
inventing a generic setup process.

---

### 2.5 `docs/user_guide.md`

This explains how someone uses the FINISHED product (not how to train it —
that's `implementation_details.md`). Structure:

1. **Running Inference** — exact CLI command from `src/inference/inference.py`,
   read its actual argparse arguments and document them precisely
2. **Using the Dashboard UI** — step by step:
   - How to start the inference server (exact command from `inference_server.py`)
   - How to open the UI
   - Screenshot placeholder:
     ```
     ![CRAFT Dashboard](screenshots/ui_dashboard.png)
     <!-- SCREENSHOT NEEDED: Full dashboard view with a sample question entered
          and both baseline/CRAFT responses visible side by side. -->
     ```
   - Screenshot placeholder for a response in progress:
     ```
     ![Live inference example](screenshots/ui_inference_example.png)
     <!-- SCREENSHOT NEEDED: Close-up of the step-by-step CRAFT response panel,
          showing structured Step 1/Step 2/Final Answer formatting. -->
     ```
3. **Switching Between Models** — if the UI/config supports multiple models
   (Phi-3-Mini, Qwen2.5-7B, Gemma-4-E4B), document exactly how a user switches
   between them, based on actual UI/config code
4. **Interpreting the Output** — explain the JSON response fields actually
   returned by the inference server (read the actual response schema in
   `inference_server.py`)

---

### 2.6 `docs/features.md`

List the salient features of the project, each backed by an actual implemented
mechanism (not aspirational claims):

```markdown
## Salient Features

### 1. Deterministic, Noise-Free Reward Signal
[Explain Component A as actually implemented]

### 2. Self-Supervised Step-Level Learning
[Explain Component B as actually implemented]

### 3. Live Adaptive Curriculum
[Explain Component C as actually implemented]

### 4. Model-Agnostic Training Framework
[Only include this claim if the actual codebase demonstrates it —
 e.g., multiple config files for different models exist and were actually used]

### 5. Fully On-Device Deployment
[Explain the GGUF quantization pipeline as actually implemented]
```

For each feature, if there is a corresponding benchmark number or measured
result, cite it. If not, do not invent one.

---

### 2.7 `docs/benchmark_results.md`

**This file must be built ENTIRELY from real result files. Do not write a single
number from memory or estimation.**

1. Search the repository for actual results files:
   - `results/*.json`
   - `craft_output/*.json`
   - `craft_output/benchmark_report.md`
2. Parse and extract real accuracy numbers for each model/benchmark combination found
3. Build a comparison table using ONLY confirmed numbers:

```markdown
| Model | Benchmark | Baseline | CRAFT | Δ | Source File |
|---|---|---|---|---|---|
```

4. If an ablation study results file exists, include that table too
5. If certain results are missing (e.g., Gemma-4-E4B MMLU was never run), write
   that explicitly: `Not yet evaluated — see [task/issue reference if any]`
   rather than omitting the row silently or guessing a number

Add screenshot placeholder:
```
![Benchmark comparison chart](screenshots/benchmark_comparison.png)
<!-- SCREENSHOT NEEDED: Bar chart or table visualization of baseline vs CRAFT
     across all three benchmarks and all evaluated models. -->
```

---

### 2.8 `docs/api_reference.md`

Document the actual inference server API by reading `inference_server.py` directly:

For each FastAPI route found in the actual code, document:
- Method and path (exact, e.g., `POST /infer`)
- Request body schema (exact fields from the actual Pydantic model)
- Response schema (exact fields from the actual return statement)
- One example request/response pair

```markdown
### POST /infer

**Request:**
```json
{
  "question": "string",
  "model_type": "baseline | craft"
}
```

**Response:**
[Exact fields as found in the actual route handler return statement]
```

Do not document an endpoint that doesn't exist in the code, and do not omit one
that does.

---

## STEP 3 — Screenshot Placeholder Convention

Every place a screenshot would help must use this exact format so a human can
find and fill them in later:

```markdown
![Alt text describing the image](screenshots/descriptive_filename.png)
<!-- SCREENSHOT NEEDED: One clear sentence describing exactly what should be
     captured, including what state the UI/terminal should be in. -->
```

Maintain a master list at the bottom of `docs/architecture.md`:

```markdown
## Screenshot Checklist (for human to capture before final submission)

- [ ] screenshots/architecture_diagram.png — full pipeline diagram
- [ ] screenshots/data_flow.png — data flow between phases
- [ ] screenshots/installation_success.png — terminal showing successful setup
- [ ] screenshots/ui_dashboard.png — full dashboard view
- [ ] screenshots/ui_inference_example.png — close-up of step-by-step response
- [ ] screenshots/benchmark_comparison.png — results visualization
```

Add every placeholder you create to this checklist as you go, so nothing is missed.

---

## STEP 4 — Cross-Linking

After all files are created, go back and add a short navigation section at the
top of each file linking to the others, and ensure `README.md` at the repo root
links to `docs/` with a one-line description of each file:

```markdown
## Documentation
- [Technical Stack](docs/technical_stack.md)
- [Architecture](docs/architecture.md)
- [Implementation Details](docs/implementation_details.md)
- [Installation Guide](docs/installation.md)
- [User Guide](docs/user_guide.md)
- [Features](docs/features.md)
- [Benchmark Results](docs/benchmark_results.md)
- [API Reference](docs/api_reference.md)
- [Agentic AI Development (ax.md)](docs/ax.md)
```

---

## STEP 5 — Final Self-Check Before Declaring Done

Before finishing, verify each of the following. If any check fails, fix it before
stopping:

- [ ] Every file path mentioned in any doc actually exists in the repository
- [ ] Every function/class name mentioned matches the actual code exactly (case-sensitive)
- [ ] Every hyperparameter value is copied from an actual config file, not invented
- [ ] Every benchmark number is copied from an actual results file, not invented
- [ ] Every library version matches `requirements.txt` exactly
- [ ] No file claims a feature that doesn't actually exist in the code
- [ ] Every screenshot placeholder has a clear, specific description (not generic "add image here")
- [ ] The screenshot checklist in `architecture.md` lists every placeholder created across all files
- [ ] All internal links between docs files use correct relative paths and actually resolve

**If at any point you cannot find a real value for something (a number, a file
path, a config value), write an explicit placeholder marker like
`[VERIFY: could not locate actual value for X in codebase]` rather than guessing.
A documentation file with an honest gap is far better than one with a confident
but fabricated detail — fabricated technical details are extremely easy for a
judge or reviewer to catch and will damage the credibility of the entire submission.**
