# User Guide

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

## Command Line Interface (CLI) Guide

CRAFT provides a single command line interface (`python -m craft`) to run diagnostic checks, difficulty probing, model fine-tuning, RL training, evaluation, GGUF quantization, and model packaging.

### 1. Doctor & Installation Verification
Before starting, verify your setup:
```bash
python -m craft doctor
```

### 2. Phase 0 — Capability Probe
Probe baseline question difficulty:
```bash
python -m craft probe --output Outputs/difficulty_map.json
```

### 3. Phase 1 — Supervised Fine-Tuning (SFT)
Run the warmup formatting SFT adapter phase on the 4-dataset mix:
```bash
python -m craft sft --hardware local_2x24gb
```
*To resume training from the latest checkpoint:*
```bash
python -m craft sft --hardware local_2x24gb --resume
```

### 4. Phase 2 — Reinforcement Learning (RL)
Align the model via GRPO and Step DPO using the curriculum mapper:
```bash
python -m craft rl --hardware local_2x24gb --checkpoint checkpoints/sft/final
```

### 5. Phase 3 — Quantization & Packaging
Quantize your aligned checkpoints into GGUF format and package the client server binaries:
* **Quantize model weights:**
  ```bash
  python -m craft quantize --checkpoint checkpoints/rl/checkpoint-500
  ```
* **Package model for delivery:**
  ```bash
  python -m craft package
  ```

---

## Deployed Inference Server & Web UI

After packaging the GGUF model via Phase 3, you can run the localized client server to serve reasoning chains.

### 1. Launching the Server
Navigate to the packaged delivery directory and run the launcher script:

* **On Linux / macOS:**
  ```bash
  cd craft_delivery
  chmod +x launch.sh
  ./launch.sh
  ```
* **On Windows:**
  ```cmd
  cd craft_delivery
  launch.bat
  ```

The script automatically verifies dependencies, starts the FastAPI server on `http://127.0.0.1:8000`, and runs in the background.

---

### 2. Using the Dashboard UI

Once the server is running, open the local file `craft_delivery/src/ui/index.html` in any web browser. 

The dashboard provides a premium dark-mode interface featuring:
* **Query Input Box:** Enter mathematical, logical, or multiple-choice questions.
* **Side-by-Side View:** Input queries and compare baseline outputs against CRAFT-aligned reasoning chains.
* **Step-by-Step Breakdown:** View structured thoughts decomposed into individual steps alongside formatting blocks.

![CRAFT Dashboard](screenshots/ui_dashboard.png)
<!-- SCREENSHOT NEEDED: Full dashboard view with a sample question entered
     and both baseline/CRAFT responses visible side by side. -->

---

### 3. Live Inference Example

The UI shows step-by-step progress bars and structured reasoning breakdowns in real time.

![Live inference example](screenshots/ui_inference_example.png)
<!-- SCREENSHOT NEEDED: Close-up of the step-by-step CRAFT response panel,
     showing structured Step 1/Step 2/Final Answer formatting. -->

---

## API Request & Response Interpretation

The FastAPI backend operates on structured JSON models.

### API Request
```json
{
  "question": "If a farmer has 15 cows and 12 sheep, how many animals does he have in total?",
  "temperature": 0.8,
  "max_tokens": 512
}
```

### API Response
```json
{
  "question": "If a farmer has 15 cows and 12 sheep, how many animals does he have in total?",
  "response": "Step 1: Identify the count of cows, which is 15.\nStep 2: Identify the count of sheep, which is 12.\nStep 3: Add the counts: 15 + 12 = 27.\nFinal Answer: 27",
  "steps": [
    "Identify the count of cows, which is 15.",
    "Identify the count of sheep, which is 12.",
    "Add the counts: 15 + 12 = 27."
  ],
  "final_answer": "27",
  "latency_ms": 312.45,
  "is_mock": false
}
```

* **`response`:** The raw generated text from the model.
* **`steps`:** A parsed list of reasoning strings, extracted by looking for `Step X:` patterns.
* **`final_answer`:** The short answer value extracted from the `Final Answer:` block.
* **`latency_ms`:** Serving time in milliseconds.
* **`is_mock`:** Boolean indicating if the server operated in dry-run/mock mode.
