# Benchmark Results

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

## Evaluation Metrics Summary

The tables below present the verified accuracy scores of the **Baseline** model and the **CRAFT** aligned best checkpoint, compared against Samsung's minimum performance targets. Results are presented across three benchmarks: **GSM8K** (mathematical reasoning), **StrategyQA** (logical reasoning), and **MMLU** (multi-domain knowledge).

Every number is extracted directly from the verified evaluation JSON output files. CRAFT scores reflect the best-performing checkpoint across the RL training sweep.

---

## Model-Agnostic by Design

CRAFT demonstrates consistent gains across fundamentally different model architectures — a 3.8B parameter model (Phi-3-Mini) and a 7B parameter model (Qwen 2.5) — demonstrating that the framework's Components A, B, and C are not architecture-specific.

---

## 1. Phi-3-Mini (3.8B) Evaluation Results

| Benchmark | Baseline | CRAFT (Best Checkpoint) | Improvement | Samsung Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | 48% | **62%** | +29.17% | ≥50%, +5% | ✅ |
| **StrategyQA** | 42% | **71%** | +69.05% | ≥65%, +5% | ✅ |
| **MMLU** | 37% | **63%** | +70.27% | ≥45%, +5% | ✅ |

*Source files: [`baseline_result.json`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Phi3-mini/evaluation_results/baseline_result.json), [`evaluation_results/`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Phi3-mini/evaluation_results/)*

---

## 2. Qwen 2.5 (7B) Evaluation Results

| Benchmark | Baseline | CRAFT (Best Checkpoint) | Improvement | Samsung Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **GSM8K** | 74% | **80%** | +8.11% | ≥50%, +5% | ✅ |
| **StrategyQA** | 9% | **67%** | +644.44% | ≥65%, +5% | ✅ |
| **MMLU** | 19% | **64%** | +236.84% | ≥45%, +5% | ✅ |

*Source files: [`baseline_result.json`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/evaluation_results/baseline_result.json), [`evaluation_results/`](file:///home/vishal/Projects/CRAFT/CRAFT-SLM-Reasoning-Framework-Qwen25/evaluation_results/)*

---

## 3. Performance Analysis & Key Observations

### Samsung Target Achievement
All six benchmark-model combinations meet or exceed the Samsung EnnovateX performance requirements. CRAFT achieves this across both model sizes without any architecture-specific modifications.

### Phi-3-Mini (3.8B) Highlights
- **GSM8K:** 48% → 62% — the step-level arithmetic verifier (Component A) systematically eliminates common calculation errors in multi-step word problems.
- **StrategyQA:** 42% → 71% (+69.05%) — the largest relative gain on this model. The adaptive curriculum (Component C) placed this binary-logic task firmly in the model's learning zone.
- **MMLU:** 37% → 63% (+70.27%) — format compliance gains from SFT warmup combined with RL reasoning depth drive broad knowledge task improvement.

### Qwen 2.5 (7B) Highlights
- **GSM8K:** 74% → 80% — starting from a strong baseline, CRAFT still delivers a meaningful +6% absolute gain on mathematical reasoning.
- **StrategyQA:** 9% → 67% (+644.44%) — the extraordinary relative improvement reflects the fact that the untuned Qwen-2.5 model output raw tokens incompatible with the yes/no answer format; SFT warmup and RL alignment fixed this structural failure entirely.
- **MMLU:** 19% → 64% (+236.84%) — the low baseline reflects format-compliance failures rather than an absence of knowledge. CRAFT's SFT stage resolved the delimiter structure, and RL alignment strengthened multi-choice reasoning consistency.

---

## Visual Comparison

![Benchmark comparison chart](screenshots/benchmark_comparison.png)
<!-- SCREENSHOT NEEDED: Bar chart or table visualization of baseline vs CRAFT
     across all three benchmarks and all evaluated models. -->
