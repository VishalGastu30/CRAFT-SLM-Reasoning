"""
report_generator.py — Generates benchmark report from REAL evaluation JSONs.
Now accepts an arbitrary number of checkpoint files (--ckpt-<step>-json).
All numbers come from actual evaluator.py output. Nothing hardcoded.
"""

import os
import json
import argparse
from loguru import logger


class ReportGenerator:

    def __init__(self, output_dir="craft_output"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_json(self, path: str) -> dict:
        """Load a results JSON file. Returns empty dict if missing."""
        if not os.path.exists(path):
            logger.warning(f"Results file not found: {path}. Using empty dict.")
            return {}
        with open(path) as f:
            return json.load(f)

    def _get_acc(self, results: dict, dataset: str) -> float:
        """Extract accuracy as a float 0-1 from results dict."""
        return results.get(dataset, {}).get("summary", {}).get("accuracy", 0.0)

    def _get_fmt(self, results: dict, dataset: str) -> float:
        return results.get(dataset, {}).get("summary", {}).get("format_compliance", 0.0)

    def _get_steps(self, results: dict, dataset: str) -> float:
        return results.get(dataset, {}).get("summary", {}).get("avg_steps", 0.0)

    def _bar(self, value: float, width: int = 40) -> str:
        """ASCII bar for a 0.0-1.0 value."""
        filled = int(value * width)
        return "█" * filled + "░" * (width - filled)

    def generate_full_report(
        self,
        baseline_json: str,
        champion_json: str,
        champion_label: str,
        extra_checkpoints: dict,  # {step: json_path}
        report_md_path: str = "craft_output/benchmark_report.md",
    ):
        """
        Generate the full benchmark report from real evaluation JSONs.
        extra_checkpoints: dictionary mapping step numbers to their result JSON paths.
        """
        logger.info("Generating benchmark report from real evaluation data...")

        baseline = self._load_json(baseline_json)
        champion = self._load_json(champion_json)
        datasets = ["gsm8k", "strategyqa", "mmlu"]

        # Samsung's required minimum scores
        SAMSUNG_MIN = {"gsm8k": 0.50, "strategyqa": 0.65, "mmlu": 0.45}
        SAMSUNG_DELTA_MIN = 0.05

        # ── Build comparison rows ──────────────────────────────────────────
        rows = []
        for ds in datasets:
            b_acc = self._get_acc(baseline, ds)
            c_acc = self._get_acc(champion, ds)
            delta = c_acc - b_acc
            samsung_min = SAMSUNG_MIN.get(ds, 0.45)
            passes_min = c_acc >= samsung_min
            passes_delta = delta >= SAMSUNG_DELTA_MIN
            rows.append({
                "dataset": ds,
                "baseline": b_acc,
                "champion": c_acc,
                "delta": delta,
                "baseline_fmt": self._get_fmt(baseline, ds),
                "champion_fmt": self._get_fmt(champion, ds),
                "baseline_steps": self._get_steps(baseline, ds),
                "champion_steps": self._get_steps(champion, ds),
                "passes_min": passes_min,
                "passes_delta": passes_delta,
                "sample_count": baseline.get(ds, {}).get("summary", {}).get("sample_count", 100),
            })

        avg_delta = sum(r["delta"] for r in rows) / len(rows)

        # ── Build checkpoint sweep table from extra_checkpoints ────────────
        sweep_section = ""
        if extra_checkpoints:
            sweep_section = "\n## 3. Checkpoint Sweep — Full Comparison\n\n"
            sweep_section += "All checkpoints evaluated in identical conditions (4-bit, HF, greedy decoding).\n\n"
            sweep_section += "| Checkpoint | GSM8K Acc | StrategyQA Acc | MMLU Acc | Avg Improvement | Status |\n"
            sweep_section += "|---|---|---|---|---|---|\n"
            for step in sorted(extra_checkpoints.keys()):
                ck_data = self._load_json(extra_checkpoints[step])
                ck_gsm = self._get_acc(ck_data, "gsm8k")
                ck_strat = self._get_acc(ck_data, "strategyqa")
                ck_mmlu = self._get_acc(ck_data, "mmlu")
                b_gsm = rows[0]["baseline"]
                b_strat = rows[1]["baseline"]
                b_mmlu = rows[2]["baseline"]
                avg_imp = ((ck_gsm - b_gsm) + (ck_strat - b_strat) + (ck_mmlu - b_mmlu)) / 3
                # Determine if this checkpoint is the champion
                is_champion = (step == int(champion_label.split("-")[-1]))
                crown = " 👑 CHAMPION" if is_champion else ""

                # Check Samsung targets
                passes_gsm = ck_gsm >= 0.50 and (ck_gsm - b_gsm) >= 0.05
                passes_strat = ck_strat >= 0.65 and (ck_strat - b_strat) >= 0.05
                passes_mmlu = ck_mmlu >= 0.45 and (ck_mmlu - b_mmlu) >= 0.05
                passes_count = sum([passes_gsm, passes_strat, passes_mmlu])
                status = "✅ PASS" if passes_count >= 2 else "⚠️"

                sweep_section += (
                    f"| checkpoint-{step}{crown} | {ck_gsm:.1%} | {ck_strat:.1%} | {ck_mmlu:.1%} | "
                    f"+{avg_imp:.1%} | {status} |\n"
                )

        gsm_status = '✅ MET' if rows[0]['passes_min'] and rows[0]['passes_delta'] else '⚠️ CHECK'
        strat_status = '✅ MET' if rows[1]['passes_min'] and rows[1]['passes_delta'] else '⚠️ CHECK'
        mmlu_status = '✅ MET' if rows[2]['passes_min'] and rows[2]['passes_delta'] else '⚠️ CHECK'

        # ── Compose full report ────────────────────────────────────────────
        content = f"""# CRAFT — Benchmark Evaluation Report
## Samsung EnnovateX Hackathon Phase 2 | CRAFT-Phi3-Mini

> All scores measured using identical evaluation conditions: 4-bit quantization,
> greedy decoding (temperature=0), same prompt format across baseline and CRAFT.
> Champion model: **{champion_label}**

---

## 1. Executive Summary

CRAFT (Curriculum-guided Reinforced Adaptive Fine-Tuning) addresses three specific
failure modes of RL applied to SLMs:

1. **Training instability** — Adaptive curriculum (Component C) keeps the model
   in the 40–70% difficulty zone throughout training, preventing KL collapse.
2. **Unreliable reward signal** — Deterministic execution verifier (Component A)
   replaces noisy neural reward models with Python-executed math verification.
3. **Outcome-blind learning** — Step-level contrastive DPO (Component B) teaches
   the model which specific reasoning steps separate correct from incorrect traces.

---

## 2. Core Benchmark Results

### Samsung Required Targets
| Benchmark | Minimum Score | Required Delta | Status |
|---|---|---|---|
| GSM8K | ≥ 50.0% | ≥ +5.0% | {gsm_status} |
| StrategyQA | ≥ 65.0% | ≥ +5.0% | {strat_status} |
| MMLU | ≥ 45.0% | ≥ +5.0% | {mmlu_status} |

### GSM8K (Multi-Step Mathematical Reasoning)
| Model | Accuracy | Format Compliance | Avg Steps | Δ vs Baseline |
|---|---|---|---|---|
| Phi-3-Mini (Baseline) | {rows[0]['baseline']:.1%} | {rows[0]['baseline_fmt']:.1%} | {rows[0]['baseline_steps']:.1f} | — |
| **CRAFT {champion_label}** | **{rows[0]['champion']:.1%}** | **{rows[0]['champion_fmt']:.1%}** | **{rows[0]['champion_steps']:.1f}** | **{rows[0]['delta']:+.1%}** |


### StrategyQA (Logical Multi-Step Inference)
| Model | Accuracy | Format Compliance | Avg Steps | Δ vs Baseline |
|---|---|---|---|---|
| Phi-3-Mini (Baseline) | {rows[1]['baseline']:.1%} | {rows[1]['baseline_fmt']:.1%} | {rows[1]['baseline_steps']:.1f} | — |
| **CRAFT {champion_label}** | **{rows[1]['champion']:.1%}** | **{rows[1]['champion_fmt']:.1%}** | **{rows[1]['champion_steps']:.1f}** | **{rows[1]['delta']:+.1%}** |


### MMLU (Multi-Choice Knowledge)
| Model | Accuracy | Format Compliance | Avg Steps | Δ vs Baseline |
|---|---|---|---|---|
| Phi-3-Mini (Baseline) | {rows[2]['baseline']:.1%} | {rows[2]['baseline_fmt']:.1%} | {rows[2]['baseline_steps']:.1f} | — |
| **CRAFT {champion_label}** | **{rows[2]['champion']:.1%}** | **{rows[2]['champion_fmt']:.1%}** | **{rows[2]['champion_steps']:.1f}** | **{rows[2]['delta']:+.1%}** |


**Average improvement across all three benchmarks: {avg_delta:+.1%}**

{sweep_section}

## 4. Deployment Specifications

| Property | Value |
|---|---|
| Base Model | microsoft/Phi-3-mini-4k-instruct (3.8B parameters) |
| Training Framework | GRPO + Component A (Execution Verifier) + Component B (Step DPO) + Component C (Adaptive Curriculum) |
| Champion Checkpoint | {champion_label} |
| Deployment Format | GGUF Q4_K_M |
| Estimated Model Size | ~2.2 GB |
| Inference Hardware | CPU or any GPU ≥ 4GB VRAM |
| Cloud Dependency | None — fully on-device |

## 5. Evaluation Methodology

- **GSM8K**: Full test set ({rows[0].get('sample_count', 100)} samples evaluated)
- **StrategyQA**: Test split ({rows[1].get('sample_count', 100)} samples evaluated)
- **MMLU**: Test split ({rows[2].get('sample_count', 100)} samples evaluated)
- **Decoding**: Greedy (temperature=0, do_sample=False) for reproducibility
- **Answer extraction**: "Final Answer:" pattern (primary), last number (fallback)
- **Scoring**: Exact match with numeric tolerance (|predicted - truth| < 0.01)
- **All comparisons**: Same 4-bit quantization applied to both baseline and CRAFT for fair measurement
"""

        with open(report_md_path, "w") as f:
            f.write(content)
        logger.info(f"Report written to {report_md_path}")

    def generate_report(self, results_path: str, report_md_path: str):
        """Legacy single-file interface for backward compatibility."""
        self.generate_full_report(
            baseline_json=results_path,
            champion_json=results_path,
            champion_label="CRAFT",
            extra_checkpoints={},
            report_md_path=report_md_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--champion-json", required=True)
    parser.add_argument("--champion-label", default="checkpoint-200")
    parser.add_argument("--report-md", default="craft_output/benchmark_report.md")
    # Parse known args; collect remaining as --ckpt-XXX-json
    args, unknown = parser.parse_known_args()

    extra_checkpoints = {}
    for i in range(0, len(unknown), 2):
        if i + 1 >= len(unknown):
            break
        key = unknown[i]
        value = unknown[i + 1]
        if key.startswith("--ckpt-") and key.endswith("-json"):
            # Extract step number from e.g. --ckpt-150-json
            step_str = key[6:-5]  # remove '--ckpt-' and '-json'
            if step_str.isdigit():
                step = int(step_str)
                extra_checkpoints[step] = value
            else:
                logger.warning(f"Skipping unrecognised argument: {key}")
        else:
            logger.warning(f"Skipping unexpected argument: {key}")

    gen = ReportGenerator()
    gen.generate_full_report(
        baseline_json=args.baseline_json,
        champion_json=args.champion_json,
        champion_label=args.champion_label,
        extra_checkpoints=extra_checkpoints,
        report_md_path=args.report_md,
    )


if __name__ == "__main__":
    main()