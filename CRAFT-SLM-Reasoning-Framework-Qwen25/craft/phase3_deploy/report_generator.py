"""
report_generator.py — Generates benchmark report from evaluation JSONs.
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
        if not os.path.exists(path):
            logger.warning(f"Results file not found: {path}")
            return {}
        with open(path) as f:
            return json.load(f)

    def _get_acc(self, results: dict, dataset: str) -> float:
        if "summary" in results.get(dataset, {}):
            return results[dataset]["summary"].get("accuracy", 0.0)
        return results.get(dataset, {}).get("accuracy", 0.0)

    def generate_full_report(
        self,
        baseline_json: str,
        champion_json: str,
        champion_label: str,
        extra_checkpoints: dict,
        report_md_path: str = "craft_output/benchmark_report.md",
    ):
        logger.info("Generating benchmark report...")

        baseline = self._load_json(baseline_json)
        champion = self._load_json(champion_json)
        datasets = ["gsm8k", "strategyqa", "mmlu"]

        rows = []
        for ds in datasets:
            b_acc = self._get_acc(baseline, ds)
            c_acc = self._get_acc(champion, ds)
            rows.append({
                "dataset": ds,
                "baseline": b_acc,
                "champion": c_acc,
                "delta": c_acc - b_acc,
            })

        avg_delta = sum(r["delta"] for r in rows) / len(rows)

        content = f"""# CRAFT — Benchmark Evaluation Report

## Samsung EnnovateX Hackathon Phase 2

Champion model: **{champion_label}**

## Core Benchmark Results

| Benchmark | Baseline | CRAFT | Δ |
|-----------|----------|-------|---|
| GSM8K | {rows[0]['baseline']:.1%} | {rows[0]['champion']:.1%} | {rows[0]['delta']:+.1%} |
| StrategyQA | {rows[1]['baseline']:.1%} | {rows[1]['champion']:.1%} | {rows[1]['delta']:+.1%} |
| MMLU | {rows[2]['baseline']:.1%} | {rows[2]['champion']:.1%} | {rows[2]['delta']:+.1%} |

**Average improvement: {avg_delta:+.1%}**

## Deployment Specifications

| Property | Value |
|----------|-------|
| Base Model | microsoft/Phi-3-mini-4k-instruct (3.8B) |
| Deployment Format | GGUF Q4_K_M |
| Model Size | ~2.2 GB |
| Inference | CPU or GPU ≥ 4GB VRAM |
"""

        with open(report_md_path, "w") as f:
            f.write(content)
        logger.info(f"Report written to {report_md_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--champion-json", required=True)
    parser.add_argument("--champion-label", default="CRAFT")
    parser.add_argument("--report-md", default="craft_output/benchmark_report.md")
    args = parser.parse_args()

    gen = ReportGenerator()
    gen.generate_full_report(
        baseline_json=args.baseline_json,
        champion_json=args.champion_json,
        champion_label=args.champion_label,
        extra_checkpoints={},
        report_md_path=args.report_md,
    )


if __name__ == "__main__":
    main()