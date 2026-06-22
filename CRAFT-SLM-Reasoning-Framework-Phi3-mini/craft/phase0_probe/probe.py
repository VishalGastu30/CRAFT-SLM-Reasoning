import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger
from craft.utils.logger import setup_logger
from craft.utils.hardware_detector import detect_hardware
from craft.config import load_config
from .difficulty_mapper import DifficultyMapper


class CapabilityProbe:
    """
    Main runner for Phase 0 Capability Probe.
    Probes GSM8K, AQuA-RAT, StrategyQA, and MMLU to build difficulty_map.json.
    """

    def __init__(self, config):
        self.config = config
        self.probe_cfg = config.get("probe", {})
        self.model_cfg = config.get("model", {})
        self.mapper = DifficultyMapper()

    def load_probe_datasets(self):
        """
        Loads all four datasets and samples a fixed number of questions.
        Returns a list of dicts with keys: id, question, answer, dataset.
        """
        logger.info("Loading probing datasets from HuggingFace...")
        questions = []

        # ---- 1. GSM8K (80 questions) ----
        try:
            gsm = load_dataset("gsm8k", "main", split="test")
            gsm_samples = gsm.shuffle(seed=42).select(range(min(80, len(gsm))))
            for idx, item in enumerate(gsm_samples):
                questions.append({
                    "id": f"gsm8k_{idx}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "dataset": "gsm8k"
                })
            logger.info(f"Loaded {len(gsm_samples)} GSM8K questions.")
        except Exception as e:
            logger.warning(f"GSM8K load failed: {e}. Using mock.")
            self._add_mock_gsm8k(questions)

        # ---- 2. AQuA-RAT (40 questions) ----
        try:
            aqua = load_dataset("aqua_rat", split="train")
            aqua_samples = aqua.shuffle(seed=42).select(range(min(40, len(aqua))))
            for idx, item in enumerate(aqua_samples):
                options = item["options"]
                choice_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
                full_question = f"{item['question']}\n{choice_str}"
                answer_letter = item["correct"]
                questions.append({
                    "id": f"aqua_{idx}",
                    "question": full_question,
                    "answer": answer_letter,
                    "dataset": "aqua_rat"
                })
            logger.info(f"Loaded {len(aqua_samples)} AQuA-RAT questions.")
        except Exception as e:
            logger.warning(f"AQuA-RAT load failed: {e}. Using mock.")
            self._add_mock_aqua(questions)

        # ---- 3. StrategyQA (40 questions) ----
        try:
            sqa = load_dataset("ChilleD/StrategyQA", split="train")
            sqa_samples = sqa.shuffle(seed=42).select(range(min(40, len(sqa))))
            for idx, item in enumerate(sqa_samples):
                answer = "yes" if item["answer"] else "no"
                questions.append({
                    "id": f"strategyqa_{idx}",
                    "question": item["question"],
                    "answer": answer,
                    "dataset": "strategyqa"
                })
            logger.info(f"Loaded {len(sqa_samples)} StrategyQA questions.")
        except Exception as e:
            logger.warning(f"StrategyQA load failed: {e}. Using mock.")
            self._add_mock_strategyqa(questions)

        # ---- 4. MMLU (40 questions) ----
        try:
            mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
            total = len(mmlu)
            step = max(1, total // 40)
            indices = list(range(0, total, step))[:40]
            for idx, i in enumerate(indices):
                item = mmlu[i]
                choices = item["choices"]
                choice_str = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
                full_question = f"{item['question']}\n{choice_str}"
                answer_letter = ["A", "B", "C", "D"][item["answer"]]
                questions.append({
                    "id": f"mmlu_{idx}",
                    "question": full_question,
                    "answer": answer_letter,
                    "dataset": "mmlu"
                })
            logger.info(f"Loaded {len(indices)} MMLU questions.")
        except Exception as e:
            logger.warning(f"MMLU load failed: {e}. Using mock.")
            self._add_mock_mmlu(questions)

        logger.info(f"Total probed questions: {len(questions)}")
        return questions

    def _add_mock_gsm8k(self, questions):
        mock = [
            ("If Weng earns $12/hour for 5 hours, how much?", "60"),
            ("A farmer has 15 cows and 12 sheep. Total animals?", "27"),
            ("What is 15% of 240?", "36")
        ]
        for i, (q, a) in enumerate(mock):
            questions.append({"id": f"gsm8k_mock_{i}", "question": q, "answer": a, "dataset": "gsm8k"})

    def _add_mock_aqua(self, questions):
        mock = [
            ("What is 2+2? Options:\nA. 3\nB. 4\nC. 5\nD. 6", "B"),
            ("Solve for x: 2x = 10\nA. 2\nB. 5\nC. 10\nD. 20", "B")
        ]
        for i, (q, a) in enumerate(mock):
            questions.append({"id": f"aqua_mock_{i}", "question": q, "answer": a, "dataset": "aqua_rat"})

    def _add_mock_strategyqa(self, questions):
        mock = [
            ("Would a penguin survive in the Sahara?", "no"),
            ("Is a violin a string instrument?", "yes")
        ]
        for i, (q, a) in enumerate(mock):
            questions.append({"id": f"strategyqa_mock_{i}", "question": q, "answer": a, "dataset": "strategyqa"})

    def _add_mock_mmlu(self, questions):
        mock = [
            ("What is the capital of France?\nA. London\nB. Paris\nC. Rome\nD. Berlin", "B"),
            ("Which planet is known as the Red Planet?\nA. Earth\nB. Mars\nC. Jupiter\nD. Saturn", "B")
        ]
        for i, (q, a) in enumerate(mock):
            questions.append({"id": f"mmlu_mock_{i}", "question": q, "answer": a, "dataset": "mmlu"})

    def run(self, output_path="Outputs/difficulty_map.json", dry_run=False):
        setup_logger()
        logger.info("Starting Phase 0 Capability Probe (4 benchmarks)...")

        questions = self.load_probe_datasets()
        model = None
        tokenizer = None

        if not dry_run:
            model_name = self.model_cfg.get("name", "microsoft/Phi-3-mini-4k-instruct")
            logger.info(f"Loading model {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                )
        else:
            logger.info("DRY RUN – using mock generations.")

        self.mapper.build_map(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            n_samples=self.probe_cfg.get("n_samples", 5),
            temperature=self.probe_cfg.get("temperature", 0.8)
        )
        self.mapper.save_map(output_path)

        diffs = [v["difficulty"] for v in self.mapper.difficulty_map.values()]
        buckets = {
            "easy [0.0-0.3]": len([d for d in diffs if d <= 0.3]),
            "medium [0.4-0.7]": len([d for d in diffs if 0.4 <= d <= 0.7]),
            "hard [0.8-1.0]": len([d for d in diffs if d >= 0.8])
        }
        logger.info(f"Probe completed. Statistics: {buckets}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="phi3_mini")
    parser.add_argument("--hardware", default="kaggle")
    parser.add_argument("--output", default="Outputs/difficulty_map.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.hardware)
    probe = CapabilityProbe(config)
    probe.run(output_path=args.output, dry_run=args.dry_run)