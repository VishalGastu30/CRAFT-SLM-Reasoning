"""
multi_dataset_sampler.py — Samples training questions from multiple datasets.
Provides a balanced mix of GSM8K (math), StrategyQA (logic), MMLU (knowledge),
and AQuA-RAT (algebraic reasoning) during RL training.

Mix ratios:
    - GSM8K:      40% (math reasoning)
    - StrategyQA: 25% (logical yes/no)
    - MMLU:       20% (multi‑choice knowledge)
    - AQuA-RAT:   15% (algebraic reasoning, letter answers A‑E)
"""

import random
from loguru import logger
from datasets import load_dataset
from typing import List, Dict, Optional


class MultiDatasetSampler:
    """
    Samples from GSM8K, StrategyQA, MMLU, and AQuA-RAT for RL training.
    """

    def __init__(
        self,
        gsm8k_weight: float = 0.40,
        strategyqa_weight: float = 0.25,
        mmlu_weight: float = 0.20,
        aqua_weight: float = 0.15,
        n_preload: int = 500,
        difficulty_mapper=None,
        curriculum=None,
    ):
        self.weights = {
            "gsm8k": gsm8k_weight,
            "strategyqa": strategyqa_weight,
            "mmlu": mmlu_weight,
            "aqua": aqua_weight,
        }
        self.datasets: Dict[str, List[dict]] = {}
        self.difficulty_mapper = difficulty_mapper
        self.curriculum = curriculum

        logger.info(
            f"MultiDatasetSampler: mix = "
            f"GSM8K {gsm8k_weight*100:.0f}% / "
            f"StrategyQA {strategyqa_weight*100:.0f}% / "
            f"MMLU {mmlu_weight*100:.0f}% / "
            f"AQuA-RAT {aqua_weight*100:.0f}%"
        )
        self._preload(n_preload)

    def _preload(self, n: int):
        """Preload questions from each dataset into memory."""
        
        # ── GSM8K ──────────────────────────────────────────────────────────
        try:
            gsm_ds = load_dataset("gsm8k", "main", split="train")
            gsm_questions = []
            for item in gsm_ds:
                gsm_questions.append({
                    "question": item["question"],
                    "answer": item["answer"],   # Full text with "#### N"
                    "dataset": "gsm8k",
                })
            random.shuffle(gsm_questions)
            self.datasets["gsm8k"] = gsm_questions[:n]
            logger.info(f"Preloaded {len(self.datasets['gsm8k'])} GSM8K training questions.")
        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            self.datasets["gsm8k"] = []

        # ── StrategyQA ─────────────────────────────────────────────────────
        try:
            sqa_ds = load_dataset("ChilleD/StrategyQA", split="train")
            sqa_questions = []
            for item in sqa_ds:
                sqa_questions.append({
                    "question": item["question"],
                    "answer": str(item["answer"]),
                    "dataset": "strategyqa",
                })
            random.shuffle(sqa_questions)
            self.datasets["strategyqa"] = sqa_questions[:n]
            logger.info(f"Preloaded {len(self.datasets['strategyqa'])} StrategyQA training questions.")
        except Exception as e:
            logger.error(f"Failed to load StrategyQA: {e}")
            self.datasets["strategyqa"] = []

        # ── MMLU ────────────────────────────────────────────────────────────
        try:
            mmlu_ds = load_dataset("cais/mmlu", "all", split="validation")
            mmlu_questions = []
            for item in mmlu_ds:
                choices = item["choices"]
                choices_text = "\n".join([
                    f"A. {choices[0]}", f"B. {choices[1]}",
                    f"C. {choices[2]}", f"D. {choices[3]}",
                ])
                full_question = f"{item['question']}\n\n{choices_text}"
                answer_letter = ["A", "B", "C", "D"][item["answer"]]
                mmlu_questions.append({
                    "question": full_question,
                    "answer": answer_letter,
                    "dataset": "mmlu",
                })
            random.shuffle(mmlu_questions)
            self.datasets["mmlu"] = mmlu_questions[:n]
            logger.info(f"Preloaded {len(self.datasets['mmlu'])} MMLU training questions.")
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}")
            self.datasets["mmlu"] = []

        # ── AQuA‑RAT ────────────────────────────────────────────────────────
        try:
            aqua_ds = load_dataset("aqua_rat", split="train")
            aqua_questions = []
            for item in aqua_ds:
                # Format options as A. ... B. ... C. ... D. ... E. ...
                choices_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(item["options"])])
                full_question = f"{item['question']}\n\n{choices_text}"
                aqua_questions.append({
                    "question": full_question,
                    "answer": item["correct"],   # Already "A", "B", "C", "D", "E"
                    "dataset": "aqua",
                })
            random.shuffle(aqua_questions)
            self.datasets["aqua"] = aqua_questions[:n]
            logger.info(f"Preloaded {len(self.datasets['aqua'])} AQuA‑RAT training questions.")
        except Exception as e:
            logger.error(f"Failed to load AQuA‑RAT: {e}")
            self.datasets["aqua"] = []

    def sample_one(self) -> Optional[dict]:
        """
        Sample one question weighted by dataset mix ratios.
        Falls back to any available dataset if preferred one is empty.
        """
        available = {k: v for k, v in self.datasets.items() if v}
        if not available:
            logger.error("All datasets are empty! Cannot sample.")
            return None

        # Filter weights to available datasets and renormalize
        avail_weights = {k: self.weights.get(k, 0.25) for k in available}
        total_weight = sum(avail_weights.values())
        if total_weight == 0:
            chosen_dataset = random.choice(list(available.keys()))
        else:
            r = random.random() * total_weight
            cumulative = 0.0
            chosen_dataset = list(available.keys())[-1]
            for dataset_name, w in avail_weights.items():
                cumulative += w
                if r <= cumulative:
                    chosen_dataset = dataset_name
                    break

        question = random.choice(available[chosen_dataset])
        return question

    def sample_batch(self, batch_size: int = 1) -> List[dict]:
        """Sample a batch of questions."""
        return [q for q in (self.sample_one() for _ in range(batch_size)) if q]