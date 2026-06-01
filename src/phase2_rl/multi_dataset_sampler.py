"""
multi_dataset_sampler.py — Samples training questions from multiple datasets.
Provides a balanced mix of GSM8K (math), StrategyQA (logic), and MMLU (knowledge)
questions during RL training.

Why this is needed:
- Training only on GSM8K improves math but not logic or multi-choice reasoning.
- Samsung requires +5% on at least 2 of 3 benchmarks.
- Adding StrategyQA and MMLU data to the RL training mix is the fastest path
  to all-three-benchmark coverage.
"""

import random
from loguru import logger
from datasets import load_dataset
from typing import List, Dict, Optional


class MultiDatasetSampler:
    """
    Samples from GSM8K, StrategyQA, and MMLU for RL training.
    
    Mix ratios (configurable):
        - GSM8K: 50% (math reasoning — main strength)
        - StrategyQA: 30% (logical yes/no — weakest benchmark currently)
        - MMLU: 20% (multi-choice knowledge — broad coverage)
    
    All questions are formatted to produce <thought>/<answer> output.
    """

    def __init__(
        self,
        gsm8k_weight: float = 0.50,
        strategyqa_weight: float = 0.30,
        mmlu_weight: float = 0.20,
        n_preload: int = 500,  # questions to preload per dataset
        difficulty_mapper=None,
        curriculum=None,
    ):
        self.weights = {
            "gsm8k": gsm8k_weight,
            "strategyqa": strategyqa_weight,
            "mmlu": mmlu_weight,
        }
        self.datasets: Dict[str, List[dict]] = {}
        self.difficulty_mapper = difficulty_mapper
        self.curriculum = curriculum

        logger.info(
            f"MultiDatasetSampler: mix = "
            f"GSM8K {gsm8k_weight*100:.0f}% / "
            f"StrategyQA {strategyqa_weight*100:.0f}% / "
            f"MMLU {mmlu_weight*100:.0f}%"
        )
        self._preload(n_preload)

    def _preload(self, n: int):
        """Preload questions from each dataset into memory."""
        
        # ── GSM8K ──────────────────────────────────────────────────────────
        try:
            gsm_ds = load_dataset("openai/gsm8k", "main", split="train")
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
            # Try primary source
            sqa_ds = load_dataset("wics/strategy-qa", split="test")
            sqa_questions = []
            for item in sqa_ds:
                sqa_questions.append({
                    "question": item["question"],
                    "answer": str(item["answer"]),  # True/False → "True"/"False"
                    "dataset": "strategyqa",
                })
            random.shuffle(sqa_questions)
            self.datasets["strategyqa"] = sqa_questions[:n]
            logger.info(
                f"Preloaded {len(self.datasets['strategyqa'])} StrategyQA training questions."
            )
        except Exception as e1:
            logger.warning(f"Primary StrategyQA failed ({e1}), trying alternative...")
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
                logger.info(
                    f"Preloaded {len(self.datasets['strategyqa'])} StrategyQA questions "
                    f"(alternative source)."
                )
            except Exception as e2:
                logger.error(f"Both StrategyQA sources failed: {e2}")
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
                    "answer": answer_letter,   # "A", "B", "C", "D"
                    "dataset": "mmlu",
                })
            random.shuffle(mmlu_questions)
            self.datasets["mmlu"] = mmlu_questions[:n]
            logger.info(f"Preloaded {len(self.datasets['mmlu'])} MMLU training questions.")
        except Exception as e:
            logger.error(f"Failed to load MMLU: {e}")
            self.datasets["mmlu"] = []

    def sample_one(self) -> Optional[dict]:
        """
        Sample one question weighted by dataset mix ratios.
        Falls back to any available dataset if preferred one is empty.
        """
        # Weighted random dataset selection
        available = {k: v for k, v in self.datasets.items() if v}
        if not available:
            logger.error("All datasets are empty! Cannot sample.")
            return None

        # Filter weights to available datasets and renormalize
        avail_weights = {k: self.weights.get(k, 0.33) for k in available}
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

    def _get_strategyqa(self) -> Optional[dict]:
        """Get a random StrategyQA question."""
        if self.datasets.get("strategyqa"):
            return random.choice(self.datasets["strategyqa"])
        return None

    def _get_mmlu(self) -> Optional[dict]:
        """Get a random MMLU question."""
        if self.datasets.get("mmlu"):
            return random.choice(self.datasets["mmlu"])
        return None

    def sample_batch(self, batch_size: int = 1) -> List[dict]:
        """Sample a batch of questions."""
        return [q for q in (self.sample_one() for _ in range(batch_size)) if q]
