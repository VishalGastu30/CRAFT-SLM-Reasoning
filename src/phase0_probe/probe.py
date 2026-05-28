import os
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from loguru import logger
from src.utils.logger import setup_logger
from src.utils.hardware_detector import detect_hardware
from src.config import load_config
from src.phase0_probe.difficulty_mapper import DifficultyMapper

class CapabilityProbe:
    """
    Main runner class for Phase 0 Capability Probe.
    Downloads the probe datasets, samples questions, initializes the model,
    and runs the DifficultyMapper to produce difficulty_map.json.
    """
    def __init__(self, config):
        self.config = config
        self.probe_cfg = config.get("probe", {})
        self.model_cfg = config.get("model", {})
        self.mapper = DifficultyMapper()

    def load_probe_datasets(self):
        """
        Loads GSM8K, StrategyQA, and MMLU datasets and samples questions.
        Returns a list of questions: [{"id": str, "question": str, "answer": str, "dataset": str}]
        """
        logger.info("Loading probing datasets from HuggingFace...")
        questions = []

        # 1. GSM8K (80 questions)
        try:
            gsm8k = load_dataset("gsm8k", "main", split="test")
            gsm8k_samples = gsm8k.shuffle(seed=42).select(range(min(80, len(gsm8k))))
            for idx, item in enumerate(gsm8k_samples):
                questions.append({
                    "id": f"gsm8k_{idx}",
                    "question": item["question"],
                    "answer": item["answer"],
                    "dataset": "gsm8k"
                })
            logger.info(f"Loaded {len(gsm8k_samples)} GSM8K questions.")
        except Exception as e:
            logger.warning(f"Failed to load GSM8K from HuggingFace: {e}. Using mock questions.")
            self._add_mock_gsm8k(questions)

        # 2. StrategyQA (60 questions)
        try:
            # Load StrategyQA via wza/strategyqa
            sqa = load_dataset("wza/strategyqa", split="train")
            sqa_samples = sqa.shuffle(seed=42).select(range(min(60, len(sqa))))
            for idx, item in enumerate(sqa_samples):
                questions.append({
                    "id": f"strategyqa_{idx}",
                    "question": item["question"],
                    "answer": "yes" if item["answer"] else "no",
                    "dataset": "strategyqa"
                })
            logger.info(f"Loaded {len(sqa_samples)} StrategyQA questions.")
        except Exception as e:
            logger.warning(f"Failed to load StrategyQA: {e}. Using mock questions.")
            self._add_mock_strategyqa(questions)

        # 3. MMLU (60 questions)
        try:
            mmlu = load_dataset("cais/mmlu", "college_mathematics", split="test")
            mmlu_samples = mmlu.shuffle(seed=42).select(range(min(60, len(mmlu))))
            for idx, item in enumerate(mmlu_samples):
                # Convert multiple choice to simple letter answer
                choices = item["choices"]
                correct_idx = item["answer"]
                answer_letter = ["A", "B", "C", "D"][correct_idx]
                
                # Format question with choices
                choice_str = "\n".join([f"{l}. {c}" for l, c in zip(["A", "B", "C", "D"], choices)])
                full_q = f"{item['question']}\nChoices:\n{choice_str}"
                
                questions.append({
                    "id": f"mmlu_{idx}",
                    "question": full_q,
                    "answer": answer_letter,
                    "dataset": "mmlu"
                })
            logger.info(f"Loaded {len(mmlu_samples)} MMLU questions.")
        except Exception as e:
            logger.warning(f"Failed to load MMLU: {e}. Using mock questions.")
            self._add_mock_mmlu(questions)

        return questions

    def _add_mock_gsm8k(self, questions):
        mock_data = [
            ("If Weng earns $12 an hour baby-sitting and works 5 hours, how much does she earn?", "60"),
            ("A farmer has 15 cows and 12 sheep. How many animals does he have in total?", "27"),
            ("What is 15% of 240?", "36"),
            ("A train travels at 60 mph for 3 hours. How far does it travel in miles?", "180"),
            ("John buys 3 books for $10 each and a bag for $15. How much did he spend in total?", "45")
        ]
        for idx, (q, a) in enumerate(mock_data):
            questions.append({
                "id": f"gsm8k_mock_{idx}",
                "question": q,
                "answer": a,
                "dataset": "gsm8k"
            })

    def _add_mock_strategyqa(self, questions):
        mock_data = [
            ("Would a penguin survive in the Sahara desert?", "no"),
            ("Can a human run faster than a cheetah?", "no"),
            ("Is a violin a string instrument?", "yes"),
            ("Can you write a book on a typewriter?", "yes")
        ]
        for idx, (q, a) in enumerate(mock_data):
            questions.append({
                "id": f"strategyqa_mock_{idx}",
                "question": q,
                "answer": a,
                "dataset": "strategyqa"
            })

    def _add_mock_mmlu(self, questions):
        mock_data = [
            ("What is the derivative of x^2 with respect to x?\nA. 2x\nB. x\nC. 2\nD. 1", "A"),
            ("Which of the following is a prime number?\nA. 4\nB. 9\nC. 11\nD. 15", "C")
        ]
        for idx, (q, a) in enumerate(mock_data):
            questions.append({
                "id": f"mmlu_mock_{idx}",
                "question": q,
                "answer": a,
                "dataset": "mmlu"
            })

    def run(self, output_path="difficulty_map.json", dry_run=False):
        """
        Executes the capability probe.
        If dry_run=True, runs without loading the model (useful for local verification).
        """
        setup_logger()
        logger.info("Starting Phase 0 Capability Probe...")
        
        questions = self.load_probe_datasets()
        
        model = None
        tokenizer = None
        
        if not dry_run:
            model_name = self.model_cfg.get("name", "microsoft/Phi-3-mini-4k-instruct")
            logger.info(f"Loading model {model_name} for capability probe...")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load in 4-bit for speed and resource limits
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Phase 0 is inference-only (no training), so we load in bfloat16
            # without quantization. This avoids any bitsandbytes CUDA dependency.
            # Phi-3-mini in bfloat16 = ~7.6GB — fits easily in Kaggle's 32GB VRAM.
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
            logger.info("Running in DRY RUN mode. Using simulated/mock generations.")

        # Probe model to build map
        self.mapper.build_map(
            model=model,
            tokenizer=tokenizer,
            questions=questions,
            n_samples=self.probe_cfg.get("n_samples", 5),
            temperature=self.probe_cfg.get("temperature", 0.8)
        )
        
        # Save map
        self.mapper.save_map(output_path)
        
        # Log bucket statistics
        diffs = [v["difficulty"] for v in self.mapper.difficulty_map.values()]
        buckets = {
            "easy [0.0-0.3]": len([d for d in diffs if d <= 0.3]),
            "medium [0.4-0.7]": len([d for d in diffs if 0.4 <= d <= 0.7]),
            "hard [0.8-1.0]": len([d for d in diffs if d >= 0.8])
        }
        logger.info(f"Capability Probe completed. Statistics: {buckets}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 0: Capability Probe")
    parser.add_argument("--config", type=str, default="phi3_mini", help="Model config name")
    parser.add_argument("--hardware", type=str, default="kaggle", help="Hardware profile name")
    parser.add_argument("--output", type=str, default="difficulty_map.json", help="Output path")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading active model weights")
    args = parser.parse_args()

    config = load_config(args.config, args.hardware)
    probe = CapabilityProbe(config)
    probe.run(output_path=args.output, dry_run=args.dry_run)
