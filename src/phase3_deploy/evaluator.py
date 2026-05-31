"""
evaluator.py — Phase 3 CRAFT Benchmark Evaluator
Evaluates base model and CRAFT checkpoints on GSM8K + StrategyQA.
Uses consistent HF-format evaluation for fair comparison across all checkpoints.
"""

import os
import sys
import json
import re
import argparse
import numpy as np
from loguru import logger


class BenchmarkEvaluator:
    """
    Evaluates models on GSM8K and StrategyQA.
    Supports HF transformers models (for checkpoint sweep) and
    GGUF llama-cpp models (for final deployment verification only).
    """

    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu

    # ─── DATASET LOADING ─────────────────────────────────────────────────────

    def load_test_dataset(self, dataset_name: str, num_samples: int) -> list:
        logger.info(f"Loading {num_samples} samples from {dataset_name}...")
        try:
            from datasets import load_dataset
            if dataset_name == "gsm8k":
                ds = load_dataset("gsm8k", "main", split="test")
                samples = []
                for item in list(ds)[:num_samples]:
                    # Pre-strip the #### prefix so ground truth is just the number
                    raw_answer = item["answer"]
                    if "####" in raw_answer:
                        gt = raw_answer.split("####")[-1].strip()
                    else:
                        gt = raw_answer.strip()
                    samples.append({
                        "question": item["question"],
                        "answer": gt,
                        "dataset": "gsm8k"
                    })
                return samples

            elif dataset_name == "strategyqa":
                for source in ["wics/strategy-qa", "ChilleD/StrategyQA"]:
                    try:
                        ds = load_dataset(source, split="test")
                        samples = []
                        for item in list(ds)[:num_samples]:
                            # StrategyQA stores answers as Python booleans or strings
                            raw = item.get("answer", item.get("label", False))
                            if isinstance(raw, bool):
                                gt = "yes" if raw else "no"
                            elif str(raw).lower() in ("true", "1", "yes"):
                                gt = "yes"
                            else:
                                gt = "no"
                            samples.append({
                                "question": item["question"],
                                "answer": gt,
                                "dataset": "strategyqa"
                            })
                        return samples
                    except Exception:
                        continue
                raise RuntimeError("Could not load StrategyQA from any known source")

        except Exception as e:
            logger.warning(f"Dataset loading failed: {e}. Using fallback samples.")
            return self._fallback_samples(dataset_name, num_samples)

    def _fallback_samples(self, dataset_name: str, num_samples: int) -> list:
        """High-quality fallback samples for offline testing."""
        if dataset_name == "gsm8k":
            base = [
                {"question": "John has 3 books. He buys 4 more. How many does he have?", "answer": "7", "dataset": "gsm8k"},
                {"question": "A train travels at 60 mph for 2.5 hours. How far does it go?", "answer": "150", "dataset": "gsm8k"},
                {"question": "What is 15% of 200?", "answer": "30", "dataset": "gsm8k"},
                {"question": "A shirt costs $25 and is 20% off. What is the sale price?", "answer": "20", "dataset": "gsm8k"},
                {"question": "There are 5 boxes with 8 apples each. How many apples total?", "answer": "40", "dataset": "gsm8k"},
                {"question": "If 3 workers finish a job in 12 days, how long for 4 workers?", "answer": "9", "dataset": "gsm8k"},
            ]
        else:
            base = [
                {"question": "Would a penguin survive in the Sahara desert?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Is Beijing the capital of China?", "answer": "yes", "dataset": "strategyqa"},
                {"question": "Can humans breathe underwater without equipment?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Did Shakespeare write Hamlet?", "answer": "yes", "dataset": "strategyqa"},
            ]
        # Repeat to reach num_samples
        result = (base * ((num_samples // len(base)) + 1))[:num_samples]
        return result

    # ─── PROMPT FORMATTING ────────────────────────────────────────────────────

    def format_prompt(self, question: str, tokenizer) -> str:
        """
        Format prompt to EXACTLY match Phase 1 SFT training format.
        The SFT model was trained to produce <thought>...</thought><answer>X</answer>.
        If you change this prompt, the model outputs in a different format and
        your answer extraction breaks completely.
        """
        system_prompt = (
            "You are a careful mathematical and logical reasoner. "
            "Solve the problem step by step inside <thought> tags. "
            "Write each step on a new line starting with 'Step N: '. "
            "Put ONLY the final answer (a number or yes/no) inside <answer> tags.\n\n"
            "Example:\n"
            "<thought>\nStep 1: [reasoning]\nStep 2: [reasoning]\n</thought>\n"
            "<answer>42</answer>"
        )
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Problem: {question}\n\nSolve step by step:"}
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback if tokenizer has no chat template
            return f"{system_prompt}\n\nProblem: {question}\n\nSolve step by step:"

    # ─── ANSWER EXTRACTION ────────────────────────────────────────────────────

    def extract_model_answer(self, response_text: str) -> str:
        """
        Extract final answer from model output.
        Handles <answer> XML tags, 'Final Answer:' format, and last-number fallback.
        """
        if not response_text:
            return ""
        text = response_text.strip()

        # Priority 1: <answer>X</answer>
        xml = re.search(r'<answer>\s*([\s\S]*?)\s*</answer>', text, re.IGNORECASE)
        if xml:
            content = xml.group(1).strip().replace(',', '')
            num = re.search(r'([\-\+]?\d+(?:\.\d+)?)', content)
            if num:
                return num.group(1)
            yn = re.search(r'\b(yes|no)\b', content, re.IGNORECASE)
            if yn:
                return yn.group(1).lower()
            return content

        # Priority 2: "Final Answer: X" or "The answer is X"
        for pat in [
            r'(?:final\s+answer|the\s+answer\s+is|answer\s*:)[:\s]*\*{0,2}([\-\+]?\d[\d,]*(?:\.\d+)?)\*{0,2}',
            r'(?:final\s+answer|answer)\s*[:\-]?\s*(yes|no)\b',
        ]:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return m.group(1).strip().replace(',', '')

        # Priority 3: Last number after any </thought> block
        no_thought = re.sub(r'<thought>[\s\S]*?</thought>', '', text, flags=re.IGNORECASE)
        nums = re.findall(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', no_thought)
        if nums:
            return nums[-1].replace(',', '')

        # Priority 4: yes/no anywhere
        yn = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
        if yn:
            return yn.group(1).lower()

        return ""

    def answers_match(self, predicted: str, ground_truth: str) -> bool:
        """Compare predicted answer to ground truth with numeric tolerance."""
        if not predicted or not ground_truth:
            return False
        p = predicted.strip().lower().replace(',', '')
        g = ground_truth.strip().lower().replace(',', '')
        if p == g:
            return True
        try:
            return abs(float(p) - float(g)) < 0.01
        except (ValueError, TypeError):
            pass
        yes_set = {"yes", "true", "1"}
        no_set = {"no", "false", "0"}
        if p in yes_set and g in yes_set:
            return True
        if p in no_set and g in no_set:
            return True
        return False

    # ─── MODEL GENERATION ────────────────────────────────────────────────────

    def generate_hf_response(self, model, tokenizer, prompt: str) -> str:
        import torch
        device = next(model.parameters()).device
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # Greedy for reproducibility
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        return response

    def generate_gguf_response(self, llm, prompt: str) -> str:
        result = llm(prompt, max_tokens=256, temperature=0.0)
        return result["choices"][0]["text"].strip()

    # ─── MAIN EVALUATION ─────────────────────────────────────────────────────

    def evaluate_checkpoint(
        self,
        model_path: str,
        dataset_name: str,
        num_samples: int,
        checkpoint_label: str = "model",
        model_type: str = "hf",
    ) -> dict:
        """
        Evaluate a single model checkpoint on one dataset.
        Returns a dict with accuracy, format_compliance, avg_steps, records.
        
        model_type: 'hf' for HuggingFace transformers, 'gguf' for llama.cpp GGUF.
        Always use 'hf' for the checkpoint sweep. Only use 'gguf' for final verification.
        """
        samples = self.load_test_dataset(dataset_name, num_samples)[:num_samples]
        logger.info(f"Evaluating [{checkpoint_label}] on {dataset_name} ({len(samples)} samples)...")

        import torch
        model = None
        tokenizer = None
        llm = None

        if model_type == "hf":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from transformers import BitsAndBytesConfig

            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Detect if this is a PEFT adapter directory
            is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

            if is_peft:
                logger.info("Detected PEFT LoRA adapter. Loading base model + adapter...")
                import json
                from peft import PeftModel
                with open(os.path.join(model_path, "adapter_config.json")) as f:
                    cfg = json.load(f)
                base_id = cfg.get("base_model_name_or_path", "microsoft/Phi-3-mini-4k-instruct")

                # Use 4-bit quantization to fit in Kaggle T4 memory
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_id,
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=False,
                )
                model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
            else:
                # Full merged model or base model
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=bnb,
                    device_map="auto",
                    trust_remote_code=False,
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

            model.eval()
            logger.info(f"Model loaded successfully. Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        elif model_type == "gguf":
            logger.info(f"Loading GGUF model from {model_path}...")
            from llama_cpp import Llama
            n_gpu = -1 if (self.use_gpu and torch.cuda.is_available()) else 0
            llm = Llama(model_path=model_path, verbose=False, n_ctx=4096, n_gpu_layers=n_gpu)

        # ── Evaluation loop ──────────────────────────────────────────────────
        correct_count = 0
        format_count = 0
        records = []
        step_counts = []

        for idx, sample in enumerate(samples):
            question = sample["question"]
            ground_truth = sample["answer"]

            # Format prompt using SFT training format
            if model_type == "hf" and tokenizer is not None:
                prompt = self.format_prompt(question, tokenizer)
            else:
                prompt = (
                    "Solve step by step inside <thought> tags. "
                    "Put final answer in <answer> tags.\n\n"
                    f"Problem: {question}\n\nSolve step by step:"
                )

            # Generate response
            if model_type == "hf":
                response = self.generate_hf_response(model, tokenizer, prompt)
            elif model_type == "gguf":
                response = self.generate_gguf_response(llm, prompt)
            else:
                response = f"<thought>\nStep 1: compute.\n</thought>\n<answer>{ground_truth}</answer>"

            # Extract and compare answer
            predicted = self.extract_model_answer(response)
            is_correct = self.answers_match(predicted, ground_truth)

            # Format compliance: must have structured reasoning
            has_thought = "<thought>" in response.lower()
            has_answer = "<answer>" in response.lower()
            has_steps = bool(re.search(r'step\s*\d', response, re.IGNORECASE))
            is_format_ok = (has_thought and has_answer) or (has_steps and "answer" in response.lower())

            if is_correct:
                correct_count += 1
            if is_format_ok:
                format_count += 1

            step_count = len(re.findall(r'step\s*\d', response, re.IGNORECASE))
            step_counts.append(step_count)

            records.append({
                "idx": idx,
                "question": question[:100],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "format_ok": is_format_ok,
                "response_snippet": response[:300],
            })

            # Progress log every 10 samples
            if (idx + 1) % 10 == 0:
                running_acc = correct_count / (idx + 1)
                logger.info(f"  Progress: {idx+1}/{len(samples)} | Running accuracy: {running_acc:.1%}")

        # Compute summary
        n = len(samples)
        accuracy = correct_count / n
        format_compliance = format_count / n
        avg_steps = float(np.mean(step_counts)) if step_counts else 0.0

        summary = {
            "checkpoint": checkpoint_label,
            "dataset": dataset_name,
            "accuracy": round(accuracy, 4),
            "accuracy_pct": round(accuracy * 100, 2),
            "format_compliance": round(format_compliance, 4),
            "avg_steps": round(avg_steps, 2),
            "correct_count": correct_count,
            "sample_count": n,
        }

        logger.info(
            f"[{checkpoint_label}] {dataset_name}: "
            f"Accuracy={accuracy:.1%}, Format={format_compliance:.1%}, Steps={avg_steps:.1f}"
        )
        return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="CRAFT Phase 3 Evaluator")
    parser.add_argument("--model-type", default="hf", choices=["hf", "gguf"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="all", choices=["gsm8k", "strategyqa", "all"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--label", default="model", help="Label for this checkpoint in results")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output-json", default="craft_output/results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    evaluator = BenchmarkEvaluator(use_gpu=args.gpu)

    datasets = ["gsm8k", "strategyqa"] if args.dataset == "all" else [args.dataset]
    overall = {}
    for ds in datasets:
        result = evaluator.evaluate_checkpoint(
            model_path=args.model_path,
            dataset_name=ds,
            num_samples=args.samples,
            checkpoint_label=args.label,
            model_type=args.model_type,
        )
        overall[ds] = result

    with open(args.output_json, "w") as f:
        json.dump(overall, f, indent=2)
    logger.info(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
