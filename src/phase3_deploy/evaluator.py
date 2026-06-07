"""
evaluator.py — Phase 3 CRAFT Benchmark Evaluator
Evaluates base model and CRAFT checkpoints on GSM8K, StrategyQA, and MMLU.
Uses consistent plain‑text format matching Phase 1 SFT (no XML tags).
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
    Evaluates models on GSM8K, StrategyQA, and MMLU.
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

            elif dataset_name == "mmlu":
                ds = load_dataset("cais/mmlu", "all", split="test")
                total = len(ds)
                indices = list(range(0, total, max(1, total // num_samples)))[:num_samples]
                samples = []
                for idx in indices:
                    item = ds[idx]
                    choices = item["choices"]
                    choices_text = "\n".join([
                        f"A. {choices[0]}",
                        f"B. {choices[1]}",
                        f"C. {choices[2]}",
                        f"D. {choices[3]}",
                    ])
                    full_question = f"{item['question']}\n\n{choices_text}"
                    answer_letter = ["A", "B", "C", "D"][item["answer"]]
                    samples.append({
                        "question": full_question,
                        "answer": answer_letter,
                        "dataset": "mmlu"
                    })
                return samples

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
        elif dataset_name == "strategyqa":
            base = [
                {"question": "Would a penguin survive in the Sahara desert?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Is Beijing the capital of China?", "answer": "yes", "dataset": "strategyqa"},
                {"question": "Can humans breathe underwater without equipment?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Did Shakespeare write Hamlet?", "answer": "yes", "dataset": "strategyqa"},
            ]
        else:  # mmlu
            base = [
                {"question": "What is the capital of France?\n\nA. London\nB. Paris\nC. Rome\nD. Berlin", "answer": "B", "dataset": "mmlu"},
                {"question": "Which planet is known as the Red Planet?\n\nA. Earth\nB. Mars\nC. Jupiter\nD. Saturn", "answer": "B", "dataset": "mmlu"},
                {"question": "What is the square root of 64?\n\nA. 6\nB. 7\nC. 8\nD. 9", "answer": "C", "dataset": "mmlu"},
                {"question": "Who wrote 'Romeo and Juliet'?\n\nA. William Shakespeare\nB. Charles Dickens\nC. Jane Austen\nD. Mark Twain", "answer": "A", "dataset": "mmlu"}
            ]
        result = (base * ((num_samples // len(base)) + 1))[:num_samples]
        return result

    # ─── PROMPT FORMATTING (Plain Text, No XML Tags) ─────────────────────────

    def format_prompt(self, question: str, dataset: str, tokenizer) -> str:
        """
        Format prompt to EXACTLY match Phase 1 SFT and Phase 2 RL training format.
        Model expects: numbered steps and "Final Answer: X".
        """
        if dataset == "strategyqa":
            system = (
                "You are a reasoning assistant. Answer the yes/no question step by step. "
                "You MUST format your explanation as a sequence of steps starting with "
                "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: yes' or 'Final Answer: no'."
            )
        elif dataset == "mmlu":
            system = (
                "You are a reasoning assistant. Answer the multiple-choice question step by step. "
                "You MUST format your explanation as a sequence of steps starting with "
                "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: A' (or B, C, D)."
            )
        else:
            system = (
                "You are a reasoning assistant. Solve the math problem step by step. "
                "You MUST format your explanation as a sequence of steps starting with "
                "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: [value]' "
                "where [value] is the short final answer."
            )
        try:
            messages = [{"role": "system", "content": system},
                        {"role": "user", "content": f"Problem: {question}\n\nSolve step by step:"}]
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{system}\n\nProblem: {question}\n\nSolve step by step:"

    # ─── ANSWER EXTRACTION (No dataset-specific mapping; returns raw string) ──

    def extract_model_answer(self, response_text: str) -> str:
        """
        Extract final answer from model output as a raw string.
        Prioritises 'Final Answer:' exactly as trained, then bare patterns,
        then semantic answers (yes/no or letter) anywhere, and only last number
        as a last resort. Does NOT perform dataset-specific mapping (e.g., 1→A).
        That mapping is done later in evaluate_checkpoint based on dataset.
        """
        if not response_text:
            return ""
        text = response_text.strip()

        # Priority 1: "Final Answer: X"
        final_match = re.search(r'Final Answer:\s*([^\n]+)', text, re.IGNORECASE)
        if final_match:
            ans = final_match.group(1).strip()
            # Extract number (e.g., "6 apples" → "6")
            num = re.search(r'([+\-]?\d[\d,]*(?:\.\d+)?)', ans)
            if num:
                return num.group(1).replace(',', '')
            # Extract yes/no
            yn = re.search(r'\b(yes|no)\b', ans, re.IGNORECASE)
            if yn:
                return yn.group(1).lower()
            # Extract letter (A, B, C, D) – case-insensitive
            letter = re.search(r'\b([A-Da-d])\b', ans)
            if letter:
                return letter.group(1).upper()
            return ans

        # Priority 2: Bare yes/no or letter as entire response
        bare_yesno = re.match(r'^\s*(yes|no)[\.\!\?]?\s*$', text, re.IGNORECASE)
        if bare_yesno:
            return bare_yesno.group(1).lower()
        bare_letter = re.match(r'^\s*([A-Da-d])[\.\!\?]?\s*$', text)
        if bare_letter:
            return bare_letter.group(1).upper()

        # Priority 3: <answer> tags (backward compatibility)
        xml = re.search(r'<answer>\s*([\s\S]*?)\s*</answer>', text, re.IGNORECASE)
        if xml:
            content = xml.group(1).strip().replace(',', '')
            num = re.search(r'([+\-]?\d+(?:\.\d+)?)', content)
            if num:
                return num.group(1)
            yn = re.search(r'\b(yes|no)\b', content, re.IGNORECASE)
            if yn:
                return yn.group(1).lower()
            letter = re.search(r'^([A-Da-d])\.?$', content.strip())
            if letter:
                return letter.group(1).upper()
            return content

        # Priority 4: yes/no anywhere (before numbers)
        yn2 = re.search(r'\b(yes|no)\b', text, re.IGNORECASE)
        if yn2:
            return yn2.group(1).lower()

        # Priority 5: single letter A-D anywhere (case-insensitive) – fixes baseline MMLU extraction
        let = re.search(r'\b([A-Da-d])\b', text)
        if let:
            return let.group(1).upper()

        # Priority 6: last number anywhere (fallback for GSM8K and numeric baselines)
        nums = re.findall(r'([+\-]?\d[\d,]*(?:\.\d+)?)', text)
        if nums:
            return nums[-1].replace(',', '')

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
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,  # greedy for reproducibility
                temperature=1.0,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response

    def generate_gguf_response(self, llm, prompt: str) -> str:
        result = llm(
            prompt,
            max_tokens=256,
            temperature=0.0,
            repeat_penalty=1.1,
            stop=["<|end|>", "<|user|>", "<|assistant|>"]
        )
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

            is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

            if is_peft:
                logger.info("Detected PEFT LoRA adapter. Loading base model + adapter...")
                import json
                from peft import PeftModel
                with open(os.path.join(model_path, "adapter_config.json")) as f:
                    cfg = json.load(f)
                base_id = cfg.get("base_model_name_or_path", "microsoft/Phi-3-mini-4k-instruct")
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_id, quantization_config=bnb, device_map="auto", trust_remote_code=False
                )
                model = PeftModel.from_pretrained(base_model, model_path, is_trainable=False)
            else:
                bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, quantization_config=bnb, device_map="auto", trust_remote_code=False
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

        # Build system prompt for GGUF case (plain text)
        if dataset_name == "strategyqa":
            system_prompt = (
                "You are a reasoning assistant. Answer the yes/no question step by step. "
                "Format steps as 'Step 1:', 'Step 2:', etc. End with 'Final Answer: yes' or 'Final Answer: no'."
            )
        elif dataset_name == "mmlu":
            system_prompt = (
                "You are a reasoning assistant. Answer the multiple-choice question step by step. "
                "Format steps as 'Step 1:', 'Step 2:', etc. End with 'Final Answer: A' (or B, C, D)."
            )
        else:
            system_prompt = (
                "You are a reasoning assistant. Solve the math problem step by step. "
                "Format steps as 'Step 1:', 'Step 2:', etc. End with 'Final Answer: [value]'."
            )

        # Evaluation loop
        correct_count = 0
        format_count = 0
        records = []
        step_counts = []

        for idx, sample in enumerate(samples):
            question = sample["question"]
            ground_truth = sample["answer"]

            # Build prompt
            if model_type == "gguf":
                prompt = (
                    f"<|system|>\n{system_prompt}<|end|>\n"
                    f"<|user|>\nProblem: {question}\n\nSolve step by step:<|end|>\n"
                    f"<|assistant|>\n"
                )
            elif model_type == "hf" and tokenizer is not None:
                prompt = self.format_prompt(question, dataset_name, tokenizer)
            else:
                prompt = f"{system_prompt}\n\nProblem: {question}\n\nSolve step by step:"

            # Generate response
            if model_type == "hf":
                response = self.generate_hf_response(model, tokenizer, prompt)
            elif model_type == "gguf":
                response = self.generate_gguf_response(llm, prompt)
            else:
                response = f"Step 1: compute.\nFinal Answer: {ground_truth}"

            # Extract raw answer (no dataset-specific mapping yet)
            predicted_raw = self.extract_model_answer(response)

            # --- FIX for GSM8K: if extraction returned "yes"/"no", it's a stray word from reasoning.
            # Re-extract using only number-finding logic (last number in response).
            if dataset_name == "gsm8k" and predicted_raw.lower() in ("yes", "no"):
                nums = re.findall(r'([+\-]?\d[\d,]*(?:\.\d+)?)', response)
                if nums:
                    predicted_raw = nums[-1].replace(',', '')
                else:
                    predicted_raw = ""

            # --- DATASET-SPECIFIC POST-PROCESSING (fixes baseline numeric outputs) ---
            predicted = predicted_raw
            if dataset_name == "mmlu":
                # Map single digit 1-4 to letter
                digit_to_letter = {"1": "A", "2": "B", "3": "C", "4": "D"}
                if predicted in digit_to_letter:
                    predicted = digit_to_letter[predicted]
            elif dataset_name == "strategyqa":
                # Map "1" -> "yes", "0" -> "no"
                if predicted == "1":
                    predicted = "yes"
                elif predicted == "0":
                    predicted = "no"
                # Map boolean strings
                elif predicted.lower() == "true":
                    predicted = "yes"
                elif predicted.lower() == "false":
                    predicted = "no"
            # GSM8K: no mapping needed

            is_correct = self.answers_match(predicted, ground_truth)

            # Format compliance: must have steps and final answer
            has_steps = bool(re.search(r'step\s*\d', response, re.IGNORECASE))
            has_final = bool(re.search(r'final answer', response, re.IGNORECASE))
            is_format_ok = has_steps and has_final

            if is_correct:
                correct_count += 1
            if is_format_ok:
                format_count += 1

            step_count = len(re.findall(r'step\s*\d', response, re.IGNORECASE))
            step_counts.append(step_count)

            if idx < 5:
                logger.info(f"Sample {idx} GT: {ground_truth} | Pred: {predicted} | Correct: {is_correct}")
                logger.debug(f"Sample {idx} Raw Response: {response[:300]}")

            records.append({
                "idx": idx,
                "question": question[:100],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "correct": is_correct,
                "format_ok": is_format_ok,
                "response_snippet": response[:300],
            })

            if (idx + 1) % 10 == 0:
                running_acc = correct_count / (idx + 1)
                logger.info(f"  Progress: {idx+1}/{len(samples)} | Running accuracy: {running_acc:.1%}")

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

        logger.info(f"[{checkpoint_label}] {dataset_name}: Accuracy={accuracy:.1%}, Format={format_compliance:.1%}, Steps={avg_steps:.1f}")
        return {"summary": summary, "records": records}


def main():
    parser = argparse.ArgumentParser(description="CRAFT Phase 3 Evaluator")
    parser.add_argument("--model-type", default="hf", choices=["hf", "gguf"])
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="all", choices=["gsm8k", "strategyqa", "mmlu", "all"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--label", default="model", help="Label for this checkpoint in results")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--output-json", default="craft_output/results.json")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    evaluator = BenchmarkEvaluator(use_gpu=args.gpu)

    datasets = ["gsm8k", "strategyqa", "mmlu"] if args.dataset == "all" else [args.dataset]
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