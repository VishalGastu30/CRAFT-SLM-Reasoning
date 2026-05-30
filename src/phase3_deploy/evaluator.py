import os
import sys
import json
import argparse
import numpy as np
from loguru import logger

from src.phase2_rl.component_a.reward_scorer import RewardScorer
from src.phase0_probe.sampler import extract_final_answer, check_answer_correct

class BenchmarkEvaluator:
    """
    Evaluates SLM reasoning models (Base, SFT, and CRAFT RL) 
    on mathematical (GSM8K) and logical (StrategyQA) benchmarks.
    """
    def __init__(self, use_gpu=False):
        self.scorer = RewardScorer()
        self.use_gpu = use_gpu

    def load_test_dataset(self, dataset_name="gsm8k", num_samples=50) -> list:
        """Loads evaluation dataset with failproof fallbacks for offline testing."""
        logger.info(f"Loading test samples for dataset: {dataset_name} (samples={num_samples})")
        
        try:
            from datasets import load_dataset
            if dataset_name == "gsm8k":
                ds = load_dataset("gsm8k", "main", split="test")
                samples = []
                for item in list(ds)[:num_samples]:
                    samples.append({
                        "question": item["question"],
                        "answer": item["answer"].split("####")[-1].strip(),
                        "dataset": "gsm8k"
                    })
                return samples
            elif dataset_name == "strategyqa":
                # For StrategyQA, load from hub or fallback
                ds = load_dataset("wiz-a/strategyqa", split="train") # wiz-a strategyqa commonly available
                samples = []
                for item in list(ds)[:num_samples]:
                    samples.append({
                        "question": item["question"],
                        "answer": "yes" if item["answer"] else "no",
                        "dataset": "strategyqa"
                    })
                return samples
        except Exception as e:
            logger.warning(f"Could not load dataset {dataset_name} via datasets API: {e}. Falling back to high-quality local mock data!")
            
        # Fallback high quality mocks
        if dataset_name == "gsm8k":
            return [
                {"question": "John has 3 books. He buys 2 more. How many books does he have?", "answer": "5", "dataset": "gsm8k"},
                {"question": "What is 15% of 240?", "answer": "36", "dataset": "gsm8k"},
                {"question": "If a shirt costs $20 and is on a 20% discount, what is the final price?", "answer": "16", "dataset": "gsm8k"},
                {"question": "A train travels at 60 mph for 3 hours. How far does it go?", "answer": "180", "dataset": "gsm8k"},
                {"question": "Evaluate 10 + 5 * 2.", "answer": "20", "dataset": "gsm8k"}
            ] * (num_samples // 5 + 1)
        else:
            return [
                {"question": "Would a penguin survive in the Sahara desert?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Can a human walk on water without aid?", "answer": "no", "dataset": "strategyqa"},
                {"question": "Is Beijing the capital of China?", "answer": "yes", "dataset": "strategyqa"},
                {"question": "Do dogs bark?", "answer": "yes", "dataset": "strategyqa"},
                {"question": "Is the moon made of green cheese?", "answer": "no", "dataset": "strategyqa"}
            ] * (num_samples // 5 + 1)

    def generate_hf_response(self, model, tokenizer, prompt: str) -> str:
        """Generates response using PyTorch Hugging Face model."""
        import torch
        device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False, # Greedier decoding for evaluation consistency
                pad_token_id=tokenizer.eos_token_id
            )
        return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_gguf_response(self, llm, prompt: str) -> str:
        """Generates response using local GGUF llama.cpp model."""
        result = llm(prompt, max_tokens=256, temperature=0.0) # Greedy
        return result["choices"][0]["text"].strip()

    def evaluate_model(self, model_type="hf", model_path=None, dataset_name="gsm8k", num_samples=20) -> dict:
        """
        Runs evaluation on the specified model and dataset.
        model_type: 'hf' (HuggingFace Transformers model) or 'gguf' (llama-cpp-python GGUF path) or 'mock'
        """
        samples = self.load_test_dataset(dataset_name, num_samples)[:num_samples]
        
        # Load Model
        model = None
        tokenizer = None
        llm = None
        
        if model_type == "hf":
            logger.info(f"Loading HF model/tokenizer from: {model_path}")
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            import json
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
            
            # Check if this is a PEFT adapter by looking for adapter_config.json
            is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
            
            if is_peft:
                logger.info("Detected PEFT adapter. Loading base model first...")
                from peft import PeftModel
                with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
                    config = json.load(f)
                base_model_id = config.get("base_model_name_or_path", "microsoft/Phi-3-mini-4k-instruct")
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=False
                )
                model = PeftModel.from_pretrained(base_model, model_path).to(device)
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=False
                ).to(device)
                
            model.eval()
        elif model_type == "gguf":
            logger.info(f"Loading GGUF model from: {model_path}")
            from llama_cpp import Llama
            llm = Llama(model_path=model_path, verbose=False)
        else:
            logger.info("Running evaluation in mock mode.")
            
        correct_count = 0
        format_compliant_count = 0
        total_steps = []
        scores = []
        
        results_records = []
        
        for idx, sample in enumerate(samples):
            question = sample["question"]
            gold = sample["answer"]
            
            prompt = f"<|user|>\n{question}\n<|assistant|>\n"
            
            # Generate response
            if model_type == "hf":
                response = self.generate_hf_response(model, tokenizer, prompt)
            elif model_type == "gguf":
                response = self.generate_gguf_response(llm, prompt)
            else:
                # Mock response generator
                is_correct = (idx % 2 == 0)
                ans = gold if is_correct else f"wrong_{gold}"
                response = f"Step 1: Parse the parameters.\nStep 2: Calculate the solution.\nFinal Answer: {ans}"
                
            # Verify and score using RewardScorer
            reward, is_correct = self.scorer.score_with_success({"answer": gold}, response)
            metrics = {"reward": reward, "final_correct": float(is_correct)}
            
            # Check format compliance (must contain Step 1: and Final Answer: or similar)
            is_format_compliant = "step 1" in response.lower() and "answer" in response.lower()
            if is_format_compliant:
                format_compliant_count += 1
                
            correct = metrics["final_correct"]
            if correct > 0.5:
                correct_count += 1
                
            scores.append(metrics["reward"])
            step_count = response.lower().count("step")
            total_steps.append(step_count)
            
            results_records.append({
                "question": question,
                "gold_answer": gold,
                "model_response": response,
                "predicted_answer": extract_final_answer(response),
                "correct": bool(correct > 0.5),
                "reward": metrics["reward"],
                "format_compliant": is_format_compliant
            })
            
        accuracy = correct_count / len(samples)
        format_compliance = format_compliant_count / len(samples)
        avg_steps = np.mean(total_steps) if total_steps else 0
        avg_reward = np.mean(scores) if scores else 0
        
        summary = {
            "dataset": dataset_name,
            "accuracy": accuracy,
            "format_compliance": format_compliance,
            "avg_steps": float(avg_steps),
            "avg_reward": float(avg_reward),
            "sample_count": len(samples)
        }
        
        logger.info(f"Evaluation finished for {dataset_name}. Accuracy: {accuracy:.2%}, Format Compliance: {format_compliance:.2%}")
        
        return {
            "summary": summary,
            "records": results_records
        }

def main():
    parser = argparse.ArgumentParser(description="Phase 3: CRAFT Model Evaluator")
    parser.add_argument("--model-type", type=str, default="mock", choices=["hf", "gguf", "mock"], help="Type of model backend to load")
    parser.add_argument("--model-path", type=str, default=None, help="Path/name of HF or GGUF model")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "strategyqa", "all"], help="Dataset to evaluate on")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples to evaluate")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU for HF model evaluation")
    parser.add_argument("--output-json", type=str, default="craft_output/evaluation_results.json", help="Path to save evaluation summary json")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    
    evaluator = BenchmarkEvaluator(use_gpu=args.gpu)
    
    datasets_to_run = ["gsm8k", "strategyqa"] if args.dataset == "all" else [args.dataset]
    
    overall_results = {}
    
    for ds in datasets_to_run:
        res = evaluator.evaluate_model(
            model_type=args.model_type,
            model_path=args.model_path,
            dataset_name=ds,
            num_samples=args.samples
        )
        overall_results[ds] = res
        
    # Write summary
    with open(args.output_json, "w") as f:
        json.dump(overall_results, f, indent=4)
        
    logger.info(f"Evaluation results successfully saved to: {args.output_json}")

if __name__ == "__main__":
    main()
