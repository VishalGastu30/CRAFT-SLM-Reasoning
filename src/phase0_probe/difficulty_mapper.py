import json
import os
import random
from loguru import logger
from .sampler import extract_final_answer, check_answer_correct

class DifficultyMapper:
    """
    Measures the difficulty of questions based on model capability (pass@k rate)
    and handles saving/loading the difficulty map and filtering for the 'learning zone'.
    """
    def __init__(self):
        self.difficulty_map = {}

    def build_map(self, model, tokenizer, questions, n_samples=5, temperature=0.8):
        """
        Runs pass@k sampling on a list of questions and records difficulty.
        Each question must be a dict: {"id": str, "question": str, "answer": str, "dataset": str}
        """
        logger.info(f"Probing {len(questions)} questions with k={n_samples}...")
        self.difficulty_map = {}

        for i, q in enumerate(questions):
            q_id = q.get("id", f"q_{i}")
            question_text = q["question"]
            ground_truth = q["answer"]
            dataset_name = q.get("dataset", "unknown")

            if model is None or tokenizer is None:
                # Simulated / Mock run for testing offline
                correct_count = random.randint(0, n_samples)
            else:
                correct_count = 0
                prompt = f"<|user|>\n{question_text}\n<|assistant|>\n"
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

                for _ in range(n_samples):
                    try:
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=temperature,
                            pad_token_id=tokenizer.eos_token_id
                        )
                        response = tokenizer.decode(
                            outputs[0][inputs.input_ids.shape[1]:],
                            skip_special_tokens=True
                        )
                        pred_answer = extract_final_answer(response)
                        if check_answer_correct(pred_answer, ground_truth):
                            correct_count += 1
                    except Exception as e:
                        logger.error(f"Error generating sample for question {q_id}: {e}")

            # Capability score is correct_count / n_samples
            capability_score = correct_count / n_samples
            # Difficulty is 1.0 - capability_score
            difficulty = round(1.0 - capability_score, 2)

            self.difficulty_map[q_id] = {
                "question": question_text,
                "answer": ground_truth,
                "dataset": dataset_name,
                "capability": capability_score,
                "difficulty": difficulty
            }

            if (i + 1) % 10 == 0:
                logger.info(f"Probed {i + 1}/{len(questions)} questions...")

        logger.info("Capability probing completed.")
        return self.difficulty_map

    def save_map(self, path="difficulty_map.json"):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.difficulty_map, f, indent=2)
        logger.info(f"Saved difficulty map ({len(self.difficulty_map)} items) to {path}")

    def load_map(self, path="difficulty_map.json"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Difficulty map not found at {path}")
        with open(path, "r") as f:
            self.difficulty_map = json.load(f)
        logger.info(f"Loaded difficulty map ({len(self.difficulty_map)} items) from {path}")
        return self.difficulty_map

    def get_learning_zone(self, low=0.4, high=0.7):
        learning_zone_questions = []
        for q_id, info in self.difficulty_map.items():
            diff = info.get("difficulty", 0.5)
            if low <= diff <= high:
                learning_zone_questions.append({
                    "id": q_id,
                    **info
                })
        logger.info(f"Found {len(learning_zone_questions)} questions in learning zone [{low}, {high}]")
        return learning_zone_questions