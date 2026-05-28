from loguru import logger
from .execution_verifier import score_reasoning_chain as math_score_chain, extract_steps
from .nli_verifier import NLIVerifier
from src.phase0_probe.sampler import extract_final_answer, check_answer_correct

class RewardScorer:
    """
    Unified reward scorer for Phase 2 RL training.
    Routes queries to the Math Execution Verifier or the logical NLI Verifier
    depending on the question type.
    """
    def __init__(self, nli_model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.nli_verifier = NLIVerifier(model_name=nli_model_name, device=device)

    def is_math_problem(self, question: str, dataset_name: str = "") -> bool:
        """Determines if a question is mathematical or logical."""
        dataset_name = dataset_name.lower()
        if "gsm" in dataset_name or "aqua" in dataset_name or "math" in dataset_name:
            return True
            
        # Heuristic check for math expressions/symbols in question
        math_indicators = ["+", "-", "*", "/", "=", "calculate", "solve", "fraction", "percent", "how many", "sum"]
        q_lower = question.lower()
        
        # Check if contains numbers
        has_digits = any(char.isdigit() for char in question)
        
        # Match if both digits and math keywords are present
        if has_digits and any(ind in q_lower for ind in math_indicators):
            return True
            
        return False

    def score_response(self, question: str, response_text: str, ground_truth: str, dataset_name: str = "") -> dict:
        """
        Main routing function that returns a comprehensive metrics dict including
        overall reward (0.0 to 1.0) and components.
        """
        if self.is_math_problem(question, dataset_name):
            # Route to Math Execution Verifier
            metrics = math_score_chain(response_text, ground_truth)
            metrics["verifier_type"] = "math"
            return metrics
        else:
            # Route to NLI logical verifier
            steps = extract_steps(response_text)
            pred_final = extract_final_answer(response_text)
            
            final_correct = 1.0 if check_answer_correct(pred_final, ground_truth) else 0.0
            
            if not steps:
                return {
                    "final_correct": final_correct,
                    "mean_step_score": 0.0,
                    "reward": 0.4 * final_correct,
                    "step_count": 0,
                    "verifier_type": "nli"
                }
                
            mean_step_score = self.nli_verifier.score_logical_chain(steps)
            reward = 0.4 * final_correct + 0.6 * mean_step_score
            
            return {
                "final_correct": final_correct,
                "mean_step_score": mean_step_score,
                "reward": reward,
                "step_count": len(steps),
                "verifier_type": "nli"
            }
