from loguru import logger
from .component_a.reward_scorer import RewardScorer


class RewardCombiner:
    """
    Combines rewards from Component A and format penalties.
    Formula: R_total = max(0.0, R_A - format_penalty)
    """

    def __init__(self):
        self.scorer = RewardScorer()

    def calculate_format_penalty(self, response_text: str) -> float:
        penalty = 0.0
        response_lower = response_text.lower()

        has_reasoning = (
            "<thought>" in response_lower
            or "</thought>" in response_lower
            or "step 1" in response_lower
            or "step1" in response_lower
            or "step 2" in response_lower
        )
        if not has_reasoning:
            penalty += 0.2

        has_answer = (
            "<answer>" in response_lower
            or "</answer>" in response_lower
            or "final answer" in response_lower
            or "####" in response_lower
            or "answer:" in response_lower
        )
        if not has_answer:
            penalty += 0.2

        return penalty

    def combine_rewards(
        self,
        question: str,
        response_text: str,
        ground_truth: str,
        dataset_name: str = ""
    ) -> dict:
        question_data = {
            "question": question,
            "answer": ground_truth,
            "dataset": dataset_name,
        }

        r_a, is_correct = self.scorer.score_with_success(question_data, response_text)
        penalty = self.calculate_format_penalty(response_text)
        r_total = max(0.0, r_a - penalty)

        from .component_a.reward_scorer import extract_steps
        from .component_a.execution_verifier import ExecutionVerifier
        
        steps = extract_steps(response_text)
        if steps:
            verifier = ExecutionVerifier()
            step_score = verifier.score_steps(steps)
        else:
            step_score = 0.5

        return {
            "reward_raw": r_a,
            "format_penalty": penalty,
            "reward_combined": r_total,
            "is_correct": is_correct,
            "step_score": step_score,
        }