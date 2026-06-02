"""
reward_combiner.py — Combines Component A reward with format penalties.

Formula: R_total = R_A - format_penalty (clamped to [0,1]).
"""

from loguru import logger
from .component_a.reward_scorer import RewardScorer


class RewardCombiner:
    """
    Combines rewards from:
    1. Component A: Deterministic execution verifier (math/logic).
    2. Format consistency penalties (step formatting, answer tags).

    Formula: R_total = max(0.0, R_A - format_penalty)
    """

    def __init__(self):
        """Initialize with the fixed RewardScorer."""
        self.scorer = RewardScorer()

    def calculate_format_penalty(self, response_text: str) -> float:
        """
        Penalizes outputs that fail to follow expected formatting rules.
        Accepts either XML-tag format (<thought>/<answer>) OR text format
        (Step N: / Final Answer: / ####). Penalty is applied if NEITHER
        format is detected.

        Returns:
            float penalty in [0.0, 0.4] (0.2 for missing reasoning, 0.2 for missing answer).
        """
        penalty = 0.0
        response_lower = response_text.lower()

        # Check for reasoning structure: XML tags OR step keywords
        has_reasoning = (
            "<thought>" in response_lower
            or "</thought>" in response_lower
            or "step 1" in response_lower
            or "step1" in response_lower
            or "step 2" in response_lower
        )
        if not has_reasoning:
            penalty += 0.2

        # Check for final answer: XML tags OR text delimiter
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
        """
        Compute the final total reward combining Component A score and format penalty.

        Args:
            question: The original question text.
            response_text: The model's generated output.
            ground_truth: The expected answer (string, may contain #### for GSM8K).
            dataset_name: "gsm8k", "strategyqa", "mmlu", etc.

        Returns:
            dict with keys:
                - reward_raw: Raw R_A from Component A.
                - format_penalty: Penalty applied.
                - reward_combined: R_total = max(0, R_A - penalty).
                - is_correct: Whether the final answer matched ground truth.
                - step_score: Reasoning step score (0.0-1.0).
        """
        # Build question_data dict expected by RewardScorer
        question_data = {
            "question": question,
            "answer": ground_truth,
            "dataset": dataset_name,
        }

        # Get Component A score and correctness flag
        r_a, is_correct = self.scorer.score_with_success(question_data, response_text)

        # Calculate format penalty
        penalty = self.calculate_format_penalty(response_text)

        # Combined reward, bounded between 0.0 and 1.0
        r_total = max(0.0, r_a - penalty)

        # Extract step score from response (approximate)
        from src.phase2_rl.component_a.reward_scorer import extract_steps
        from src.phase2_rl.component_a.execution_verifier import ExecutionVerifier
        
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