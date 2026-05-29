from loguru import logger
from .component_a.reward_scorer import RewardScorer

class RewardCombiner:
    """
    Combines rewards from:
    1. Component A: Math Execution & logical NLI verifiers.
    2. Format consistency penalties (checks step formatting).
    Formula: R_total = R_A - format_penalty
    """
    def __init__(self, nli_model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.scorer = RewardScorer(nli_model_name=nli_model_name, device=device)

    def calculate_format_penalty(self, response_text: str) -> float:
        """
        Penalizes outputs that fail to follow the expected formatting rules.
        Accepts either XML-tag format (<thought>/<answer>) OR text format (Step 1: / Final Answer:).
        Only penalizes if NEITHER format is present.
        """
        penalty = 0.0
        response_lower = response_text.lower()

        # Check for reasoning structure: XML tags OR step keywords
        has_reasoning = (
            "<thought>" in response_lower
            or "</thought>" in response_lower
            or "step 1" in response_lower
            or "step1" in response_lower
        )
        if not has_reasoning:
            penalty += 0.2

        # Check for final answer: XML tags OR text delimiter
        has_answer = (
            "<answer>" in response_lower
            or "</answer>" in response_lower
            or "final answer" in response_lower
            or "####" in response_lower
        )
        if not has_answer:
            penalty += 0.2

        return penalty

    def combine_rewards(self, question: str, response_text: str, ground_truth: str, dataset_name: str = "") -> dict:
        """
        Computes the final total reward combining step verifications and format checking.
        Returns a dictionary of raw metrics and the final reward value.
        """
        # Get Component A scores (Math or NLI)
        metrics = self.scorer.score_response(
            question=question,
            response_text=response_text,
            ground_truth=ground_truth,
            dataset_name=dataset_name
        )
        
        r_a = metrics.get("reward", 0.0)
        
        # Calculate format penalty
        penalty = self.calculate_format_penalty(response_text)
        
        # Combined reward, bounded between 0.0 and 1.0
        r_total = max(0.0, r_a - penalty)
        
        metrics["format_penalty"] = penalty
        metrics["reward_combined"] = r_total
        
        return metrics
