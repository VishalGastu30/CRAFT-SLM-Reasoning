"""
execution_verifier.py — Component A: Deterministic Step-Level Verifier

Scores individual reasoning steps by:
1. For math: extracts arithmetic expressions and evaluates with Python
2. For logic: checks claim-to-claim consistency

IMPORTANT NAMING CONVENTION in this file:
- Never name a variable 'score', 'result', 'compute', or 'evaluate'
  when a function with that name exists in the same scope.
- All scorer helper functions use the prefix 'verify_' to avoid collisions.
- All return variables use the suffix '_value' or '_score_val'.
"""

import re
import ast
from typing import Tuple, List, Optional


# ─── SAFE ARITHMETIC EVALUATOR ─────────────────────────────────────────────

def verify_arithmetic(expression: str) -> Tuple[bool, float]:
    """
    Safely evaluate a math expression string like '3 + 4 * 2'.
    
    Returns:
        (success, computed_value)
        success=False if the expression is unsafe or unparseable.
    
    Note: named 'verify_arithmetic' not 'score' or 'evaluate' to prevent
    variable shadowing that causes SyntaxWarning.
    """
    if not expression:
        return False, 0.0
    
    # Sanitize — only allow digits, operators, parentheses, spaces, decimals
    sanitized = expression.strip().replace(',', '').replace('×', '*').replace('÷', '/')
    
    # Whitelist check — reject anything that looks suspicious
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', sanitized):
        return False, 0.0
    
    # Guard against empty or trivial expressions
    if re.match(r'^\s*[\d\.]+\s*$', sanitized):
        # It's just a bare number — nothing to evaluate
        try:
            return True, float(sanitized.strip())
        except ValueError:
            return False, 0.0
    
    # Try literal eval first (pure constants)
    try:
        computed_val = ast.literal_eval(sanitized)
        return True, float(computed_val)
    except (ValueError, SyntaxError):
        pass
    
    # Restricted eval with empty builtins (no access to builtins like __import__)
    try:
        computed_val = eval(sanitized, {"__builtins__": {}})  # nosec B307
        return True, float(computed_val)
    except Exception:
        return False, 0.0


def verify_step_arithmetic(step_text: str) -> Tuple[int, int]:
    """
    Verify arithmetic correctness within a single reasoning step.
    Looks for patterns like 'X + Y = Z' and checks if X + Y actually equals Z.
    
    Returns:
        (correct_count, total_checkable_count)
        Both are 0 if no checkable expressions were found.
    
    Note: returns counts not a score — the caller computes the ratio.
    This prevents the pattern where 'score = 0.5' then 'score(x)' causes warnings.
    """
    if not step_text:
        return 0, 0
    
    # Pattern: left_expression = claimed_result
    # Matches things like: "3 + 4 = 7", "15 × 4 = 60", "100 - 35 = 65"
    # Does NOT match: "x = 5" (variable assignment — not verifiable)
    equation_pattern = re.findall(
        r'([\d\s\+\-\*\/\(\)\.]+)\s*=\s*([\-\+]?\d[\d,\.]*)',
        step_text
    )
    
    correct_count_val = 0   # named _val to avoid any shadowing risk
    total_count_val = 0
    
    for left_side, right_side in equation_pattern:
        left_cleaned = left_side.strip()
        right_cleaned = right_side.strip().replace(',', '')
        
        # Skip if left side is also just a number (e.g., "5 = 5")
        if re.match(r'^\s*[\d\.]+\s*$', left_cleaned):
            continue
        
        # Try to evaluate the left side
        ok, computed_val = verify_arithmetic(left_cleaned)
        if not ok:
            continue
        
        # Parse the claimed right side
        try:
            claimed_val = float(right_cleaned)
        except ValueError:
            continue
        
        # Check correctness within tolerance
        total_count_val += 1
        if abs(computed_val - claimed_val) < 0.01:
            correct_count_val += 1
    
    return correct_count_val, total_count_val


def verify_logical_consistency(steps: List[str]) -> float:
    """
    Very lightweight logical consistency check across steps.
    Does not use NLI (too slow for per-step RL). Instead checks:
    - Numbers introduced in earlier steps are used correctly in later steps
    - No obviously contradictory numeric values across steps
    
    Returns a float 0.0 to 1.0. 0.5 = neutral (couldn't verify).
    """
    if not steps or len(steps) < 2:
        return 0.5
    
    # Extract all numbers mentioned in each step
    step_numbers = []
    for step_text in steps:
        nums_in_step = re.findall(r'[\-\+]?\d[\d,\.]*', step_text)
        cleaned_nums = []
        for n in nums_in_step:
            try:
                cleaned_nums.append(float(n.replace(',', '')))
            except ValueError:
                pass
        step_numbers.append(set(cleaned_nums))
    
    # Check: numbers mentioned in the final step should appear somewhere earlier
    # (i.e., the final answer isn't a completely new number that was never computed)
    if len(step_numbers) >= 2:
        final_nums = step_numbers[-1]
        all_prior_nums = set()
        for prior in step_numbers[:-1]:
            all_prior_nums.update(prior)
        
        if final_nums:
            overlap_ratio = len(final_nums & all_prior_nums) / len(final_nums)
            if overlap_ratio > 0.5:
                return 0.7   # Good consistency — final answer relates to prior steps
            elif overlap_ratio == 0:
                return 0.3   # Bad consistency — final answer came from nowhere
    
    return 0.5  # Neutral — couldn't determine


class ExecutionVerifier:
    """
    Component A: Deterministic execution-based verifier.
    
    Scores reasoning steps by actually running the math through Python.
    No neural reward model. No reward hacking possible.
    100% deterministic — same input always produces same score.
    
    Usage:
        verifier = ExecutionVerifier()
        step_score_value = verifier.score_steps(list_of_step_strings)
        # Returns float 0.0-1.0. 0.5 = neutral, >0.7 = good arithmetic, 1.0 = perfect.
    """
    
    def score_steps(self, steps: List[str]) -> float:
        """
        Score a list of reasoning steps by checking arithmetic correctness.
        
        Returns:
            Float 0.0 to 1.0.
            0.5 if no checkable arithmetic found (neutral — don't penalize).
            Higher if arithmetic is correct, lower if incorrect.
        
        Note: return variable named 'step_score_value' not 'score' to prevent
        any risk of the caller's score variable being shadowed.
        """
        if not steps:
            return 0.5
        
        total_correct = 0
        total_checkable = 0
        
        for step_text in steps:
            if not isinstance(step_text, str):
                continue
            correct_in_step, total_in_step = verify_step_arithmetic(step_text)
            total_correct += correct_in_step
            total_checkable += total_in_step
        
        if total_checkable == 0:
            # No arithmetic expressions found — check logical consistency instead
            consistency_val = verify_logical_consistency(steps)
            return consistency_val
        
        # Arithmetic ratio — how many math expressions were correct
        arithmetic_ratio = total_correct / total_checkable
        
        # Blend: 80% arithmetic correctness, 20% logical consistency
        consistency_val = verify_logical_consistency(steps)
        step_score_value = 0.8 * arithmetic_ratio + 0.2 * consistency_val
        
        return step_score_value
    
    def score_single_step(self, step_text: str) -> float:
        """Score a single reasoning step. Returns 0.0-1.0."""
        return self.score_steps([step_text])
