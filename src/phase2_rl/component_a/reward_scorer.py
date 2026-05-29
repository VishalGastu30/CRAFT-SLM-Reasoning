"""
reward_scorer.py — Component A: Deterministic Execution Verifier (v2)
Fixed: handles all GSM8K ground truth formats, robust answer extraction,
       no variable name shadowing that causes SyntaxWarning.
"""

import re
import ast
from typing import Tuple


# ─── GROUND TRUTH EXTRACTION ─────────────────────────────────────────────────

def extract_ground_truth(question_data: dict) -> str:
    """
    Extract the numeric/boolean answer from a dataset record.
    
    Handles:
    - GSM8K: "Some text\n#### 9"  →  "9"
    - StrategyQA: True/False/yes/no  →  "yes"/"no"
    - AQuA-RAT: "correct: (A)" or direct letter  →  letter
    - Direct numeric string  →  that number
    """
    raw = question_data.get("answer", "")
    raw_str = str(raw).strip()
    
    # GSM8K format — "#### 42" at end of answer field
    gsm_match = re.search(r'####\s*([\-\+]?\d[\d,]*(?:\.\d+)?)', raw_str)
    if gsm_match:
        return gsm_match.group(1).replace(',', '').strip()
    
    # StrategyQA — boolean True/False (Python bool or string)
    if raw_str.lower() in ("true", "yes", "1"):
        return "yes"
    if raw_str.lower() in ("false", "no", "0"):
        return "no"
    
    # AQuA-RAT — multiple choice letter (A/B/C/D/E)
    aqua_match = re.search(r'\b([A-E])\b', raw_str)
    if aqua_match:
        return aqua_match.group(1).upper()
    
    # Direct numeric (possibly with commas)
    num_match = re.search(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', raw_str)
    if num_match:
        return num_match.group(1).replace(',', '').strip()
    
    return raw_str


# ─── MODEL ANSWER EXTRACTION ─────────────────────────────────────────────────

def extract_model_answer(response_text: str) -> str:
    """
    Extract model's final answer from generated text.
    Tries multiple formats in priority order.
    """
    if not response_text:
        return ""
    
    cleaned = response_text.strip()
    
    # Priority 1: <answer>X</answer> — the format we trained for
    xml_match = re.search(
        r'<answer>\s*([\s\S]*?)\s*</answer>',
        cleaned, re.IGNORECASE
    )
    if xml_match:
        content = xml_match.group(1).strip()
        # Handle "<answer>The answer is 42</answer>"
        num = re.search(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', content)
        if num:
            return num.group(1).replace(',', '')
        yesno = re.search(r'\b(yes|no)\b', content, re.IGNORECASE)
        if yesno:
            return yesno.group(1).lower()
        letter = re.search(r'\b([A-E])\b', content)
        if letter:
            return letter.group(1).upper()
        return content
    
    # Priority 2: "Final Answer: X" or "The answer is X" or "Answer: X"
    for patt in [
        r'(?:final\s+answer|the\s+answer\s+is|answer\s*is|answer\s*:)\s*[:\-]?\s*([\-\+]?\d[\d,]*(?:\.\d+)?)',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*\*{0,2}([\-\+]?\d[\d,]*(?:\.\d+)?)\*{0,2}',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*(yes|no)\b',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*([A-E])\b',
    ]:
        m = re.search(patt, cleaned, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(',', '')
    
    # Priority 3: "= 42" at end of a line (common in step-by-step)
    eq_end = re.search(r'=\s*([\-\+]?\d[\d,]*(?:\.\d+)?)\s*$', cleaned, re.MULTILINE)
    if eq_end:
        return eq_end.group(1).replace(',', '')
    
    # Priority 4: Last number that appears AFTER any </thought> block
    after_thought = re.sub(r'<thought>[\s\S]*?</thought>', '', cleaned, flags=re.IGNORECASE)
    numbers = re.findall(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', after_thought)
    if numbers:
        return numbers[-1].replace(',', '')
    
    # Priority 5: Any yes/no after thought block
    yesno = re.search(r'\b(yes|no)\b', after_thought, re.IGNORECASE)
    if yesno:
        return yesno.group(1).lower()
    
    # Priority 6: Last number anywhere in response (last resort)
    all_nums = re.findall(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', cleaned)
    if all_nums:
        return all_nums[-1].replace(',', '')
    
    return ""


# ─── ANSWER COMPARISON ───────────────────────────────────────────────────────

def answers_match(model_ans: str, ground_truth: str) -> bool:
    """
    Compare model answer to ground truth.
    Handles numeric tolerance, boolean synonyms, letter choices.
    
    Note: parameter named model_ans not 'score' to avoid shadowing function names.
    """
    if not model_ans or not ground_truth:
        return False
    
    ma = model_ans.strip().lower().replace(',', '')
    gt = ground_truth.strip().lower().replace(',', '')
    
    # Exact string match
    if ma == gt:
        return True
    
    # Numeric comparison with tolerance
    try:
        ma_num = float(ma)
        gt_num = float(gt)
        # Use relative tolerance for large numbers, absolute for small
        if abs(gt_num) > 1.0:
            return abs(ma_num - gt_num) / (abs(gt_num) + 1e-9) < 0.01
        return abs(ma_num - gt_num) < 0.01
    except (ValueError, TypeError):
        pass
    
    # Boolean synonym groups
    yes_group = {"yes", "true", "correct", "1", "right"}
    no_group = {"no", "false", "incorrect", "0", "wrong"}
    if ma in yes_group and gt in yes_group:
        return True
    if ma in no_group and gt in no_group:
        return True
    
    return False


# ─── STEP SCORING ────────────────────────────────────────────────────────────

def safe_eval_math(expr: str) -> Tuple[bool, float]:
    """
    Safely evaluate a math expression like '3 + 4 * 2'.
    Only allows digits and arithmetic operators.
    Returns (success, result).
    
    Note: function named safe_eval_math not 'score' to avoid name collisions.
    """
    # Whitelist check — only allow safe chars
    sanitized = expr.strip().replace(',', '')
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', sanitized):
        return False, 0.0
    
    # Try literal eval first (handles pure numbers)
    try:
        result = ast.literal_eval(sanitized)
        return True, float(result)
    except (ValueError, SyntaxError):
        pass
    
    # Restricted eval for arithmetic
    try:
        result = eval(sanitized, {"__builtins__": {}})  # nosec
        return True, float(result)
    except Exception:
        return False, 0.0


def score_reasoning_steps(steps: list) -> float:
    """
    Score reasoning steps by checking arithmetic correctness.
    Looks for patterns like '3 + 4 = 7' and verifies the equality.
    Returns 0.0 to 1.0.
    
    Note: function named score_reasoning_steps (not 'score') to avoid
    variable shadowing that causes SyntaxWarning.
    """
    if not steps:
        return 0.5  # Neutral: can't verify what isn't there
    
    total_checkable = 0
    total_correct = 0
    
    for step_text in steps:
        # Find "expression = claimed_result" patterns
        # e.g. "3 + 4 = 7" or "15 × 4 = 60"
        matches = re.findall(
            r'([\d\s\+\-\*\/\(\)\.]+)\s*=\s*([\d\.\-\+,]+)',
            step_text
        )
        
        for left_side, right_side in matches:
            left_clean = left_side.strip()
            right_clean = right_side.strip().replace(',', '')
            
            # Skip trivial: if left side is just a single number, skip
            if re.match(r'^\s*[\d\.]+\s*$', left_clean):
                continue
            
            ok, computed_val = safe_eval_math(left_clean)
            if not ok:
                continue
            
            try:
                claimed_val = float(right_clean)
            except ValueError:
                continue
            
            total_checkable += 1
            if abs(computed_val - claimed_val) < 0.01:
                total_correct += 1
    
    if total_checkable == 0:
        return 0.5  # No checkable steps — neutral score
    
    return total_correct / total_checkable


# ─── STEP EXTRACTION ─────────────────────────────────────────────────────────

def extract_steps(response_text: str) -> list:
    """
    Extract individual reasoning steps from model output.
    Handles <thought> blocks and "Step N:" format and numbered lists.
    """
    if not response_text:
        return []
    
    # Work inside <thought> tags if present
    thought_match = re.search(
        r'<thought>([\s\S]*?)</thought>',
        response_text, re.IGNORECASE
    )
    working = thought_match.group(1).strip() if thought_match else response_text
    
    # Remove <answer> section from working text
    working = re.sub(r'<answer>[\s\S]*?</answer>', '', working, flags=re.IGNORECASE).strip()
    
    # Try "Step N:" splits
    splits = re.split(r'\bStep\s+\d+\s*[:\-]\s*', working)
    splits = [s.strip() for s in splits if len(s.strip()) > 5]
    if len(splits) >= 2:
        return splits
    
    # Try numbered list "1. " "2. "
    splits = re.split(r'(?m)^\s*\d+\.\s+', working)
    splits = [s.strip() for s in splits if len(s.strip()) > 5]
    if len(splits) >= 2:
        return splits
    
    # Fallback: line by line
    lines = [l.strip() for l in working.split('\n') if len(l.strip()) > 5]
    return lines if lines else [working]


# ─── MAIN SCORER CLASS ───────────────────────────────────────────────────────

class RewardScorer:
    """
    Component A: Scores model responses using deterministic verification.
    Formula: R_A = 0.4 × final_correct + 0.6 × step_score
    """

    def score(self, question_data: dict, response_text: str) -> float:
        """Return just the reward float."""
        reward_val, _ = self.score_with_success(question_data, response_text)
        return reward_val

    def score_with_success(
        self, question_data: dict, response_text: str
    ) -> Tuple[float, bool]:
        """
        Returns (reward_score, is_final_answer_correct).
        
        reward_score: 0.0 to 1.0
        is_final_answer_correct: True only if model's answer matches ground truth
        """
        if not response_text or len(response_text.strip()) < 3:
            return 0.1, False
        
        # Extract answers
        ground_truth = extract_ground_truth(question_data)
        model_ans = extract_model_answer(response_text)
        
        # Check correctness
        is_correct = answers_match(model_ans, ground_truth)
        
        # Score steps
        steps = extract_steps(response_text)
        step_reward = score_reasoning_steps(steps)
        
        # Combined reward
        R_A = 0.4 * float(is_correct) + 0.6 * step_reward
        
        # Small bonus for correct format
        if '<thought>' in response_text.lower() and '<answer>' in response_text.lower():
            R_A = min(1.0, R_A + 0.05)
        
        return R_A, is_correct
