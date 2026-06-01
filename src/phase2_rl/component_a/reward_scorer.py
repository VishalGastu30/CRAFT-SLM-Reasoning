"""
reward_scorer.py — Component A: Main Reward Scorer

Formula: R_A = 0.4 × final_correct + 0.6 × step_score_value
+ 0.05 bonus if model used correct <thought>/<answer> format.

NAMING CONVENTION:
- The class method is named 'compute_score()' internally where needed.
- Local variables that hold score values use suffix '_value' or '_val'.
- Never assign a float to a variable named 'score' in this file,
  because the class has a method named 'score()'.
"""

import re
from typing import Tuple

from src.phase2_rl.component_a.execution_verifier import ExecutionVerifier


# ─── GROUND TRUTH EXTRACTION ────────────────────────────────────────────────

def extract_ground_truth(question_data: dict) -> str:
    """
    Extract the expected answer from a dataset record.
    
    Handles all dataset formats this project uses:
    - GSM8K:     "Some text\n#### 9"              →  "9"
    - AQuA-RAT:  "correct: (A)" or "A"            →  "A"
    - StrategyQA: True/False (Python bool/string) →  "yes"/"no"
    - MMLU:      "0", "1", "2", "3"              →  "A", "B", "C", "D"
    - Direct number: "42"                         →  "42"
    """
    raw = question_data.get("answer", "")
    raw_str = str(raw).strip()
    
    if not raw_str:
        return ""
    
    # GSM8K format: "#### 42" at end of answer field
    gsm_match = re.search(r'####\s*([\-\+]?\d[\d,]*(?:\.\d+)?)', raw_str)
    if gsm_match:
        return gsm_match.group(1).replace(',', '').strip()
    
    # StrategyQA / boolean
    lower_raw = raw_str.lower()
    if lower_raw in ("true", "yes", "1"):
        return "yes"
    if lower_raw in ("false", "no", "0"):
        return "no"
    
    # MMLU numeric index (0→A, 1→B, 2→C, 3→D)
    if raw_str in ("0", "1", "2", "3"):
        return ["A", "B", "C", "D"][int(raw_str)]
    
    # AQuA-RAT letter choice
    letter_match = re.search(r'\b([A-E])\b', raw_str)
    if letter_match:
        return letter_match.group(1).upper()
    
    # Direct numeric (possibly with commas or decimals)
    num_match = re.search(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', raw_str)
    if num_match:
        return num_match.group(1).replace(',', '').strip()
    
    return raw_str


# ─── MODEL ANSWER EXTRACTION ────────────────────────────────────────────────

def extract_model_answer(response_text: str) -> str:
    """
    Extract the model's final answer from its generated text.
    
    Tries formats in priority order — highest confidence first.
    Always returns a string (empty string if nothing found).
    """
    if not response_text:
        return ""
    
    cleaned = response_text.strip()
    
    # ── Priority 1: <answer>X</answer> ──
    # This is the format we trained for in SFT. Most reliable.
    xml_match = re.search(
        r'<answer>\s*([\s\S]*?)\s*</answer>',
        cleaned, re.IGNORECASE
    )
    if xml_match:
        inner = xml_match.group(1).strip()
        
        # Inner might be "The answer is 42" — extract the number
        num_inner = re.search(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', inner)
        if num_inner:
            return num_inner.group(1).replace(',', '')
        
        # Inner might be "yes" or "no"
        yesno_inner = re.search(r'\b(yes|no)\b', inner, re.IGNORECASE)
        if yesno_inner:
            return yesno_inner.group(1).lower()
        
        # Inner might be a letter (AQuA-RAT style)
        letter_inner = re.search(r'\b([A-E])\b', inner)
        if letter_inner:
            return letter_inner.group(1).upper()
        
        # Return raw inner content as last resort
        return inner
    
    # ── Priority 2: "Final Answer: X" patterns ──
    final_answer_patterns = [
        r'(?:final\s+answer|the\s+answer\s+is|answer\s*:)\s*[:\-]?\s*([\-\+]?\d[\d,]*(?:\.\d+)?)',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*\*{0,2}([\-\+]?\d[\d,]*(?:\.\d+)?)\*{0,2}',
        r'(?:therefore|thus|so)[,\s]+(?:the\s+answer\s+is\s+)?([\-\+]?\d[\d,]*(?:\.\d+)?)',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*(yes|no)\b',
        r'(?:final\s+answer|answer)\s*[:\-]?\s*([A-E])\b',
    ]
    for patt in final_answer_patterns:
        match_found = re.search(patt, cleaned, re.IGNORECASE)
        if match_found:
            extracted_val = match_found.group(1).strip().replace(',', '')
            return extracted_val
    
    # ── Priority 3: GSM8K native #### format ──
    gsm_match = re.search(r'####\s*([\-\+]?\d[\d,]*(?:\.\d+)?)', cleaned)
    if gsm_match:
        return gsm_match.group(1).replace(',', '')
    
    # ── Priority 4: "= X" at end of a step line ──
    eq_end_match = re.search(
        r'=\s*([\-\+]?\d[\d,]*(?:\.\d+)?)\s*$',
        cleaned, re.MULTILINE
    )
    if eq_end_match:
        return eq_end_match.group(1).replace(',', '')
    
    # ── Priority 5: Any number appearing AFTER </thought> ──
    after_thought = re.sub(
        r'<thought>[\s\S]*?</thought>', '', cleaned, flags=re.IGNORECASE
    )
    after_nums = re.findall(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', after_thought)
    if after_nums:
        return after_nums[-1].replace(',', '')
    
    # ── Priority 6: yes/no anywhere after thought ──
    yesno_after = re.search(r'\b(yes|no)\b', after_thought, re.IGNORECASE)
    if yesno_after:
        return yesno_after.group(1).lower()
    
    # ── Priority 7: Last number anywhere (absolute last resort) ──
    all_nums = re.findall(r'([\-\+]?\d[\d,]*(?:\.\d+)?)', cleaned)
    if all_nums:
        return all_nums[-1].replace(',', '')
    
    return ""


# ─── ANSWER COMPARISON ──────────────────────────────────────────────────────

def answers_match(model_answer: str, ground_truth: str) -> bool:
    """
    Compare extracted model answer to ground truth.
    
    Handles:
    - Numeric values with tolerance (3.999 ≈ 4.000)
    - Boolean synonyms (yes/true/correct all match each other)
    - Multiple-choice letters (A, B, C, D, E)
    - Exact string match as fallback
    
    Note: parameters named 'model_answer' and 'ground_truth' not 'score'
    to prevent any variable shadowing risk.
    """
    if not model_answer or not ground_truth:
        return False
    
    ma_clean = model_answer.strip().lower().replace(',', '')
    gt_clean = ground_truth.strip().lower().replace(',', '')
    
    # Exact match (handles letters like A, B, yes, no)
    if ma_clean == gt_clean:
        return True
    
    # Numeric comparison with tolerance
    try:
        ma_num = float(ma_clean)
        gt_num = float(gt_clean)
        # Relative tolerance for large numbers, absolute for small
        if abs(gt_num) > 1.0:
            return abs(ma_num - gt_num) / (abs(gt_num) + 1e-9) < 0.01
        return abs(ma_num - gt_num) < 0.01
    except (ValueError, TypeError):
        pass
    
    # Boolean synonym groups
    affirmative = {"yes", "true", "correct", "1", "right"}
    negative = {"no", "false", "incorrect", "0", "wrong"}
    if ma_clean in affirmative and gt_clean in affirmative:
        return True
    if ma_clean in negative and gt_clean in negative:
        return True
    
    return False


# ─── STEP EXTRACTION ────────────────────────────────────────────────────────

def extract_steps(response_text: str) -> list:
    """
    Extract individual reasoning steps from model output.
    
    Returns a list of step strings. Each string is one reasoning step.
    Returns at least [response_text] if no step structure is found.
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
    working = re.sub(
        r'<answer>[\s\S]*?</answer>', '', working, flags=re.IGNORECASE
    ).strip()
    
    # Try "Step N:" splits
    step_splits = re.split(r'\bStep\s+\d+\s*[:\-]\s*', working)
    step_splits = [s.strip() for s in step_splits if len(s.strip()) > 5]
    if len(step_splits) >= 2:
        return step_splits
    
    # Try numbered list "1. " "2. "
    num_splits = re.split(r'(?m)^\s*\d+\.\s+', working)
    num_splits = [s.strip() for s in num_splits if len(s.strip()) > 5]
    if len(num_splits) >= 2:
        return num_splits
    
    # Fallback: line by line
    lines = [line.strip() for line in working.split('\n') if len(line.strip()) > 5]
    return lines if lines else [working]


# ─── MAIN SCORER CLASS ──────────────────────────────────────────────────────

class RewardScorer:
    """
    Component A: Scores model responses using deterministic verification.
    
    Formula: R_A = 0.4 × final_correct + 0.6 × step_score_value
    Bonus:   +0.05 if model uses <thought> and <answer> tags correctly
    
    Usage:
        scorer = RewardScorer()
        reward_val, is_correct = scorer.score_with_success(question, response)
    """
    
    def __init__(self):
        self._verifier = ExecutionVerifier()
    
    def score(self, question_data: dict, response_text: str) -> float:
        """Return just the reward float. Calls score_with_success internally."""
        reward_val, _ = self.score_with_success(question_data, response_text)
        return reward_val
    # Note: local variable is 'reward_val' not 'score' — prevents shadowing.
    
    def score_with_success(
        self,
        question_data: dict,
        response_text: str
    ) -> Tuple[float, bool]:
        """
        Score a model response against a question.
        
        Returns:
            (reward_score, is_final_answer_correct)
            reward_score: float 0.0 to 1.0
            is_final_answer_correct: bool — used for mean_success logging
        """
        if not response_text or len(response_text.strip()) < 3:
            return 0.1, False
        
        # Extract ground truth and model answer
        ground_truth = extract_ground_truth(question_data)
        model_answer = extract_model_answer(response_text)
        
        # Check if final answer is correct
        is_correct = answers_match(model_answer, ground_truth)
        
        # Score reasoning steps
        steps = extract_steps(response_text)
        step_score_value = self._verifier.score_steps(steps)
        
        # Format detection
        has_thought = '<thought>' in response_text.lower()
        has_answer = '<answer>' in response_text.lower()
        
        # Gated Combined Reward
        if not is_correct:
            # Wrong answer: tiny reward only for maintaining some format
            # This gives gradient signal to keep exploring, but makes being wrong unprofitable
            format_signal = 0.05 if (has_thought and has_answer) else 0.0
            R_A = 0.05 + format_signal   # Max: 0.10 for wrong answer
        else:
            # Correct answer: now reward quality of reasoning
            R_A = 0.7 + 0.25 * step_score_value
            if has_thought and has_answer:
                R_A = min(1.0, R_A + 0.05)  # Max: 1.0 for correct + quality + format
        
        return R_A, is_correct
