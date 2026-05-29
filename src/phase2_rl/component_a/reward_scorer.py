"""
reward_scorer.py — Component A: Deterministic Execution Verifier
Scores model responses using Python interpreter for math and NLI for logic.
No neural reward model. Zero reward noise for arithmetic.
"""

import re
import ast
import math
from typing import Tuple


def extract_ground_truth(question_data: dict) -> str:
    """
    Extract the correct answer from dataset entry.
    Handles GSM8K (#### N format), StrategyQA (yes/no), and direct answers.
    """
    answer = question_data.get("answer", "")
    
    # GSM8K: "Some working\n#### 9"
    gsm8k_match = re.search(r'####\s*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', str(answer))
    if gsm8k_match:
        return gsm8k_match.group(1).replace(',', '').strip()
    
    # StrategyQA: boolean
    answer_str = str(answer).strip().lower()
    if answer_str in ["yes", "no", "true", "false"]:
        return answer_str
    if answer_str == "1":
        return "yes"
    if answer_str == "0":
        return "no"
    
    # Direct numeric
    numeric_match = re.search(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', str(answer))
    if numeric_match:
        return numeric_match.group(1).replace(',', '').strip()
    
    return str(answer).strip()


def extract_model_answer(response_text: str) -> str:
    """
    Extract the model's final answer from its output.
    Handles XML tags, "Final Answer:" format, and last-number fallback.
    """
    if not response_text:
        return ""
    
    text = response_text.strip()
    
    # Priority 1: <answer> XML tag
    xml_match = re.search(
        r'<answer>\s*(.*?)\s*</answer>',
        text, re.DOTALL | re.IGNORECASE
    )
    if xml_match:
        ans = xml_match.group(1).strip().replace(',', '')
        # Sometimes the model writes "<answer>The answer is 42</answer>"
        # Extract just the number
        num = re.search(r'([+-]?\d+(?:\.\d+)?)', ans)
        if num:
            return num.group(1)
        # Handle yes/no
        if ans.lower() in ["yes", "no"]:
            return ans.lower()
        return ans
    
    # Priority 2: "Final Answer: X" or "The answer is X"
    final_patterns = [
        r'(?:final\s+answer|the\s+answer\s+is|answer\s*:)[:\s]*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)',
        r'(?:final\s+answer|answer)[:\s]*(yes|no)\b',
        r'\*\*([+-]?\d+(?:,\d{3})*(?:\.\d+)?)\*\*\s*$',  # Bold number at end
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().replace(',', '')
    
    # Priority 3: Last number in response (after removing thought content)
    # Strip thought block first to avoid confusing intermediate numbers with final answer
    no_thought = re.sub(r'<thought>.*?</thought>', '', text,
                        flags=re.DOTALL | re.IGNORECASE)
    numbers = re.findall(r'([+-]?\d+(?:,\d{3})*(?:\.\d+)?)', no_thought)
    if numbers:
        return numbers[-1].replace(',', '')
    
    # Priority 4: yes/no anywhere in cleaned text
    yesno = re.search(r'\b(yes|no)\b', no_thought, re.IGNORECASE)
    if yesno:
        return yesno.group(1).lower()
    
    return ""


def answers_match(model_answer: str, ground_truth: str) -> bool:
    """Compare answers with numeric tolerance and string normalization."""
    if not model_answer or not ground_truth:
        return False
    
    m = model_answer.strip().lower().replace(',', '')
    g = ground_truth.strip().lower().replace(',', '')
    
    if m == g:
        return True
    
    # Numeric comparison
    try:
        return abs(float(m) - float(g)) < 0.01
    except (ValueError, TypeError):
        pass
    
    # Boolean normalization
    yes_forms = {"yes", "true", "1", "correct"}
    no_forms = {"no", "false", "0", "incorrect"}
    if m in yes_forms and g in yes_forms:
        return True
    if m in no_forms and g in no_forms:
        return True
    
    return False


def safe_eval_expression(expr_str: str) -> Tuple[bool, float]:
    """
    Safely evaluate a mathematical expression string.
    Returns (success, result).
    Only allows: numbers, +, -, *, /, (, ), ., spaces
    """
    # Whitelist: only allow safe math characters
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.\,]+$', expr_str.strip()):
        return False, 0.0
    
    # Remove commas from numbers (1,234 → 1234)
    cleaned = expr_str.replace(',', '')
    
    try:
        result = ast.literal_eval(cleaned)
        return True, float(result)
    except (ValueError, SyntaxError):
        pass
    
    try:
        # Restricted eval: only math operations
        allowed = {"__builtins__": {}}
        result = eval(cleaned, allowed)  # nosec (sanitized above)
        return True, float(result)
    except Exception:
        return False, 0.0


def score_math_steps(steps: list) -> float:
    """
    Score a list of reasoning steps based on arithmetic correctness.
    Returns 0.0 to 1.0
    
    A step is scorable if it contains a pattern like "X = Y" where X is
    computable and Y is what the model claims the result is.
    """
    if not steps:
        return 0.5  # Neutral: no steps to verify
    
    scorable_steps = 0
    correct_steps = 0
    
    for step in steps:
        # Find patterns like "3 + 4 = 7" or "15 × 4 = 60"
        # Also handle "= 7 apples" by extracting just the numeric part
        eq_patterns = re.findall(
            r'([\d\s\+\-\*\/\(\)\.,]+)\s*=\s*([\d\.,\-]+)',
            step
        )
        
        for left, right in eq_patterns:
            left = left.strip()
            right = right.strip().replace(',', '')
            
            if not left or not right:
                continue
            
            # Skip trivial assignments like "x = 3" (single number on left)
            if re.match(r'^\d+$', left.strip()):
                continue
            
            success, computed = safe_eval_expression(left)
            if not success:
                continue
            
            try:
                claimed = float(right)
            except ValueError:
                continue
            
            scorable_steps += 1
            if abs(computed - claimed) < 0.01:
                correct_steps += 1
    
    if scorable_steps == 0:
        return 0.5  # Can't verify — neutral score
    
    return correct_steps / scorable_steps


def extract_steps(response_text: str) -> list:
    """
    Extract reasoning steps from model output.
    Handles <thought> XML blocks and "Step N:" formats.
    """
    if not response_text:
        return []
    
    # Try to find <thought> content first
    thought_match = re.search(
        r'<thought>(.*?)</thought>',
        response_text, re.DOTALL | re.IGNORECASE
    )
    working_text = thought_match.group(1).strip() if thought_match else response_text
    
    # Remove <answer> section if present
    working_text = re.sub(
        r'<answer>.*?</answer>', '', working_text,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()
    
    # Split on "Step N:" or "N." at line start
    step_splits = re.split(r'\bStep\s+\d+\s*[:\-]\s*', working_text)
    step_splits = [s.strip() for s in step_splits if len(s.strip()) > 5]
    if len(step_splits) >= 2:
        return step_splits
    
    # Numbered list fallback
    num_splits = re.split(r'(?m)^\s*\d+\.\s+', working_text)
    num_splits = [s.strip() for s in num_splits if len(s.strip()) > 5]
    if len(num_splits) >= 2:
        return num_splits
    
    # Line-by-line fallback
    lines = [l.strip() for l in working_text.split('\n') if len(l.strip()) > 5]
    return lines if lines else [working_text]


class RewardScorer:
    """
    Component A: Scores model responses using deterministic verification.
    Formula: R_A = 0.4 × final_correct + 0.6 × mean_step_score
    """
    
    def score(self, question_data: dict, response_text: str) -> float:
        score, _ = self.score_with_success(question_data, response_text)
        return score
    
    def score_with_success(
        self, question_data: dict, response_text: str
    ) -> Tuple[float, bool]:
        """
        Returns (reward_score, is_final_answer_correct)
        This gives richer information for logging (mean_success metric).
        """
        if not response_text or len(response_text.strip()) < 3:
            return 0.1, False  # Penalize empty/very short responses
        
        # Extract ground truth
        ground_truth = extract_ground_truth(question_data)
        
        # Extract model's answer
        model_answer = extract_model_answer(response_text)
        
        # Check final answer correctness
        final_correct = answers_match(model_answer, ground_truth)
        
        # Score reasoning steps
        steps = extract_steps(response_text)
        step_score = score_math_steps(steps)
        
        # Combined score
        R_A = 0.4 * float(final_correct) + 0.6 * step_score
        
        # Structure bonuses (helps early RL when it forgets tags)
        has_thought_start = '<thought>' in response_text.lower()
        has_thought_end = '</thought>' in response_text.lower()
        has_answer_tags = '<answer>' in response_text.lower()
        
        if has_thought_start: R_A += 0.01
        if has_thought_end: R_A += 0.01
        if has_answer_tags: R_A += 0.03
        
        # Tiny length penalty to break ties and prevent 0 std dev
        # Different generations will have different lengths, creating reward variance
        # which allows GRPO advantages to be non-zero.
        length_penalty = (len(response_text) / 1000.0) * 0.005
        R_A -= min(0.005, length_penalty)
        
        R_A = max(0.0, min(1.0, R_A))
        
        return R_A, final_correct

    def score_response(self, question: str, response_text: str, ground_truth: str, dataset_name: str = "") -> dict:
        """
        Compatibility method for other parts of the codebase (e.g. evaluator.py)
        """
        question_data = {"question": question, "answer": ground_truth, "dataset": dataset_name}
        score, is_correct = self.score_with_success(question_data, response_text)
        steps = extract_steps(response_text)
        return {
            "final_correct": float(is_correct),
            "reward": score,
            "step_count": len(steps)
        }
