import re
from typing import Tuple
from src.phase2_rl.component_a.execution_verifier import ExecutionVerifier

def extract_ground_truth(question_data: dict) -> str:
    raw = str(question_data.get("answer", ""))
    gsm_match = re.search(r'####\s*([+\-]?\d[\d,]*(?:\.\d+)?)', raw)
    if gsm_match:
        return gsm_match.group(1).replace(',', '')
    lower = raw.lower()
    if lower in ("true", "yes", "1"):
        return "yes"
    if lower in ("false", "no", "0"):
        return "no"
    letter = re.search(r'\b([A-E])\b', raw)
    if letter:
        return letter.group(1)
    num = re.search(r'([+\-]?\d[\d,]*(?:\.\d+)?)', raw)
    if num:
        return num.group(1).replace(',', '')
    return raw.strip()

def extract_model_answer(response_text: str) -> str:
    if not response_text:
        return ""
    text = response_text.strip()
    # Priority 1: <answer> tags
    xml = re.search(r'<answer>\s*([\s\S]*?)\s*</answer>', text, re.IGNORECASE)
    if xml:
        inner = xml.group(1).strip()
        num = re.search(r'([+\-]?\d[\d,]*(?:\.\d+)?)', inner)
        if num:
            return num.group(1).replace(',', '')
        yn = re.search(r'\b(yes|no)\b', inner, re.IGNORECASE)
        if yn:
            return yn.group(1).lower()
        letter = re.search(r'\b([A-E])\b', inner)
        if letter:
            return letter.group(1).upper()
        return inner
    # Priority 2: "Final Answer: X"
    for pat in [
        r'(?:final\s+answer|the\s+answer\s+is|answer\s*:)[:\s]*([+\-]?\d[\d,]*(?:\.\d+)?)',
        r'(?:final\s+answer|answer)[:\s]*(yes|no)\b'
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).replace(',', '')
    # Priority 3: #### format
    gsm = re.search(r'####\s*([+\-]?\d[\d,]*(?:\.\d+)?)', text)
    if gsm:
        return gsm.group(1).replace(',', '')
    # Priority 4: last number after </thought>
    after_thought = re.sub(r'<thought>[\s\S]*?</thought>', '', text, flags=re.IGNORECASE)
    nums = re.findall(r'([+\-]?\d[\d,]*(?:\.\d+)?)', after_thought)
    if nums:
        return nums[-1].replace(',', '')
    yn2 = re.search(r'\b(yes|no)\b', after_thought, re.IGNORECASE)
    if yn2:
        return yn2.group(1).lower()
    return ""

def answers_match(model_answer: str, ground_truth: str) -> bool:
    if not model_answer or not ground_truth:
        return False
    ma = model_answer.strip().lower().replace(',', '')
    gt = ground_truth.strip().lower().replace(',', '')
    if ma == gt:
        return True
    try:
        return abs(float(ma) - float(gt)) < 0.01
    except:
        pass
    yes_set = {"yes", "true", "1"}
    no_set = {"no", "false", "0"}
    if ma in yes_set and gt in yes_set:
        return True
    if ma in no_set and gt in no_set:
        return True
    return False

def extract_steps(response_text: str) -> list:
    if not response_text:
        return []
    thought = re.search(r'<thought>([\s\S]*?)</thought>', response_text, re.IGNORECASE)
    working = thought.group(1).strip() if thought else response_text
    working = re.sub(r'<answer>[\s\S]*?</answer>', '', working, flags=re.IGNORECASE)
    splits = re.split(r'\bStep\s+\d+\s*[:\-]\s*', working)
    splits = [s.strip() for s in splits if len(s.strip()) > 5]
    if len(splits) >= 2:
        return splits
    splits2 = re.split(r'(?m)^\s*\d+\.\s+', working)
    splits2 = [s.strip() for s in splits2 if len(s.strip()) > 5]
    if len(splits2) >= 2:
        return splits2
    lines = [l.strip() for l in working.split('\n') if len(l.strip()) > 5]
    return lines if lines else [working]

class RewardScorer:
    def __init__(self):
        self._verifier = ExecutionVerifier()

    def score(self, question_data: dict, response_text: str) -> float:
        rew, _ = self.score_with_success(question_data, response_text)
        return rew

    def score_with_success(self, question_data: dict, response_text: str) -> Tuple[float, bool]:
        if not response_text or len(response_text.strip()) < 3:
            return 0.1, False
        gt = extract_ground_truth(question_data)
        pred = extract_model_answer(response_text)
        correct = answers_match(pred, gt)
        steps = extract_steps(response_text)
        step_score = self._verifier.score_steps(steps)
        R_A = 0.4 * float(correct) + 0.6 * step_score
        has_format = ('<thought>' in response_text.lower() and '<answer>' in response_text.lower())
        if has_format:
            R_A = min(1.0, R_A + 0.05)
        return R_A, correct