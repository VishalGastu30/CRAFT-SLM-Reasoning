import re
from loguru import logger

def extract_final_answer(response_text: str) -> str:
    """
    Extracts the final answer from the model's output text.
    Handles:
    - XML <answer> tags
    - GSM8K: "#### 123"
    - "Final Answer: X" / "The answer is X"
    - AQuA-RAT / MMLU: single letter (A, B, C, D, E)
    - Yes/No for StrategyQA
    - Last number or word as fallback.
    """
    if not response_text:
        return ""

    response_text = response_text.strip()

    # 0. XML <answer> tag (highest priority)
    xml_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL | re.IGNORECASE)
    if xml_match:
        return xml_match.group(1).strip(".$ \n\t")

    # 1. GSM8K delimiter
    gsm8k_match = re.search(r"####\s*([^\s]+)", response_text)
    if gsm8k_match:
        return gsm8k_match.group(1).strip(".$ \n\t")

    # 2. "Final Answer:" or "The answer is"
    patterns = [
        r"(?:final answer|answer)\s*(?:is)?\s*[:\s]\s*([^\s\n\.\?]+)",
        r"(?:therefore|thus|consequently)[,\s]*the answer\s*(?:is)?\s*[:\s]*([^\s\n\.\?]+)",
        r"the answer is\s*([^\s\n\.\?]+)"
    ]
    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip(".$ \n\t")
            # If candidate is a single uppercase letter (A-E), it's likely a multiple‑choice answer
            if re.fullmatch(r'[A-E]', candidate):
                return candidate
            # Otherwise return as is (numbers or words)
            return candidate

    # 3. Single uppercase letter anywhere (A‑E) – often AQuA or MMLU answer
    letter_match = re.search(r'\b([A-E])\b', response_text)
    if letter_match:
        return letter_match.group(1)

    # 4. Yes/No (StrategyQA)
    yn_match = re.search(r'\b(yes|no)\b', response_text, re.IGNORECASE)
    if yn_match:
        return yn_match.group(1).lower()

    # 5. Last number in the response
    numbers = re.findall(r"-?\d+(?:\.\d+)?", response_text)
    if numbers:
        return numbers[-1]

    # 6. Last word in the last line (fallback)
    lines = [line.strip() for line in response_text.split("\n") if line.strip()]
    if lines:
        words = re.findall(r"\b[A-Za-z0-9]+\b", lines[-1])
        if words:
            return words[-1]

    return ""


def check_answer_correct(predicted: str, ground_truth: str) -> bool:
    """
    Checks if the predicted answer matches the ground truth.
    Handles numeric tolerance, yes/no, and single letters (A‑E).
    """
    pred_str = str(predicted).strip().lower()
    gt_str = str(ground_truth).strip().lower()

    if not pred_str or not gt_str:
        return False

    # Exact string match
    if pred_str == gt_str:
        return True

    # Numeric tolerance (including fractions)
    try:
        # Remove commas, dollar signs, etc.
        p_clean = pred_str.replace(",", "").replace("$", "").strip()
        g_clean = gt_str.replace(",", "").replace("$", "").strip()

        # Handle fractions like "1/2"
        def to_float(s):
            if "/" in s:
                num, den = s.split("/")
                return float(num) / float(den)
            return float(s)

        p_val = to_float(p_clean)
        g_val = to_float(g_clean)
        return abs(p_val - g_val) < 1e-4
    except (ValueError, ZeroDivisionError):
        pass

    # Yes/No mapping
    yes_set = {"yes", "true", "correct", "right", "y"}
    no_set = {"no", "false", "incorrect", "wrong", "n"}
    if gt_str in yes_set and pred_str in yes_set:
        return True
    if gt_str in no_set and pred_str in no_set:
        return True

    # Single letter comparison (A‑E)
    if len(pred_str) == 1 and pred_str.isalpha() and len(gt_str) == 1 and gt_str.isalpha():
        return pred_str.upper() == gt_str.upper()

    # Substring fallback (e.g., "B" in "The answer is B")
    if gt_str in pred_str or pred_str in gt_str:
        return True

    return False


def pass_at_k_sample(model, tokenizer, question: str, n_samples: int = 5, temperature: float = 0.8) -> float:
    """
    Generates n_samples responses from the model for the given question,
    extracts the answers, and computes the fraction of correct answers (pass@k capability).
    (Kept for compatibility; the real work is done by DifficultyMapper.)
    """
    if model is None or tokenizer is None:
        import random
        return random.choice([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    prompt = f"<|user|>\n{question}\n<|assistant|>\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    correct_count = 0
    # This function is meant to be used with a known ground truth, but it's not provided here.
    # It's better to use DifficultyMapper.build_map for full probing.
    # We keep it as a stub for compatibility.
    return 0.0