import re
import ast
from typing import Tuple, List

def verify_arithmetic(expr: str) -> Tuple[bool, float]:
    if not expr:
        return False, 0.0
    sanitized = expr.strip().replace(',', '').replace('×', '*').replace('÷', '/')
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', sanitized):
        return False, 0.0
    if re.match(r'^\s*[\d\.]+\s*$', sanitized):
        try:
            return True, float(sanitized)
        except:
            return False, 0.0
    try:
        val = ast.literal_eval(sanitized)
        return True, float(val)
    except:
        pass
    try:
        val = eval(sanitized, {"__builtins__": {}})
        return True, float(val)
    except:
        return False, 0.0

def verify_step_arithmetic(step_text: str) -> Tuple[int, int]:
    if not step_text:
        return 0, 0
    eqs = re.findall(r'([\d\s\+\-\*\/\(\)\.]+)\s*=\s*([+\-]?\d[\d,\.]*)', step_text)
    correct = 0
    total = 0
    for left, right in eqs:
        left_clean = left.strip()
        right_clean = right.strip().replace(',', '')
        if re.match(r'^\s*[\d\.]+\s*$', left_clean):
            continue
        ok, computed = verify_arithmetic(left_clean)
        if not ok:
            continue
        try:
            claimed = float(right_clean)
        except:
            continue
        total += 1
        if abs(computed - claimed) < 0.01:
            correct += 1
    return correct, total

def verify_logical_consistency(steps: List[str]) -> float:
    if len(steps) < 2:
        return 0.5
    step_nums = []
    for s in steps:
        nums = re.findall(r'[+\-]?\d[\d,\.]*', s)
        nums_float = []
        for n in nums:
            try:
                nums_float.append(float(n.replace(',', '')))
            except:
                pass
        step_nums.append(set(nums_float))
    final = step_nums[-1]
    prior = set()
    for p in step_nums[:-1]:
        prior.update(p)
    if final:
        overlap = len(final & prior) / len(final)
        if overlap > 0.5:
            return 0.7
        elif overlap == 0:
            return 0.3
    return 0.5

class ExecutionVerifier:
    def score_steps(self, steps: List[str]) -> float:
        if not steps:
            return 0.5
        total_correct = 0
        total_checkable = 0
        for step in steps:
            if not isinstance(step, str):
                continue
            c, t = verify_step_arithmetic(step)
            total_correct += c
            total_checkable += t
        if total_checkable == 0:
            return verify_logical_consistency(steps)
        arithmetic_ratio = total_correct / total_checkable
        consistency = verify_logical_consistency(steps)
        return 0.8 * arithmetic_ratio + 0.2 * consistency

    def score_single_step(self, step_text: str) -> float:
        return self.score_steps([step_text])