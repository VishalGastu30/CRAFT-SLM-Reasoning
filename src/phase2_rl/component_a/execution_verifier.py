import re
import ast
import operator
from typing import List, Tuple, Dict, Any
from loguru import logger
from src.phase0_probe.sampler import extract_final_answer, check_answer_correct

# Safe AST operators for mathematical evaluation
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos
}

def safe_eval(expr_str: str) -> float:
    """
    Evaluates a mathematical expression string safely using AST to prevent code injection.
    Only allows basic arithmetic operations and numbers.
    """
    expr_str = expr_str.replace("=", "").strip()
    # Replace common multiplication symbols and whitespace
    expr_str = expr_str.replace("x", "*").replace("X", "*").replace(",", "")
    
    try:
        node = ast.parse(expr_str, mode='eval').body
        return _eval_node(node)
    except Exception as e:
        logger.debug(f"Failed to safely evaluate expression '{expr_str}': {e}")
        return float('nan')

def _eval_node(node):
    if isinstance(node, ast.Num):  # <3.8 compatibility
        return node.n
    elif isinstance(node, ast.Constant):  # >=3.8
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise TypeError(f"Unsupported operator: {op_type}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise TypeError(f"Unsupported unary operator: {op_type}")
        operand = _eval_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    else:
        raise TypeError(f"Unsupported AST node type: {type(node)}")

def extract_math_expressions(step_text: str) -> List[Tuple[str, str]]:
    """
    Extracts equation patterns from a step, e.g. "3 * 10 = 30" or "x = 4 + 2".
    Returns a list of tuples: (left_hand_side_expression, right_hand_side_value)
    """
    # Regex to capture equations like expression = expression/number
    # Ignores isolated single numbers or step labels
    equations = re.findall(r"([0-9\+\-\*\/\(\)\s\.\^xX]+)\s*=\s*(-?[0-9\s\.\/]+)", step_text)
    
    valid_equations = []
    for lhs, rhs in equations:
        lhs = lhs.strip()
        rhs = rhs.strip()
        # Ensure lhs actually has operators to avoid matching trivial "number = number" or "Step 1 = ..."
        if any(op in lhs for op in ["+", "-", "*", "/", "^"]):
            # Normalize 'x' or 'X' multiplication to '*'
            lhs_norm = lhs.replace("x", "*").replace("X", "*")
            valid_equations.append((lhs_norm, rhs))
            
    return valid_equations

def verify_expression(lhs_expr: str, rhs_val: str) -> bool:
    """
    Verifies if LHS mathematical expression matches RHS mathematical expression or number.
    Returns True if mathematically correct, False otherwise.
    """
    try:
        lhs_evaluated = safe_eval(lhs_expr)
        rhs_evaluated = safe_eval(rhs_val)
        
        if float('nan') in (lhs_evaluated, rhs_evaluated) or float('inf') in (lhs_evaluated, rhs_evaluated):
            return False
            
        return abs(lhs_evaluated - rhs_evaluated) < 1e-4
    except Exception:
        return False

def extract_steps(response_text: str) -> List[str]:
    """
    Splits reasoning traces into individual step blocks by looking for step patterns:
    "Step 1:", "Step 2:", etc.
    """
    if not response_text:
        return []
        
    # Split using step prefix delimiters, keeping the step text intact
    steps = re.split(r"(?:^|\n)Step \d+\s*[:\-]\s*", response_text.strip())
    
    # Filter out empty or extremely small matches (which might precede the first Step 1)
    cleaned_steps = [s.strip() for s in steps if len(s.strip()) > 3]
    
    # If no explicit steps were formatted but we have lines, treat each line as a step as fallback
    if not cleaned_steps:
        cleaned_steps = [line.strip() for line in response_text.split("\n") if len(line.strip()) > 5]
        
    return cleaned_steps

def score_reasoning_chain(response_text: str, ground_truth: str) -> Dict[str, Any]:
    """
    Calculates execution verifier scores for a full mathematical reasoning chain.
    Formula: R_A = 0.4 * final_correct + 0.6 * mean_step_score
    Returns a dictionary of detailed metrics.
    """
    steps = extract_steps(response_text)
    pred_final = extract_final_answer(response_text)
    
    final_correct = 1.0 if check_answer_correct(pred_final, ground_truth) else 0.0
    
    if not steps:
        return {
            "final_correct": final_correct,
            "mean_step_score": 0.0,
            "reward": 0.4 * final_correct,
            "step_count": 0,
            "invalid_steps": 0
        }
        
    step_scores = []
    invalid_steps = 0
    
    for step in steps:
        equations = extract_math_expressions(step)
        if not equations:
            # Step has no math expressions, defaults to neutral score (1.0) to avoid penalizing textual explanation
            step_scores.append(1.0)
            continue
            
        # If there are mathematical equations in this step, verify them
        step_correct = True
        for lhs, rhs in equations:
            if not verify_expression(lhs, rhs):
                step_correct = False
                break
                
        if step_correct:
            step_scores.append(1.0)
        else:
            step_scores.append(0.0)
            invalid_steps += 1
            
    mean_step_score = sum(step_scores) / len(step_scores)
    reward = 0.4 * final_correct + 0.6 * mean_step_score
    
    return {
        "final_correct": final_correct,
        "mean_step_score": mean_step_score,
        "reward": reward,
        "step_count": len(steps),
        "invalid_steps": invalid_steps
    }
