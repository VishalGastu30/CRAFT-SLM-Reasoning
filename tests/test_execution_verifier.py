import pytest
from src.phase2_rl.component_a.execution_verifier import (
    safe_eval,
    extract_math_expressions,
    verify_expression,
    extract_steps,
    score_reasoning_chain
)
from src.phase2_rl.component_a.reward_scorer import RewardScorer

def test_safe_eval():
    assert safe_eval("3 * 10") == 30
    assert safe_eval("15 + 5 - 2") == 18
    assert safe_eval("2^3") == 8
    assert safe_eval("10 / 4") == 2.5
    assert safe_eval("-5") == -5
    
    # Verify non-math or potentially dangerous expressions fail safely to NaN
    import math
    assert math.isnan(safe_eval("import os; os.system('echo')"))
    assert math.isnan(safe_eval("__import__('os').system('echo')"))
    assert math.isnan(safe_eval("x = 5"))
    assert math.isnan(safe_eval("3 + [1, 2]"))

def test_extract_math_expressions():
    # Math expressions
    step_with_math = "Step 1: John buys 3 books at $10 each so 3 * 10 = 30."
    exprs = extract_math_expressions(step_with_math)
    assert len(exprs) == 1
    assert exprs[0] == ("3 * 10", "30")
    
    # Multiple math expressions
    step_multiple = "First we find 4 + 5 = 9, and then 9 * 2 = 18."
    exprs_mult = extract_math_expressions(step_multiple)
    assert len(exprs_mult) == 2
    assert exprs_mult[0] == ("4 + 5", "9")
    assert exprs_mult[1] == ("9 * 2", "18")
    
    # Text step with no equations
    step_no_math = "Step 2: John walks to the store."
    assert len(extract_math_expressions(step_no_math)) == 0

def test_verify_expression():
    assert verify_expression("3 * 10", "30") is True
    assert verify_expression("15 + 5", "21") is False
    assert verify_expression("1/3", "0.33333") is True
    assert verify_expression("2 + 2", "5") is False

def test_extract_steps():
    text = "Step 1: First buy items. Step 2: Pay for items. Step 3: Go home."
    steps = extract_steps(text)
    assert len(steps) == 3
    assert steps[0] == "First buy items."
    assert steps[1] == "Pay for items."
    assert steps[2] == "Go home."

def test_score_reasoning_chain():
    # Perfect score: final answer is correct, and all math steps are correct
    perfect_response = (
        "Step 1: 3 * 10 = 30.\n"
        "Step 2: 30 + 15 = 45.\n"
        "Final Answer: 45"
    )
    result_perfect = score_reasoning_chain(perfect_response, "45")
    assert result_perfect["final_correct"] == 1.0
    assert result_perfect["mean_step_score"] == 1.0
    assert result_perfect["reward"] == 1.0
    assert result_perfect["invalid_steps"] == 0
    
    # Flawed reasoning: final answer correct, but one math step is incorrect
    flawed_response = (
        "Step 1: 3 * 10 = 25.\n"  # Wrong math step
        "Step 2: 25 + 20 = 45.\n"
        "Final Answer: 45"
    )
    result_flawed = score_reasoning_chain(flawed_response, "45")
    assert result_flawed["final_correct"] == 1.0
    assert result_flawed["mean_step_score"] == 0.5
    assert result_flawed["reward"] == 0.4 * 1.0 + 0.6 * 0.5  # 0.7
    assert result_flawed["invalid_steps"] == 1

def test_reward_scorer_routing():
    scorer = RewardScorer()
    
    # Verify math routing
    assert scorer.is_math_problem("What is 15% of 240?", "gsm8k") is True
    assert scorer.is_math_problem("Would a penguin survive in Sahara?", "strategyqa") is False
    
    # Test scoring routing
    math_response = "Step 1: 15/100 * 240 = 36.\nFinal Answer: 36"
    res_math = scorer.score_response("What is 15% of 240?", math_response, "36", "gsm8k")
    assert res_math["verifier_type"] == "math"
    assert res_math["reward"] == 1.0
    
    # Test logical logic routing (mock-mode DeBERTa fallback)
    logic_response = "Step 1: Penguins are native to cold climates.\nStep 2: Sahara is very hot.\nFinal Answer: no"
    res_logic = scorer.score_response("Would a penguin survive in Sahara?", logic_response, "no", "strategyqa")
    assert res_logic["verifier_type"] == "nli"
    assert res_logic["reward"] > 0.0
