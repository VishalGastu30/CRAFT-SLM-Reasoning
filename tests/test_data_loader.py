import pytest
from src.phase1_sft.data_loader import format_gsm8k_target, format_aqua_target

def test_format_gsm8k_target():
    # Test valid GSM8K format with steps
    answer_text = "To find the total, John first buys books. 3 * 10 = 30. Then he adds a bag. 30 + 15 = 45. #### 45"
    formatted = format_gsm8k_target(answer_text)
    
    assert "Step 1:" in formatted
    assert "Final Answer: 45" in formatted
    assert "####" not in formatted
    
    # Test GSM8K without #### delimiter
    no_delimiter = "Simple answer is 12"
    formatted_no_del = format_gsm8k_target(no_delimiter)
    assert "Step 1:" in formatted_no_del
    assert "Final Answer: Simple answer is 12" in formatted_no_del

def test_format_aqua_target():
    # Test AQuA format
    rationale = "Let the number be x. 2x + 5 = 15. 2x = 10. x = 5."
    options = "['A. 2', 'B. 3', 'C. 5', 'D. 7']"
    correct_letter = "C"
    
    formatted = format_aqua_target(rationale, options, correct_letter)
    
    assert "Step 1:" in formatted
    assert "Evaluating options:" in formatted
    assert "A. 2, B. 3, C. 5, D. 7" in formatted
    assert "Final Answer: C" in formatted
