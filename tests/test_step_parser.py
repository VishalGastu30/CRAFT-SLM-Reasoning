import pytest
from src.phase2_rl.component_b.step_parser import StepParser
from src.phase2_rl.component_b.contrastive_builder import ContrastiveBuilder

def test_step_parser_standard():
    parser = StepParser()
    trace = "Step 1: First buy items.\nStep 2: Pay for them.\nFinal Answer: 15"
    
    steps = parser.parse_trace(trace)
    assert len(steps) == 2
    assert steps[0]["step_num"] == 1
    assert steps[0]["content"] == "First buy items."
    assert steps[1]["step_num"] == 2
    assert steps[1]["content"] == "Pay for them."

def test_step_parser_edge_cases():
    parser = StepParser()
    
    # Trace with blank lines and irregular spacing
    trace_messy = "\n\nStep 1 :  First buy items.   \n\n\nStep 2-  Pay for them.   \nFinal Answer: yes"
    steps = parser.parse_trace(trace_messy)
    assert len(steps) == 2
    assert steps[0]["content"] == "First buy items."
    assert steps[1]["content"] == "Pay for them."
    
    # Fallback to lines if no explicit step tags are present
    trace_plain = "First plain reasoning line.\nSecond plain reasoning line.\nFinal Answer: 5"
    steps_plain = parser.parse_trace(trace_plain)
    assert len(steps_plain) >= 2
    assert steps_plain[0]["content"] == "First plain reasoning line."

def test_contrastive_builder():
    builder = ContrastiveBuilder()
    
    question = "What is 3 * 10?"
    
    # Perfect correct trace
    trace_correct = "Step 1: John buys 3 books at $10. 3 * 10 = 30.\nStep 2: Done.\nFinal Answer: 30"
    # Trace with math error on Step 1
    trace_error = "Step 1: John buys 3 books at $10. 3 * 10 = 25.\nStep 2: Done.\nFinal Answer: 25"
    
    traces = [
        {"trace_text": trace_correct},
        {"trace_text": trace_error}
    ]
    
    pairs = builder.build_step_pairs(question, traces, "30")
    
    # There should be exactly 1 step pair since Step 1 differs and one is mathematically incorrect
    assert len(pairs) == 1
    assert "What is 3 * 10?" in pairs[0]["prompt"]
    assert "Step 1:" in pairs[0]["prompt"]
    
    # The correct step is chosen, incorrect is rejected
    assert "3 * 10 = 30" in pairs[0]["chosen"]
    assert "3 * 10 = 25" in pairs[0]["rejected"]
