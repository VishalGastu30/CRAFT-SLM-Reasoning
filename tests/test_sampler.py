import pytest
from src.phase0_probe.sampler import extract_final_answer, check_answer_correct
from src.phase0_probe.difficulty_mapper import DifficultyMapper

def test_extract_final_answer():
    # Test GSM8K style
    assert extract_final_answer("The steps are done. #### 42") == "42"
    assert extract_final_answer("Step 1: calculate. #### 3.14") == "3.14"
    
    # Test standard patterns
    assert extract_final_answer("Final Answer: 15") == "15"
    assert extract_final_answer("Therefore, the answer is yes") == "yes"
    assert extract_final_answer("The answer is: 250") == "250"
    
    # Test fallbacks
    assert extract_final_answer("The result is 100") == "100"
    assert extract_final_answer("No numbers but word") == "word"
    assert extract_final_answer("") == ""

def test_check_answer_correct():
    # Exact match
    assert check_answer_correct("42", "42") is True
    assert check_answer_correct("yes", "yes") is True
    
    # Whitespace/case tolerance
    assert check_answer_correct("  42  ", "42") is True
    assert check_answer_correct("YES", "yes") is True
    
    # Numerical tolerance
    assert check_answer_correct("3.14159", "3.1416") is True
    assert check_answer_correct("3.1", "3") is False
    assert check_answer_correct("$12.50", "12.5") is True
    assert check_answer_correct("1,000", "1000") is True
    
    # Fractions
    assert check_answer_correct("1/2", "0.5") is True
    assert check_answer_correct("3/4", "0.75") is True
    
    # Boolean words normalization
    assert check_answer_correct("true", "yes") is True
    assert check_answer_correct("no", "false") is True
    assert check_answer_correct("incorrect", "no") is True

def test_difficulty_mapper():
    mapper = DifficultyMapper()
    
    # Build simulated map
    questions = [
        {"id": "q1", "question": "What is 1+1?", "answer": "2", "dataset": "gsm8k"},
        {"id": "q2", "question": "What is 2+2?", "answer": "4", "dataset": "gsm8k"},
        {"id": "q3", "question": "Is earth flat?", "answer": "no", "dataset": "strategyqa"}
    ]
    
    # Since model/tokenizer is None, it runs simulated difficulty values
    mapper.build_map(None, None, questions, n_samples=5)
    
    assert len(mapper.difficulty_map) == 3
    for q_id in ["q1", "q2", "q3"]:
        assert q_id in mapper.difficulty_map
        assert "difficulty" in mapper.difficulty_map[q_id]
        
    # Test learning zone
    # Set custom difficulties to test filtering
    mapper.difficulty_map["q1"]["difficulty"] = 0.2  # Easy
    mapper.difficulty_map["q2"]["difficulty"] = 0.5  # Medium (in zone)
    mapper.difficulty_map["q3"]["difficulty"] = 0.9  # Hard
    
    zone = mapper.get_learning_zone(low=0.4, high=0.7)
    assert len(zone) == 1
    assert zone[0]["id"] == "q2"
