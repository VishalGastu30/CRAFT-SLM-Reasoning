import pytest
from src.phase2_rl.component_c.accuracy_tracker import AccuracyTracker
from src.phase2_rl.component_c.kl_controller import KLController
from src.phase2_rl.component_c.curriculum_engine import CurriculumEngine

def test_accuracy_tracker():
    tracker = AccuracyTracker(window_size=10, stability_threshold=0.7)
    
    # Half-full window check (stability requires len >= window_size//2, i.e. 5)
    for _ in range(4):
        tracker.add(True)
    assert tracker.is_stable() is False  # Not enough data yet
    
    # 5th element added
    tracker.add(True)
    assert tracker.is_stable() is True
    assert tracker.get_accuracy() == 1.0
    
    # Add incorrect samples
    for _ in range(5):
        tracker.add(False)
    # accuracy is now 5/10 = 0.5
    assert tracker.get_accuracy() == 0.5
    assert tracker.is_stable() is False
    
    # Reset
    tracker.reset()
    assert tracker.get_accuracy() == 0.0
    assert len(tracker.window) == 0

def test_kl_controller():
    controller = KLController(initial_beta=0.04, kl_target=0.1, min_beta=0.01, max_beta=0.2)
    
    # Test stable / within bounds KL
    assert controller.step(0.12) == 0.04  # No change
    
    # Test high KL (should tighten)
    beta_high = controller.step(0.16)  # exceeds 1.5x target (0.15)
    assert beta_high == 0.06  # 0.04 * 1.5
    
    # Test low KL (should loosen)
    beta_low = controller.step(0.04)  # below 0.5x target (0.05)
    assert beta_low == 0.054  # 0.06 * 0.9

def test_curriculum_engine():
    # Setup dummy difficulty map
    dummy_map = {
        "q1": {"difficulty": 0.2, "question": "q1", "answer": "a1"},
        "q2": {"difficulty": 0.5, "question": "q2", "answer": "a2"},
        "q3": {"difficulty": 0.7, "question": "q3", "answer": "a3"},
        "q4": {"difficulty": 0.8, "question": "q4", "answer": "a4"}
    }
    
    # Save map temporarily
    import json
    import os
    map_path = "test_difficulty_map.json"
    with open(map_path, "w") as f:
        json.dump(dummy_map, f)
        
    try:
        engine = CurriculumEngine(
            difficulty_map_path=map_path,
            initial_range=(0.4, 0.7),
            window_size=10,
            stability_threshold=0.8
        )
        
        # Verify active pool (only q2 and q3 should be in range 0.4 to 0.7)
        pool = engine.get_active_pool()
        assert len(pool) == 2
        q_ids = {q["id"] for q in pool}
        assert "q2" in q_ids
        assert "q3" in q_ids
        
        # Test batch sampling
        batch = engine.sample_batch(2)
        assert len(batch) == 2
        
        # Test dynamic difficulty expansion on stability
        for _ in range(5):
            engine.update_accuracy(True)
            
        # Accuracy tracker is stable now -> boundaries expanded
        assert engine.max_difficulty > 0.7
        
        # Pool should now include q4 (difficulty 0.9)
        new_pool = engine.get_active_pool()
        new_q_ids = {q["id"] for q in new_pool}
        assert "q4" in new_q_ids
        
    finally:
        if os.path.exists(map_path):
            os.remove(map_path)
