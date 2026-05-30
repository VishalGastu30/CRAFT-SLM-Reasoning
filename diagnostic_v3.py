# ════════════════════════════════════════════════════════════════════════
# CRAFT DIAGNOSTIC CELL v3
# Run this FIRST before any training. Takes ~90 seconds.
# If any test fails, DO NOT start training — fix the issue first.
# ════════════════════════════════════════════════════════════════════════

import sys, os, re, torch
sys.path.insert(0, '/kaggle/working/CRAFT-SLM-Reasoning')

print("=" * 65)
print("CRAFT Phase 2 Diagnostics v3")
print("=" * 65)

all_passed = True

# ── TEST 0: SFT checkpoint exists ──────────────────────────────────────
print("\n[0] Checking required file paths...")
SFT_PATH = "checkpoints/sft/final"  # adjust if yours differs
DIFF_MAP = "data/difficulty_map.json"

for name, path_check in [("SFT checkpoint", SFT_PATH), ("Difficulty map", DIFF_MAP)]:
    if os.path.exists(path_check):
        print(f"  ✅ {name}: {path_check}")
    else:
        print(f"  ❌ {name} NOT FOUND: {path_check}")
        print(f"     → Did you run Phase 0 and Phase 1 in this session?")
        all_passed = False

# ── TEST 1: Reward scorer basic correctness ─────────────────────────────
print("\n[1] Testing reward scorer answer extraction...")
try:
    from src.phase2_rl.component_a.reward_scorer import RewardScorer
    scorer = RewardScorer()
    
    # Case 1: correct answer in <answer> tags
    resp_correct = (
        "<thought>\nStep 1: 5 + 3 = 8\nStep 2: 8 - 2 = 6\n</thought>"
        "\n<answer>6</answer>"
    )
    q_correct = {"answer": "6"}
    r_correct, s_correct = scorer.score_with_success(q_correct, resp_correct)
    
    if s_correct and r_correct > 0.7:
        print(f"  ✅ Correct answer detected. score={r_correct:.3f}, success={s_correct}")
    else:
        print(f"  ❌ Correct answer NOT detected. score={r_correct:.3f}, success={s_correct}")
        all_passed = False
    
    # Case 2: wrong answer should NOT be marked correct
    resp_wrong = "<thought>\nStep 1: 5 + 3 = 8\n</thought>\n<answer>99</answer>"
    r_wrong, s_wrong = scorer.score_with_success(q_correct, resp_wrong)
    
    if not s_wrong:
        print(f"  ✅ Wrong answer correctly rejected. score={r_wrong:.3f}")
    else:
        print(f"  ❌ Wrong answer marked as correct. score={r_wrong:.3f}")
        all_passed = False
    
    # Case 3: rewards must vary (not all 0.2)
    if abs(r_correct - r_wrong) > 0.1:
        print(f"  ✅ Reward variance confirmed. Spread: {abs(r_correct - r_wrong):.3f}")
    else:
        print(f"  ❌ Rewards too similar ({r_correct:.3f} vs {r_wrong:.3f}) — GRPO will not learn")
        all_passed = False
    
    # Case 4: GSM8K #### format parsing
    q_gsm = {"answer": "Janet sells eggs.\n#### 9"}
    resp_gsm = "<thought>\nStep 1: 9 eggs.\n</thought>\n<answer>9</answer>"
    r_gsm, s_gsm = scorer.score_with_success(q_gsm, resp_gsm)
    
    if s_gsm:
        print(f"  ✅ GSM8K #### format parsed correctly. score={r_gsm:.3f}")
    else:
        print(f"  ❌ GSM8K #### format NOT parsed. score={r_gsm:.3f}")
        all_passed = False

except Exception as scorer_error:
    print(f"  ❌ RewardScorer failed to import or run: {scorer_error}")
    all_passed = False

# ── TEST 2: Execution verifier no SyntaxWarnings ────────────────────────
print("\n[2] Testing execution verifier for SyntaxWarnings...")
import warnings
try:
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        
        from src.phase2_rl.component_a.execution_verifier import ExecutionVerifier
        verifier = ExecutionVerifier()
        
        # Run with various inputs that previously triggered warnings
        test_steps = [
            "Step 1: 3 + 4 = 7 apples",
            "Step 2: 7 - 2 = 5 apples",
            "Step 3: 5 × 3 = 15 total",
        ]
        verifier_result = verifier.score_steps(test_steps)
        
        # Check for SyntaxWarnings specifically
        syntax_warnings = [
            w for w in caught_warnings
            if issubclass(w.category, SyntaxWarning)
        ]
        
        if not syntax_warnings:
            print(f"  ✅ No SyntaxWarnings. Step score = {verifier_result:.3f}")
        else:
            for w in syntax_warnings:
                print(f"  ❌ SyntaxWarning: {w.message}")
                print(f"     in file: {w.filename}, line: {w.lineno}")
            all_passed = False
        
        # Check verifier produces sensible output
        if 0.0 <= verifier_result <= 1.0:
            print(f"  ✅ Verifier output in valid range [0,1]")
        else:
            print(f"  ❌ Verifier output out of range: {verifier_result}")
            all_passed = False

except Exception as verifier_error:
    print(f"  ❌ ExecutionVerifier failed: {verifier_error}")
    all_passed = False

# ── TEST 3: KL controller warmup ────────────────────────────────────────
print("\n[3] Testing KL controller warmup behavior...")
try:
    from src.phase2_rl.component_c.kl_controller import KLController
    kl_ctrl = KLController(
        initial_beta=0.04,
        warmup_steps=20,
        adjustment_factor=0.05,
    )
    
    # Simulate 25 steps with very low KL (as seen in early training)
    for i in range(25):
        kl_ctrl.step(kl_divergence=0.001)  # Very low KL — would normally trigger loosening
    
    beta_after_25 = kl_ctrl.get_beta()
    
    # After 25 steps with warmup=20: should only have loosened for 5 steps
    # 5 steps × 5% loosening = beta should be at ~0.04 × 0.95^5 ≈ 0.031
    # Without warmup it would be 0.04 × 0.90^25 ≈ 0.007 (near floor)
    if beta_after_25 > 0.025:
        print(f"  ✅ KL warmup working. beta after 25 steps: {beta_after_25:.4f}")
    else:
        print(f"  ❌ KL beta decayed too fast. beta after 25 steps: {beta_after_25:.4f}")
        print(f"     Expected > 0.025. This will cause instability.")
        all_passed = False

except Exception as kl_error:
    print(f"  ❌ KLController test failed: {kl_error}")
    all_passed = False

# ── TEST 4: GPU and memory check ────────────────────────────────────────
print("\n[4] Checking GPU status...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    allocated_gb = torch.cuda.memory_allocated() / 1e9
    free_gb = total_gb - allocated_gb
    
    print(f"  ✅ GPU: {gpu_name}")
    print(f"  ✅ Total VRAM: {total_gb:.1f}GB | Free: {free_gb:.1f}GB")
    
    if free_gb < 8.0:
        print(f"  ⚠️  WARNING: Less than 8GB free. Training may OOM.")
        print(f"     Restart the Kaggle session to free memory before training.")
    else:
        print(f"  ✅ Sufficient free VRAM for training.")
else:
    print("  ❌ No GPU detected. Training will be extremely slow on CPU only.")
    all_passed = False

# ── TEST 5: Advantage computation safety ────────────────────────────────
print("\n[5] Testing advantage computation safety...")
try:
    # Simulate worst case: all rewards identical (zero variance)
    identical_rewards = [0.3, 0.3, 0.3, 0.3]
    rewards_t = torch.tensor(identical_rewards, dtype=torch.float32)
    std_val = rewards_t.std().item()
    
    if std_val < 0.01:
        # Safe path: don't normalize
        advantages_test = rewards_t - rewards_t.mean()
        max_adv = advantages_test.abs().max().item()
        if max_adv < 1.0:
            print(f"  ✅ Zero-variance group handled safely. Max advantage: {max_adv:.4f}")
        else:
            print(f"  ❌ Zero-variance group produced large advantages: {max_adv:.4f}")
            all_passed = False
    
    # Normal case: varying rewards
    varied_rewards = [0.1, 0.5, 0.8, 0.3]
    rewards_t2 = torch.tensor(varied_rewards, dtype=torch.float32)
    advantages_test2 = (rewards_t2 - rewards_t2.mean()) / (rewards_t2.std() + 1e-8)
    max_adv2 = advantages_test2.abs().max().item()
    
    if max_adv2 < 5.0:
        print(f"  ✅ Normal variance group advantages in range. Max: {max_adv2:.4f}")
    else:
        print(f"  ⚠️  Large advantages in normal case: {max_adv2:.4f} (may cause instability)")

except Exception as adv_error:
    print(f"  ❌ Advantage computation test failed: {adv_error}")
    all_passed = False

# ── FINAL RESULT ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if all_passed:
    print("✅ ALL DIAGNOSTIC TESTS PASSED.")
    print("✅ Safe to start training.")
    print("=" * 65)
else:
    print("❌ SOME TESTS FAILED. Fix the issues above before training.")
    print("❌ Starting training with failed tests = wasted compute hours.")
    print("=" * 65)
