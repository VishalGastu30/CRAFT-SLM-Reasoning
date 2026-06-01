import sys
import os
import torch
import warnings

def run_diagnostics():
    print("=" * 65)
    print("CRAFT Phase 2 Diagnostics v3")
    print("=" * 65)

    all_passed = True

    # ── TEST 0: Required Files ──────────────────────────────────────
    print("\n[0] Checking required file paths...")
    SFT_PATH = "checkpoints/sft/final"
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
        
        # Case 1: correct answer
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
        
        # Case 2: wrong answer
        resp_wrong = "<thought>\nStep 1: 5 + 3 = 8\n</thought>\n<answer>99</answer>"
        r_wrong, s_wrong = scorer.score_with_success(q_correct, resp_wrong)
        
        if not s_wrong:
            print(f"  ✅ Wrong answer correctly rejected. score={r_wrong:.3f}")
        else:
            print(f"  ❌ Wrong answer marked as correct. score={r_wrong:.3f}")
            all_passed = False
            
    except Exception as e:
        print(f"  ❌ RewardScorer failed: {e}")
        all_passed = False

    # ── TEST 2: KL controller warmup ────────────────────────────────────────
    print("\n[2] Testing KL controller warmup behavior...")
    try:
        from src.phase2_rl.component_c.kl_controller import KLController
        kl_ctrl = KLController(
            initial_beta=0.04,
            warmup_steps=20,
            adjustment_factor=0.05,
        )
        for i in range(25):
            kl_ctrl.step(kl_divergence=0.001)
        
        beta_after_25 = kl_ctrl.get_beta()
        if beta_after_25 > 0.025:
            print(f"  ✅ KL warmup working. beta after 25 steps: {beta_after_25:.4f}")
        else:
            print(f"  ❌ KL beta decayed too fast. beta: {beta_after_25:.4f}")
            all_passed = False
    except Exception as e:
        print(f"  ❌ KLController test failed: {e}")
        all_passed = False

    # ── TEST 3: GPU Check ────────────────────────────────────────
    print("\n[3] Checking GPU status...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✅ GPU: {gpu_name} (Total VRAM: {total_gb:.1f}GB)")
    else:
        print("  ❌ No GPU detected. Training will be extremely slow.")
        all_passed = False

    # ── FINAL RESULT ────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    if all_passed:
        print("✅ ALL DIAGNOSTIC TESTS PASSED. Safe to start training.")
    else:
        print("❌ SOME TESTS FAILED. Fix issues above before training.")
        sys.exit(1)
    print("=" * 65)

if __name__ == "__main__":
    run_diagnostics()
