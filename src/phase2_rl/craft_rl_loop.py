"""
craft_rl_loop.py — CRAFT Phase 2 RL Training Loop
Implements GRPO with:
- Component A: Deterministic execution verifier (reward signal)
- Component B: Contrastive step-level DPO (activates at step 100)
- Component C: Adaptive curriculum + dynamic KL controller
"""

import os
import sys
import json
import argparse
import importlib
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
from loguru import logger

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.logger import setup_logger
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.hardware_detector import detect_hardware
from src.config import load_config
from src.phase0_probe.difficulty_mapper import DifficultyMapper
from src.phase2_rl.component_a.reward_scorer import RewardScorer
from src.phase2_rl.component_b.trace_generator import TraceGenerator
from src.phase2_rl.component_b.contrastive_builder import ContrastiveBuilder
from src.phase2_rl.component_b.dpo_trainer import StepDPOTrainer
from src.phase2_rl.component_c.curriculum_engine import CurriculumEngine
from src.phase2_rl.component_c.kl_controller import KLController
from src.phase2_rl.multi_dataset_sampler import MultiDatasetSampler

def _sample_from_curriculum(curriculum, step):
    question_batch = curriculum.get_next_batch(batch_size=1)
    if not question_batch:
        logger.warning(f"Step {step}: No eligible questions in curriculum range. Expanding range temporarily.")
        curriculum.expand_range_temporarily()
        question_batch = curriculum.get_next_batch(batch_size=1)
    return question_batch[0] if question_batch else None


def load_model_and_tokenizer(
    base_model_path: str,
    sft_adapter_path: str,
    config: dict,
    device: str
):
    """
    Load the base model, apply SFT LoRA adapter, and prepare for RL training.
    
    CRITICAL: The LoRA adapter parameters MUST have requires_grad=True after loading.
    This is what allows gradients to flow during the policy loss computation.
    The base model parameters remain frozen (requires_grad=False).
    """
    logger.info(f"Loading tokenizer from: {sft_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        sft_adapter_path,
        trust_remote_code=False,  # Use native HF implementation
        padding_side="left",      # Left padding is required for batch generation
    )
    
    # Ensure pad token exists (Phi-3-Mini uses eos as pad by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading base model from: {base_model_path}")
    
    # Check if flash attention is available
    flash_available = importlib.util.find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if flash_available else "eager"
    logger.info(f"Attention implementation: {attn_impl}")
    
    # 4-bit quantization config for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    
    logger.info(f"Applying SFT adapter from {sft_adapter_path}")
    
    # Apply SFT LoRA adapter
    policy_model = PeftModel.from_pretrained(
        base_model,
        sft_adapter_path,
        is_trainable=True,   # CRITICAL: must be True for RL training
    )
    # Enable gradient checkpointing to drastically save VRAM during forward passes
    policy_model.gradient_checkpointing_enable()
    
    # Verify that LoRA parameters require gradients
    lora_params_with_grad = sum(
        1 for name, p in policy_model.named_parameters()
        if 'lora' in name.lower() and p.requires_grad
    )
    logger.info(f"LoRA parameters with requires_grad=True: {lora_params_with_grad}")
    
    if lora_params_with_grad == 0:
        # Emergency fix: manually enable gradients on LoRA parameters
        logger.warning("No LoRA params have requires_grad=True. Enabling manually.")
        for name, param in policy_model.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad_(True)
        lora_params_with_grad = sum(
            1 for name, p in policy_model.named_parameters()
            if 'lora' in name.lower() and p.requires_grad
        )
        logger.info(f"After manual fix: {lora_params_with_grad} LoRA params require grad")
    
    # Load reference model (frozen copy — used for KL divergence computation)
    # This model NEVER updates — it's the anchor for stability
    logger.info("Loading frozen reference model for KL computation...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation=attn_impl,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )
    ref_model = PeftModel.from_pretrained(
        ref_base,
        sft_adapter_path,
        is_trainable=False,  # CRITICAL: reference model must NOT be trainable
    )
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    
    logger.info("Models loaded successfully.")
    return policy_model, ref_model, tokenizer


def compute_grpo_loss(
    policy_model,
    ref_model,
    tokenizer,
    prompt_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    rewards: list,
    kl_beta: float,
) -> tuple:
    """
    Compute the full GRPO + KL loss.
    
    This function is the CORE of correct RL training. Read carefully.
    
    Args:
        policy_model: The model being trained (with requires_grad=True on LoRA layers)
        ref_model: The frozen reference model (requires_grad=False everywhere)
        tokenizer: For padding/masking
        prompt_ids: The input prompt token IDs, shape [1, prompt_len]
        generated_ids: All generated sequences, shape [group_size, prompt_len + response_len]
        rewards: List of float reward scores, length = group_size
        kl_beta: Current KL penalty coefficient
    
    Returns:
        (total_loss_tensor, grpo_loss_float, kl_loss_float, advantages_list)
    """
    group_size = len(rewards)
    prompt_len = prompt_ids.shape[1]
    
    # ── GRPO ADVANTAGE COMPUTATION (safe version) ───────────────────────────────
    
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    reward_mean = rewards_tensor.mean()
    reward_std = rewards_tensor.std()
    
    centered_advantages = rewards_tensor - reward_mean
    
    # Only normalize if there's meaningful variance
    # Threshold: std must be at least 0.01 (1% of reward scale)
    if reward_std.item() > 0.01:
        advantages = centered_advantages / (reward_std + 1e-8)
    else:
        # All rewards similar — don't normalize, just center
        # This produces near-zero loss which is correct: nothing to learn from
        # a group where all outputs got the same reward
        advantages = centered_advantages
    
    # ── POLICY FORWARD PASS (with gradients) ──────────────────────────────────
    # This is the critical section. policy_model MUST be in training mode.
    # The forward pass builds the computation graph through LoRA layers.
    policy_model.train()
    
    # Run all generated sequences through policy model in one batch
    # generated_ids: [group_size, total_seq_len]
    policy_outputs = policy_model(generated_ids)
    policy_logits = policy_outputs.logits
    # policy_logits: [group_size, total_seq_len, vocab_size]
    # This tensor HAS a grad_fn because it passed through LoRA layers ✓
    
    # We only want log probs for the RESPONSE part (not the prompt)
    # Shift by 1: to predict token[t], we use logits[t-1]
    response_logits = policy_logits[:, prompt_len - 1 : -1, :]
    # response_logits: [group_size, response_len, vocab_size]
    
    response_labels = generated_ids[:, prompt_len:]
    # response_labels: [group_size, response_len]
    
    # Compute log softmax over vocabulary dimension
    response_log_probs_vocab = F.log_softmax(response_logits, dim=-1)
    # response_log_probs_vocab: [group_size, response_len, vocab_size]
    
    # Gather log prob of the ACTUAL token at each position
    token_log_probs = response_log_probs_vocab.gather(
        2, response_labels.unsqueeze(-1)
    ).squeeze(-1)
    # token_log_probs: [group_size, response_len]
    # This STILL has a grad_fn ✓
    
    # Create mask: 1 for real tokens, 0 for padding
    response_mask = (response_labels != tokenizer.pad_token_id).float()
    
    # Sum log probs over response length → sequence-level log probability
    # Divide by response length to normalize (prevents bias toward shorter responses)
    response_lengths = response_mask.sum(dim=1).clamp(min=1)
    sequence_log_probs = (token_log_probs * response_mask).sum(dim=1) / response_lengths
    # sequence_log_probs: [group_size] — HAS grad_fn ✓
    
    # GRPO policy loss
    advantages = advantages.to(sequence_log_probs.device)
    grpo_loss = -(advantages * sequence_log_probs).mean()
    # grpo_loss HAS grad_fn ✓
    
    # ── REFERENCE FORWARD PASS (no gradients) ─────────────────────────────────
    with torch.no_grad():
        ref_outputs = ref_model(generated_ids)
        ref_logits = ref_outputs.logits
        ref_response_logits = ref_logits[:, prompt_len - 1 : -1, :]
        ref_log_probs_vocab = F.log_softmax(ref_response_logits, dim=-1)
        ref_token_log_probs = ref_log_probs_vocab.gather(
            2, response_labels.unsqueeze(-1)
        ).squeeze(-1)
        ref_sequence_log_probs = (ref_token_log_probs * response_mask).sum(dim=1) / response_lengths
    
    # KL divergence: how much has the policy drifted from the reference?
    # KL(policy || ref) = E[log π_policy - log π_ref]
    # sequence_log_probs has grad, ref_sequence_log_probs is detached → kl_loss has grad ✓
    kl_loss_per_seq = sequence_log_probs - ref_sequence_log_probs.detach()
    kl_loss = torch.clamp(kl_loss_per_seq.mean(), min=0.0)
    # kl_loss HAS grad_fn ✓
    
    # Combined loss
    total_loss = grpo_loss + kl_beta * kl_loss
    # total_loss HAS grad_fn ✓ — backward() will work
    
    # Return scalar values for logging (detach from graph)
    return (
        total_loss,
        grpo_loss.item(),
        kl_loss.item(),
        advantages.tolist(),
    )


def generate_group_responses(
    policy_model,
    tokenizer,
    prompt_text: str,
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    device: str,
) -> tuple:
    """
    Generate group_size responses for a given prompt.
    
    IMPORTANT: Generation runs with torch.no_grad() — this is correct.
    Gradients are NOT needed during generation. They are needed during
    the forward pass in compute_grpo_loss(), not here.
    
    Returns:
        (prompt_ids_tensor, generated_ids_tensor, response_texts_list)
    """
    # Tokenize prompt
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,  # Maximum prompt length
    ).to(device)
    
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[1]
    
    policy_model.eval()  # Switch to eval for generation
    
    # Generate all group_size responses in one batched call
    # Batch the prompt group_size times
    batched_input_ids = prompt_ids.repeat(group_size, 1)
    batched_attention_mask = inputs["attention_mask"].repeat(group_size, 1)
    
    with torch.no_grad():
        generated_ids = policy_model.generate(
            batched_input_ids,
            attention_mask=batched_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            # Do not return dict — just the token ids
            return_dict_in_generate=False,
        )
    # generated_ids: [group_size, prompt_len + response_len]
    
    # Decode response portions only (strip the prompt)
    response_texts = []
    for i in range(group_size):
        response_ids = generated_ids[i][prompt_len:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        response_texts.append(response_text)
    
    return prompt_ids, generated_ids, response_texts


def format_prompt(question_data: dict, tokenizer) -> str:
    """
    Format a question into the correct prompt for its dataset type.
    
    CRITICAL: The system prompt format MUST match Phase 1 SFT training format.
    Three dataset-specific variants:
    - GSM8K/AQuA-RAT: math reasoning → number answer
    - StrategyQA: logical reasoning → yes/no answer
    - MMLU: multi-choice reasoning → A/B/C/D answer
    """
    question_text = question_data.get("question", question_data.get("problem", ""))
    dataset = question_data.get("dataset", "gsm8k")

    if dataset == "strategyqa":
        system_prompt = (
            "You are a careful logical reasoner. "
            "Think step by step inside <thought> tags, analyzing whether the "
            "statement is true or false. "
            "Put ONLY 'yes' or 'no' (lowercase) inside <answer> tags.\n\n"
            "Example format:\n"
            "<thought>\nStep 1: [fact check]\nStep 2: [conclusion]\n</thought>\n"
            "<answer>yes</answer>"
        )
    elif dataset == "mmlu":
        system_prompt = (
            "You are a careful reasoner answering multiple-choice questions. "
            "Think step by step inside <thought> tags, evaluating each option. "
            "Put ONLY the letter (A, B, C, or D) inside <answer> tags.\n\n"
            "Example format:\n"
            "<thought>\nStep 1: [analyze options]\nStep 2: [select best]\n</thought>\n"
            "<answer>B</answer>"
        )
    else:
        # GSM8K / AQuA-RAT / default math
        system_prompt = (
            "You are a careful mathematical and logical reasoner. "
            "Solve the problem step by step inside <thought> tags. "
            "Write each step on a new line starting with 'Step N: '. "
            "Put ONLY the final answer (a number or letter) inside <answer> tags.\n\n"
            "Example format:\n"
            "<thought>\nStep 1: [your reasoning]\nStep 2: [your reasoning]\n</thought>\n"
            "<answer>42</answer>"
        )

    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Problem: {question_text}\n\nSolve step by step:"}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        prompt = f"{system_prompt}\n\nProblem: {question_text}\n\nSolve step by step:"

    return prompt


def _validate_paths_before_training(config):
    """
    Check all required paths exist before wasting time on imports and setup.
    Call this as the very first thing in train_rl().
    """
    required_paths = {
        "SFT checkpoint": config.get("sft_checkpoint_path", "checkpoints/sft/final"),
        "Difficulty map": config.get("difficulty_map_path", "difficulty_map.json"),
    }
    
    missing = []
    for name, path_val in required_paths.items():
        if not os.path.exists(path_val):
            missing.append(f"  {name}: {path_val}")
    
    if missing:
        print("\n" + "="*60)
        print("STARTUP ERROR: Required files not found.")
        print("These files must exist before training can start:")
        for m in missing:
            print(m)
        print("\nDid you run Phase 0 and Phase 1 in this session?")
        print("Kaggle sessions are stateless — outputs from previous sessions")
        print("are lost unless saved to a Kaggle Dataset.")
        print("="*60 + "\n")
        sys.exit(1)
    
    print("✅ All required paths validated. Training can begin.")

def train_rl(
    config_name: str,
    hardware: str,
    output_dir: str,
    resume: bool = False,
):
    """
    Main CRAFT RL training function.
    
    Training flow per step:
    1. Component C selects a question from the 40-70% difficulty zone
    2. Generate group_size responses for that question
    3. Component A scores each response (deterministic execution verifier)
    4. Component B (if step >= 100): compute step-level DPO loss
    5. compute_grpo_loss(): compute policy gradient loss with KL penalty
    6. Backward pass + optimizer step
    7. Log metrics, save checkpoint every 50 steps
    """
    setup_logger()
    logger.info("Initializing CRAFT Reinforcement Learning Loop (Phase 2)...")
    
    # Load config
    config = load_config(config_name, hardware)
    
    _validate_paths_before_training(config)
    
    # Detect hardware
    hw_info = detect_hardware()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Training device: {device}")
    
    # Hardware-specific parameter adjustments
    if hardware == "kaggle":
        GROUP_SIZE = 4        # Reduced from 8 for single T4 memory
        MAX_NEW_TOKENS = 256  # Sufficient for GSM8K, fast enough for T4
        TEMPERATURE = 0.8
        SAVE_EVERY = 50       # Save frequently — Kaggle sessions die
        TOTAL_STEPS = 500
    else:
        GROUP_SIZE = config.get("group_size", 8)
        MAX_NEW_TOKENS = config.get("max_new_tokens", 512)
        TEMPERATURE = config.get("temperature", 0.8)
        SAVE_EVERY = 100
        TOTAL_STEPS = config.get("total_steps", 500)
    
    logger.info(
        f"GRPO config: group_size={GROUP_SIZE}, max_new_tokens={MAX_NEW_TOKENS}, "
        f"total_steps={TOTAL_STEPS}"
    )
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(output_dir)
    
    # Load difficulty map from Phase 0
    difficulty_mapper = DifficultyMapper()
    difficulty_mapper.load_map("difficulty_map.json")
    
    # Initialize components
    curriculum = CurriculumEngine(
        difficulty_mapper=difficulty_mapper,
        initial_range=(0.4, 0.7),
    )
    kl_controller = KLController(
        initial_beta=config.get("kl_beta_initial", 0.04),
        target_kl=config.get("kl_target", 0.1),
    )
    reward_scorer = RewardScorer()

    # Initialize multi-dataset sampler (GSM8K + StrategyQA + MMLU)
    # This replaces curriculum-only sampling for steps after 200
    multi_sampler = MultiDatasetSampler(
        gsm8k_weight=0.50,
        strategyqa_weight=0.30,
        mmlu_weight=0.20,
        n_preload=500,
        difficulty_mapper=difficulty_mapper,
        curriculum=curriculum,
    )

    # Multi-dataset switch will be configured after checkpoint logic determines start_step
    
    # Paths
    sft_checkpoint = "checkpoints/sft/final"
    base_model_path = config.get("base_model_path", "microsoft/Phi-3-mini-4k-instruct")
    
    # Use local cache if available
    local_model_cache = "/kaggle/working/model_cache/phi3_mini"
    if os.path.exists(f"{local_model_cache}/config.json"):
        base_model_path = local_model_cache
        logger.info(f"Using cached base model at {local_model_cache}")
    
    # Load models
    policy_model, ref_model, tokenizer = load_model_and_tokenizer(
        base_model_path=base_model_path,
        sft_adapter_path=sft_checkpoint,
        config=config,
        device=device,
    )
    
    # Initialize optimizer — ONLY optimize LoRA parameters
    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    logger.info(f"Trainable parameter count: {sum(p.numel() for p in trainable_params):,}")
    
    if len(trainable_params) == 0:
        raise RuntimeError(
            "No trainable parameters found! "
            "LoRA layers must have requires_grad=True. "
            "Check load_model_and_tokenizer()."
        )
    
    optimizer = AdamW(
        trainable_params,
        lr=config.get("learning_rate", 5e-5),
        weight_decay=0.01,
        eps=1e-8,
    )
    
    # Resume from checkpoint if requested
    start_step = 1
    if resume:
        latest_meta, latest_path = checkpoint_mgr.get_latest()
        if latest_meta is not None:
            resume_step = latest_meta.get("step", 0)
            logger.info(f"Resuming from step {resume_step} (checkpoint: {latest_path})")
            start_step = resume_step + 1
            
            # ── LOAD RL CHECKPOINT WEIGHTS ──────────────────────────────────────────────
            logger.info("Loading policy model weights from RL checkpoint...")
            from safetensors.torch import load_file
            from peft import set_peft_model_state_dict
            
            adapter_path = os.path.join(latest_path, "adapter_model.safetensors")
            if os.path.exists(adapter_path):
                state_dict = load_file(adapter_path)
                set_peft_model_state_dict(policy_model, state_dict)
                logger.info("✅ RL checkpoint weights successfully loaded into policy model.")
            else:
                logger.error(f"Missing adapter weights at {adapter_path}")
                sys.exit(1)
            
            # ── RESTORE KL CONTROLLER STATE ─────────────────────────────────────────────
            # The model has already trained for `resume_step` steps, so:
            # 1. Skip the warmup — it already happened in the original run
            # 2. Restore the last known beta value if saved in metadata
            saved_beta = latest_meta.get("kl_beta", None)
            if saved_beta is not None:
                kl_controller.beta = saved_beta
                logger.info(f"Restored KL beta from checkpoint: {saved_beta:.4f}")
            # Fast-forward step count past warmup so the controller is active immediately
            kl_controller._step_count = max(resume_step, kl_controller.warmup_steps + 1)
            logger.info(
                f"KL controller warmup skipped (step_count set to {kl_controller._step_count}). "
                f"Controller is ACTIVE with beta={kl_controller.beta:.4f}"
            )
            
            # ── RESTORE CURRICULUM STATE ────────────────────────────────────────────────
            saved_curr_min = latest_meta.get("curriculum_min", None)
            saved_curr_max = latest_meta.get("curriculum_max", None)
            if saved_curr_min is not None and saved_curr_max is not None:
                curriculum.current_range = [saved_curr_min, saved_curr_max]
                logger.info(
                    f"Restored curriculum range from checkpoint: "
                    f"[{saved_curr_min:.4f}, {saved_curr_max:.4f}]"
                )
            
            # ── CHECKPOINT VALIDATION AFTER LOADING ─────────────────────────────────────
            logger.info(f"Validating checkpoint at {latest_path}...")
            
            # Run one forward pass to confirm the loaded model produces valid output
            try:
                policy_model.eval()
                with torch.no_grad():
                    # Use a dummy input to test forward pass
                    dummy_input = torch.ones(
                        (1, 10), dtype=torch.long, device=device
                    )
                    test_output = policy_model(dummy_input)
                    test_logits = test_output.logits
                    
                    # Check for NaN or inf in the output
                    if torch.isnan(test_logits).any() or torch.isinf(test_logits).any():
                        logger.error(
                            "CHECKPOINT CORRUPTED: Forward pass produced NaN/inf. "
                            "This checkpoint cannot be used safely. "
                            "Start training from scratch."
                        )
                        sys.exit(1)
                
                logger.info("✅ Checkpoint validated. Forward pass produced valid output.")
                policy_model.train()
                
            except Exception as checkpoint_error:
                logger.error(
                    f"CHECKPOINT VALIDATION FAILED: {checkpoint_error}. "
                    "Start training from scratch."
                )
                sys.exit(1)
        else:
            logger.info("No checkpoint found. Starting from step 1.")
            kl_controller.reset()
    else:
        kl_controller.reset()
    
    # After which step to switch to multi-dataset sampling
    # First 200 steps: GSM8K-only (curriculum) to build math foundation
    MULTI_DATASET_START_STEP = max(start_step, 200)
    logger.info(f"Multi-dataset sampling starts at step {MULTI_DATASET_START_STEP}")
    
    logger.info(
        f"GRPO RL Training active. Steps: {start_step} -> {TOTAL_STEPS} | "
        f"Group Size: {GROUP_SIZE} | Device: {device}"
    )
    
    # Initialize Component B (Contrastive Step DPO)
    trace_generator = TraceGenerator(policy_model, tokenizer, device)
    contrastive_builder = ContrastiveBuilder()
    dpo_trainer = StepDPOTrainer()
    COMPONENT_B_START_STEP = 100  # Delay B until policy is stable
    
    # ─── MAIN TRAINING LOOP ──────────────────────────────────────────────────
    for step in range(start_step, TOTAL_STEPS + 1):
        
        # ── QUESTION SAMPLING ────────────────────────────────────────────────
        # Implement a gradual dataset transition to prevent sudden reward landscape shifts
        if step < 200:
            # 100% GSM8K curriculum (build math foundation)
            question_data = _sample_from_curriculum(curriculum, step)
        elif step < 250:
            # 80% GSM8K, 20% StrategyQA
            if step % 5 == 0:
                question_data = multi_sampler._get_strategyqa() or _sample_from_curriculum(curriculum, step)
            else:
                question_data = _sample_from_curriculum(curriculum, step)
        elif step < 350:
            # 60% GSM8K, 30% StrategyQA, 10% MMLU
            mod = step % 10
            if mod < 6:
                question_data = _sample_from_curriculum(curriculum, step)
            elif mod < 9:
                question_data = multi_sampler._get_strategyqa() or _sample_from_curriculum(curriculum, step)
            else:
                question_data = multi_sampler._get_mmlu() or _sample_from_curriculum(curriculum, step)
        else:
            # Full blend: 50% GSM8K, 30% StrategyQA, 20% MMLU
            question_data = multi_sampler.sample_one() or _sample_from_curriculum(curriculum, step)

        if question_data is None:
            logger.warning(f"Step {step}: No questions available. Skipping.")
            continue
        
        # Format prompt
        prompt_text = format_prompt(question_data, tokenizer)
        
        # ── STAGE 1: GENERATION (no gradients) ──────────────────────────────
        prompt_ids, generated_ids, response_texts = generate_group_responses(
            policy_model=policy_model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            group_size=GROUP_SIZE,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            device=device,
        )
        
        # ── STAGE 2: REWARD SCORING (no gradients, pure Python) ─────────────
        rewards = []
        success_count = 0
        for response_text in response_texts:
            score, is_correct = reward_scorer.score_with_success(
                question_data, response_text
            )
            rewards.append(score)
            if is_correct:
                success_count += 1
        
        mean_reward = sum(rewards) / len(rewards)
        mean_success = success_count / len(rewards)
        
        # ── STAGE 3: GRPO + KL LOSS (with gradients) ────────────────────────
        policy_model.train()
        optimizer.zero_grad()
        
        total_loss, grpo_loss_val, kl_loss_val, advantages = compute_grpo_loss(
            policy_model=policy_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            generated_ids=generated_ids,
            rewards=rewards,
            kl_beta=kl_controller.get_beta(),
        )
        
        # ── STAGE 4: COMPONENT B (step-level DPO, after step 100) ───────────
        dpo_loss_val = 0.0
        component_b_active = False
        
        if step >= COMPONENT_B_START_STEP:
            torch.cuda.empty_cache()
            component_b_active = True
            try:
                contrastive_pairs = contrastive_builder.build_pairs(
                    question_data=question_data,
                    response_texts=response_texts,
                    rewards=rewards,
                )
                if contrastive_pairs:
                    dpo_loss = dpo_trainer.compute_loss(
                        policy_model=policy_model,
                        ref_model=ref_model,
                        tokenizer=tokenizer,
                        contrastive_pairs=contrastive_pairs,
                        beta=kl_controller.get_beta(),
                        device=device,
                    )
                    dpo_loss_val = dpo_loss.item()
                    # Add DPO loss to total (weighted lower than GRPO)
                    total_loss = total_loss + 0.1 * dpo_loss
                
                del contrastive_pairs
                torch.cuda.empty_cache()
            except RuntimeError as oom_error:
                if "out of memory" in str(oom_error).lower():
                    logger.warning(
                        f"Step {step}: Component B OOM. Skipping this step. "
                        f"Consider reducing n_traces further."
                    )
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                logger.warning(f"Step {step}: Component B failed: {e}. Continuing without DPO.")
        
        # ── BACKWARD PASS ────────────────────────────────────────────────────
        # Verify loss has grad before backward
        if not total_loss.requires_grad:
            logger.error(
                f"Step {step}: total_loss has no grad_fn! "
                "This means all advantages were 0 (all rewards identical). "
                "Skipping this step."
            )
            continue
        
        total_loss.backward()
        
        # GRADIENT CLIPPING: prevents any single batch from making catastrophically
        # large weight updates. max_norm=1.0 is the standard RL training value.
        # If you skip this, large rewards in a single batch can corrupt the model.
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        
        # Update weights
        optimizer.step()
        optimizer.zero_grad()
        
        # ── MEMORY CLEANUP ───────────────────────────────────────────────────────────
        # Delete large tensors explicitly to prevent GPU memory accumulation.
        # Without this, after 200-300 steps you'll hit CUDA OOM.
        del prompt_ids, generated_ids, response_texts, rewards
        del total_loss, advantages
        
        # ── COMPONENT C: Update curriculum ──────────────────────────────────
        curriculum.update_accuracy(
            question_id=question_data.get("id", ""),
            is_correct=(mean_success > 0)
        )
        
        # Safety check: if success rate is very low, step back the difficulty bounds
        # IMPORTANT: Only collapse if the question was from the core math curriculum (gsm8k).
        # We don't want to collapse math difficulty just because it struggled on a new StrategyQA logic question.
        if mean_success < 0.1 and step > 50 and question_data.get("dataset", "gsm8k").lower() == "gsm8k":
             curriculum.collapse_temporarily()
        
        # ── KL CONTROLLER: Adjust beta based on measured KL ─────────────────
        current_beta = kl_controller.step(kl_loss_val)
        
        # ── PER-STEP MEMORY CLEANUP ──────────────────────────────────────────────────
        # Python's autograd creates reference cycles that leak memory until gc runs.
        # We must run gc.collect() every step to prevent OOMs on 15GB T4 GPUs.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        if step % 50 == 0:
            logger.info(
                f"Memory status at step {step}: "
                f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
            )

        # ── LOGGING ─────────────────────────────────────────────────────────
        if step % 5 == 0:
            logger.info(
                f"[Step {step}] RL GRPO Metrics: "
                f"mean_reward={mean_reward:.4f}, "
                f"mean_success={mean_success:.4f}, "
                f"kl_divergence={kl_loss_val:.4f}, "
                f"kl_beta={current_beta:.4f}, "
                f"grpo_loss={grpo_loss_val:.4f}, "
                f"dpo_loss={dpo_loss_val:.4f}, "
                f"curriculum_min={curriculum.current_range[0]:.4f}, "
                f"curriculum_max={curriculum.current_range[1]:.4f}, "
                f"component_b_active={float(component_b_active):.1f}"
            )
        
        # ── CHECKPOINT ───────────────────────────────────────────────────────
        if step % SAVE_EVERY == 0:
            checkpoint_mgr.save(policy_model, tokenizer, step, metadata={
                "step": step,
                "kl_beta": kl_controller.get_beta(),
                "curriculum_min": curriculum.current_range[0],
                "curriculum_max": curriculum.current_range[1],
            })
            logger.info(f"Checkpoint saved at step {step}")
    
    # ── FINAL SAVE ───────────────────────────────────────────────────────────
    final_path = os.path.join(output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model saved to: {final_path}")
    
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CRAFT Phase 2 RL Training")
    parser.add_argument("--config", type=str, default="phi3_mini",
                        help="Config name (phi3_mini or qwen25_7b)")
    parser.add_argument("--hardware", type=str, default="kaggle",
                        help="Hardware profile (kaggle, local, vast)")
    parser.add_argument("--output", type=str, default="checkpoints/rl",
                        help="Output directory for checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()
    
    train_rl(
        config_name=args.config,
        hardware=args.hardware,
        output_dir=args.output,
        resume=args.resume,
    )
