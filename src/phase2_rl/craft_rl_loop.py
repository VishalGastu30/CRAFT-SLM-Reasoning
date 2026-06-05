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
from peft import PeftModel
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.logger import setup_logger
from src.utils.checkpoint_manager import CheckpointManager
from src.utils.hardware_detector import detect_hardware
from src.config import load_config
from src.phase0_probe.difficulty_mapper import DifficultyMapper
from src.phase2_rl.component_a.reward_scorer import RewardScorer
from src.phase2_rl.component_b.contrastive_builder import ContrastiveBuilder
from src.phase2_rl.component_b.dpo_trainer import StepDPOTrainer
from src.phase2_rl.component_c.curriculum_engine import CurriculumEngine
from src.phase2_rl.component_c.kl_controller import KLController
from src.phase2_rl.multi_dataset_sampler import MultiDatasetSampler


def _validate_paths_before_training(config):
    required = {
        "SFT checkpoint": config.get("sft_checkpoint_path", "checkpoints/sft/final"),
        "Difficulty map": config.get("difficulty_map_path", "data/difficulty_map.json"),
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not os.path.exists(path)]
    if missing:
        print("\n" + "="*60)
        print("STARTUP ERROR: Required files missing:")
        for m in missing: print(m)
        print("="*60 + "\n")
        sys.exit(1)
    print("✅ All required paths validated.")


def load_model_and_tokenizer(base_model_path, sft_adapter_path, config, device):
    logger.info(f"Loading tokenizer from {sft_adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter_path, trust_remote_code=False, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Loading base model from {base_model_path}")
    flash_available = importlib.util.find_spec("flash_attn") is not None
    attn_impl = "flash_attention_2" if flash_available else "eager"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype=torch.bfloat16,
                                     bnb_4bit_use_double_quant=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, quantization_config=bnb_config, device_map="auto",
        attn_implementation=attn_impl, torch_dtype=torch.bfloat16, trust_remote_code=False
    )
    logger.info(f"Applying SFT adapter from {sft_adapter_path}")
    policy_model = PeftModel.from_pretrained(base_model, sft_adapter_path, is_trainable=True)
    policy_model.gradient_checkpointing_enable()
    lora_trainable = sum(1 for n, p in policy_model.named_parameters() if 'lora' in n and p.requires_grad)
    logger.info(f"LoRA trainable params: {lora_trainable}")
    if lora_trainable == 0:
        for n, p in policy_model.named_parameters():
            if 'lora' in n:
                p.requires_grad_(True)
        logger.warning("Manually enabled LoRA gradients.")

    # Reference model (frozen)
    logger.info("Loading frozen reference model...")
    ref_base = AutoModelForCausalLM.from_pretrained(
        base_model_path, quantization_config=bnb_config, device_map="auto",
        attn_implementation=attn_impl, torch_dtype=torch.bfloat16, trust_remote_code=False
    )
    ref_model = PeftModel.from_pretrained(ref_base, sft_adapter_path, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    return policy_model, ref_model, tokenizer


def compute_grpo_loss(policy_model, ref_model, tokenizer, prompt_ids, generated_ids, rewards, kl_beta):
    group_size = len(rewards)
    prompt_len = prompt_ids.shape[1]

    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    reward_mean = rewards_tensor.mean()
    reward_std = rewards_tensor.std()
    centered = rewards_tensor - reward_mean
    if reward_std > 0.01:
        advantages = centered / (reward_std + 1e-8)
    else:
        advantages = centered

    policy_model.train()
    outputs = policy_model(generated_ids)
    logits = outputs.logits

    shift_logits = logits[:, prompt_len-1:-1, :]
    shift_labels = generated_ids[:, prompt_len:]

    log_probs_vocab = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs_vocab.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != tokenizer.pad_token_id).float()
    response_lengths = mask.sum(dim=1).clamp(min=1)
    seq_log_probs = (token_log_probs * mask).sum(dim=1) / response_lengths

    grpo_loss = -(advantages.to(seq_log_probs.device) * seq_log_probs).mean()

    with torch.no_grad():
        ref_outputs = ref_model(generated_ids)
        ref_logits = ref_outputs.logits
        ref_shift_logits = ref_logits[:, prompt_len-1:-1, :]
        ref_log_probs_vocab = F.log_softmax(ref_shift_logits, dim=-1)
        ref_token_log_probs = ref_log_probs_vocab.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        ref_seq_log_probs = (ref_token_log_probs * mask).sum(dim=1) / response_lengths

    kl_per_seq = seq_log_probs - ref_seq_log_probs.detach()
    kl_loss = torch.clamp(kl_per_seq.mean(), min=0.0)

    total_loss = grpo_loss + kl_beta * kl_loss
    return total_loss, grpo_loss.item(), kl_loss.item(), advantages.tolist()


def generate_group_responses(policy_model, tokenizer, prompt_text, group_size, max_new_tokens, temperature, device):
    inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=512).to(device)
    prompt_ids = inputs["input_ids"]
    batched_ids = prompt_ids.repeat(group_size, 1)
    batched_mask = inputs["attention_mask"].repeat(group_size, 1)

    policy_model.eval()
    with torch.no_grad():
        generated = policy_model.generate(
            batched_ids, attention_mask=batched_mask,
            max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=False
        )
    prompt_len = prompt_ids.shape[1]
    responses = []
    for i in range(group_size):
        resp_ids = generated[i][prompt_len:]
        responses.append(tokenizer.decode(resp_ids, skip_special_tokens=True))
    return prompt_ids, generated, responses


def format_prompt(question_data, tokenizer):
    question_text = question_data.get("question", question_data.get("problem", ""))
    dataset = question_data.get("dataset", "gsm8k")
    
    if dataset == "strategyqa":
        system = (
            "You are a reasoning assistant. Answer the yes/no question step by step. "
            "You MUST format your explanation as a sequence of steps starting with "
            "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: yes' or 'Final Answer: no'."
        )
    elif dataset == "mmlu":
        system = (
            "You are a reasoning assistant. Answer the multiple-choice question step by step. "
            "You MUST format your explanation as a sequence of steps starting with "
            "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: A' (or B, C, D)."
        )
    elif dataset == "aqua":
        system = (
            "You are a reasoning assistant. Answer the multiple-choice question step by step. "
            "You MUST format your explanation as a sequence of steps starting with "
            "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: A' (or B, C, D, E)."
        )
    else:
        system = (
            "You are a reasoning assistant. Solve the math problem step by step. "
            "You MUST format your explanation as a sequence of steps starting with "
            "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: [value]' "
            "where [value] is the short final answer."
        )
    
    try:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"Problem: {question_text}\n\nSolve step by step:"}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        return f"{system}\n\nProblem: {question_text}\n\nSolve step by step:"


def train_rl(config_name, hardware, output_dir, resume=False):
    setup_logger()
    logger.info("Initializing CRAFT RL Training")
    config = load_config(config_name, hardware)
    _validate_paths_before_training(config)

    hw_info = detect_hardware()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if hardware == "kaggle":
        GROUP_SIZE = 4
        MAX_NEW_TOKENS = 256
        TEMPERATURE = 0.9   # increased for diversity
        SAVE_EVERY = 50
        TOTAL_STEPS = 500
    else:
        GROUP_SIZE = config.get("group_size", 8)
        MAX_NEW_TOKENS = config.get("max_new_tokens", 512)
        TEMPERATURE = config.get("temperature", 0.8)
        SAVE_EVERY = 100
        TOTAL_STEPS = config.get("total_steps", 500)

    logger.info(f"GRPO config: group_size={GROUP_SIZE}, max_new_tokens={MAX_NEW_TOKENS}, total_steps={TOTAL_STEPS}")
    checkpoint_mgr = CheckpointManager(output_dir)

    difficulty_mapper = DifficultyMapper()
    difficulty_mapper.load_map("data/difficulty_map.json")
    curriculum = CurriculumEngine(difficulty_mapper=difficulty_mapper, initial_range=(0.4, 0.7))
    kl_controller = KLController(initial_beta=0.04, target_kl=0.1, warmup_steps=20, adjustment_factor=0.02, beta_max=0.2)
    reward_scorer = RewardScorer()

    # Multi-dataset sampler used from step 1
    multi_sampler = MultiDatasetSampler(gsm8k_weight=0.4, strategyqa_weight=0.25, mmlu_weight=0.20, aqua_weight=0.15,
                                        n_preload=500, difficulty_mapper=difficulty_mapper, curriculum=curriculum)

    sft_checkpoint = "checkpoints/sft/final"
    base_model_path = config.get("base_model_path", "microsoft/Phi-3-mini-4k-instruct")
    local_cache = "/kaggle/working/model_cache/phi3_mini"
    if os.path.exists(f"{local_cache}/config.json"):
        base_model_path = local_cache
        logger.info(f"Using cached base model at {local_cache}")

    policy_model, ref_model, tokenizer = load_model_and_tokenizer(
        base_model_path, sft_checkpoint, config, device
    )

    trainable_params = [p for p in policy_model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = AdamW(trainable_params, lr=config.get("learning_rate", 5e-5), weight_decay=0.01, eps=1e-8)

    start_step = 1
    if resume:
        latest = checkpoint_mgr.get_latest()
        if latest:
            logger.info(f"Resuming from step {latest['step']}")
            start_step = latest["step"] + 1
            checkpoint_mgr.load_optimizer_state(optimizer, latest["step"])
    else:
        kl_controller.reset()

    contrastive_builder = ContrastiveBuilder()
    dpo_trainer = StepDPOTrainer()
    COMPONENT_B_START_STEP = 50   # start DPO earlier

    logger.info(f"Training from step {start_step} to {TOTAL_STEPS}")
    step = start_step
    
    # tracking for manual expansion if stuck
    low_accuracy_counter = 0
    
    while step <= TOTAL_STEPS:
        # Always sample from multi-dataset (fresh questions)
        question_data = multi_sampler.sample_one()
        if question_data is None:
            logger.warning(f"Step {step}: No question available. Skipping.")
            step += 1
            continue

        prompt_text = format_prompt(question_data, tokenizer)

        prompt_ids, generated_ids, responses = generate_group_responses(
            policy_model, tokenizer, prompt_text, GROUP_SIZE, MAX_NEW_TOKENS, TEMPERATURE, device
        )

        rewards = []
        successes = []
        for resp in responses:
            score, correct = reward_scorer.score_with_success(question_data, resp)
            rewards.append(score)
            successes.append(correct)
        mean_reward = sum(rewards) / len(rewards)
        mean_success = sum(successes) / len(successes)

        policy_model.train()
        optimizer.zero_grad()
        total_loss, grpo_loss_val, kl_loss_val, adv_list = compute_grpo_loss(
            policy_model, ref_model, tokenizer, prompt_ids, generated_ids, rewards, kl_controller.get_beta()
        )

        dpo_loss_val = 0.0
        if step >= COMPONENT_B_START_STEP:
            try:
                pairs = contrastive_builder.build_pairs(question_data, responses, rewards)
                if pairs:
                    import random
                    pairs = random.sample(pairs, min(2, len(pairs)))
                    dpo_loss = dpo_trainer.compute_loss(policy_model, ref_model, tokenizer, pairs,
                                                         beta=kl_controller.get_beta(), device=device)
                    dpo_loss_val = dpo_loss.item()
                    total_loss = total_loss + 0.1 * dpo_loss
            except Exception as e:
                logger.warning(f"Component B error: {e}")

        if total_loss.requires_grad:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
        else:
            logger.warning(f"Step {step}: loss has no grad, skipping update")

        # Update curriculum (only expand, no collapse)
        curriculum.update_accuracy(question_id=question_data.get("id", ""), is_correct=(mean_success > 0))
        
        # Manual expansion safeguard: if accuracy < 0.5 for 100 steps, expand manually
        if mean_success < 0.5:
            low_accuracy_counter += 1
        else:
            low_accuracy_counter = 0
        
        if low_accuracy_counter >= 100:
            # manually expand range
            old_min, old_max = curriculum.current_range
            new_min = max(0.4, old_min - 0.05)
            new_max = min(0.85, old_max + 0.05)
            curriculum.current_range = [new_min, new_max]
            logger.info(f"Manual expansion due to low accuracy: [{old_min:.2f},{old_max:.2f}] -> [{new_min:.2f},{new_max:.2f}]")
            low_accuracy_counter = 0

        current_beta = kl_controller.step(kl_loss_val)

        if step % 5 == 0:
            logger.info(
                f"[Step {step}] mean_reward={mean_reward:.4f}, mean_success={mean_success:.4f}, "
                f"kl={kl_loss_val:.4f}, beta={current_beta:.4f}, grpo={grpo_loss_val:.4f}, "
                f"dpo={dpo_loss_val:.4f}, curr=[{curriculum.current_range[0]:.2f},{curriculum.current_range[1]:.2f}]"
            )

        if step % SAVE_EVERY == 0:
            checkpoint_mgr.save(policy_model, tokenizer, step, metadata={
                "step": step,
                "kl_beta": current_beta,
                "curriculum_min": curriculum.current_range[0],
                "curriculum_max": curriculum.current_range[1]
            })
            logger.info(f"Checkpoint saved at step {step}")

        del prompt_ids, generated_ids, responses, rewards
        del total_loss, grpo_loss_val, kl_loss_val

        if step % 50 == 0:
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Step {step} memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

        step += 1

    final_path = os.path.join(output_dir, "final")
    policy_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    logger.info(f"Training complete. Final model: {final_path}")
    return final_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="phi3_mini")
    parser.add_argument("--hardware", default="kaggle")
    parser.add_argument("--output", default="checkpoints/rl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train_rl(args.config, args.hardware, args.output, args.resume)