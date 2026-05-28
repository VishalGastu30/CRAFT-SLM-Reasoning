import os
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from loguru import logger

from src.utils.logger import setup_logger, log_metrics
from src.utils.hardware_detector import detect_hardware
from src.utils.checkpoint_manager import CheckpointManager
from src.config import load_config
from src.phase2_rl.reward_combiner import RewardCombiner
from src.phase2_rl.component_c.curriculum_engine import CurriculumEngine
from src.phase2_rl.component_c.kl_controller import KLController
from src.phase2_rl.component_b.trace_generator import TraceGenerator
from src.phase2_rl.component_b.contrastive_builder import ContrastiveBuilder
from src.phase2_rl.component_b.dpo_trainer import StepDPOTrainer

def compute_seq_logps(model, tokenizer, prompt: str, response: str) -> torch.Tensor:
    """
    Computes the log probabilities of the tokens in the response
    conditioned on the prompt prefix.
    """
    device = model.device
    full_text = prompt + response
    full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    full_ids = full_inputs.input_ids
    prompt_len = prompt_inputs.input_ids.shape[1]
    
    # Forward pass
    with torch.set_grad_enabled(model.training):
        outputs = model(full_ids)
        logits = outputs.logits
        
    # Shift logits and labels
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = full_ids[..., 1:].contiguous()
    
    # Slice response part
    resp_logits = shift_logits[:, prompt_len - 1:]
    resp_labels = shift_labels[:, prompt_len - 1:]
    
    # Calculate log-softmax
    log_probs = torch.nn.functional.log_softmax(resp_logits, dim=-1)
    
    # Gather actual label log probabilities
    per_token_logps = torch.gather(log_probs, dim=-1, index=resp_labels.unsqueeze(-1)).squeeze(-1)
    
    return per_token_logps.sum(dim=-1)

def train_rl(config_name="phi3_mini", hardware_name="kaggle", output_dir="checkpoints/rl", resume=False):
    """Main orchestrator for CRAFT Phase 2 Reinforcement Learning Loop."""
    setup_logger()
    logger.info("Initializing CRAFT Reinforcement Learning Loop (Phase 2)...")
    
    # 1. Load config
    raw_config = load_config(config_name, hardware_name)
    hw_profile = detect_hardware()
    
    rl_cfg = raw_config.get("rl", {})
    curr_cfg = raw_config.get("curriculum", {})
    
    # Initialize tools and components
    manager = CheckpointManager(output_dir)
    curriculum = CurriculumEngine(
        difficulty_map_path="difficulty_map.json",
        initial_range=curr_cfg.get("initial_range", (0.4, 0.7)),
        window_size=curr_cfg.get("window_size", 50),
        stability_threshold=curr_cfg.get("stability_threshold", 0.7)
    )
    kl_controller = KLController(
        initial_beta=rl_cfg.get("kl_beta", 0.04),
        kl_target=rl_cfg.get("kl_target", 0.1)
    )
    combiner = RewardCombiner()
    
    # Component B: Step-level DPO components
    contrastive_builder = ContrastiveBuilder()
    dpo_trainer = StepDPOTrainer(beta=rl_cfg.get("dpo_beta", 0.1))
    
    # 2. Load model and tokenizer (resuming from Phase 1 SFT final checkpoint)
    sft_model_path = "checkpoints/sft/final"
    if not os.path.exists(sft_model_path):
        logger.warning(f"SFT Warmup checkpoint not found at {sft_model_path}! Falling back to base model: {raw_config['model']['name']}")
        sft_model_path = raw_config["model"]["name"]
        
    logger.info(f"Loading SFT model/tokenizer from: {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Reference and Active models (for KL penalty)
    base_model_name = raw_config["model"]["name"]
    
    if device == "cuda":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Policy / Active model
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
        
        # Reference model (frozen copy to compute reference KL)
        ref_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        ref_model.eval()
    else:
        logger.warning("No CUDA device detected. Running RL loop on CPU (mock mode).")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map="cpu", trust_remote_code=True)
        ref_model = None

    # Load the Phase 1 SFT Adapter correctly so we can continue training it
    from peft import PeftModel
    if sft_model_path != base_model_name:
        logger.info(f"Applying SFT adapter from {sft_model_path}")
        model = PeftModel.from_pretrained(model, sft_model_path, is_trainable=True)
        if ref_model is not None:
            ref_model = PeftModel.from_pretrained(ref_model, sft_model_path, is_trainable=False)
    else:
        # If no SFT checkpoint, initialize a fresh LoRA
        lora_config = LoraConfig(
            r=raw_config.get("sft", {}).get("lora_rank", 64),
            lora_alpha=raw_config.get("sft", {}).get("lora_alpha", 128),
            target_modules=raw_config.get("sft", {}).get("target_modules", ["q_proj", "v_proj"]),
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    
    # 3. Resume training if requested
    start_step = 1
    if resume:
        metadata, checkpoint_path = manager.get_latest()
        if checkpoint_path:
            logger.info(f"Resuming RL training from latest checkpoint: {checkpoint_path}")
            # Load lora adapter weights
            model.load_adapter(checkpoint_path, "default")
            start_step = metadata.get("global_step", start_step) + 1
            
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(rl_cfg.get("learning_rate", 5e-6)))
    
    # 4. GRPO Reinforcement Learning Loop
    total_steps = rl_cfg.get("total_steps", 500)
    group_size = rl_cfg.get("grpo_group_size", 8)
    temperature = rl_cfg.get("temperature", 0.8)
    dpo_activation_step = rl_cfg.get("dpo_step_activation", 100)
    
    logger.info(f"GRPO RL Training active. Steps: {start_step} -> {total_steps} | Group Size: {group_size} | Device: {device}")
    
    for step in range(start_step, total_steps + 1):
        # A. Sample question from Curriculum Engine
        question_data = curriculum.sample_question()
        if not question_data:
            logger.warning("No questions available in Curriculum pool. Skipping step!")
            continue
            
        question_text = question_data["question"]
        ground_truth = question_data["answer"]
        dataset_name = question_data["dataset"]
        
        # Format user prompt
        prompt = f"<|user|>\n{question_text}\n<|assistant|>\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        # B. Sample group_size paths (GRPO)
        group_responses = []
        group_rewards = []
        group_successes = []
        
        if ref_model is not None:
            # Policy model generation
            # GRPO generates group_size answers for a single question
            for _ in range(group_size):
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.eos_token_id
                )
                response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                group_responses.append(response)
                
                # C. Compute combined rewards (Component A: Verifiers + format checks)
                scoring_metrics = combiner.combine_rewards(
                    question=question_text,
                    response_text=response,
                    ground_truth=ground_truth,
                    dataset_name=dataset_name
                )
                
                reward = scoring_metrics["reward_combined"]
                group_rewards.append(reward)
                group_successes.append(scoring_metrics["final_correct"])
        else:
            # CPU Mock execution
            for i in range(group_size):
                # Simulated group reward distribution
                sim_reward = float(np.random.choice([0.0, 0.4, 0.7, 1.0]))
                group_rewards.append(sim_reward)
                group_successes.append(1.0 if sim_reward > 0.5 else 0.0)
                group_responses.append(f"Step 1: Mock step {i}.\nFinal Answer: mock")
                
        # D. GRPO Reward Normalization (Advantage calculation)
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards) if len(group_rewards) > 1 else 1.0
        std_reward = max(1e-5, std_reward)
        
        group_advantages = [(r - mean_reward) / std_reward for r in group_rewards]
        
        # E. Update Curriculum tracker using average final correctness rate
        mean_success = np.mean(group_successes)
        curriculum.update_accuracy(mean_success)
        
        # F. Policy Gradient Weight Update
        total_loss = None
        grpo_loss_val = 0.0
        dpo_loss_val = 0.0
        mean_kl = 0.0
        
        # Check if Component B is activated (Step DPO starts at step 100)
        component_b_active = step >= dpo_activation_step
        
        if ref_model is not None:
            # Active model is in training mode
            model.train()
            optimizer.zero_grad()
            
            # Compute GRPO loss
            grpo_losses = []
            kl_divs = []
            
            for idx, response in enumerate(group_responses):
                advantage = group_advantages[idx]
                
                # Compute log probabilities under current policy
                policy_logps = compute_seq_logps(model, tokenizer, prompt, response)
                
                # Compute log probabilities under reference model
                with torch.no_grad():
                    ref_logps = compute_seq_logps(ref_model, tokenizer, prompt, response)
                    
                # Approximate KL divergence
                kl_div = (policy_logps - ref_logps).mean()
                kl_divs.append(kl_div.item())
                
                # Actor loss: -advantage * policy_logps + kl_beta * kl_div
                actor_loss = -advantage * policy_logps + kl_beta * kl_div
                grpo_losses.append(actor_loss)
                
            grpo_loss = torch.stack(grpo_losses).mean()
            grpo_loss_val = grpo_loss.item()
            total_loss = grpo_loss
            
            # Compute KL metrics for controller
            mean_kl = float(np.mean(kl_divs))
            kl_beta = kl_controller.step(mean_kl)
            
            # Check and compute Component B (Step DPO)
            if component_b_active:
                traces = [{"trace_text": r} for r in group_responses]
                # Build DPO-ready positive/negative step pairs
                contrastive_pairs = contrastive_builder.build_step_pairs(
                    question=question_text,
                    traces=traces,
                    ground_truth=ground_truth,
                    dataset_name=dataset_name
                )
                
                if contrastive_pairs:
                    dpo_losses = []
                    for pair in contrastive_pairs:
                        dpo_loss = dpo_trainer.compute_dpo_loss(
                            model=model,
                            ref_model=ref_model,
                            tokenizer=tokenizer,
                            prompt=pair["prompt"],
                            chosen=pair["chosen"],
                            rejected=pair["rejected"]
                        )
                        dpo_losses.append(dpo_loss)
                    
                    dpo_loss_mean = torch.stack(dpo_losses).mean()
                    dpo_loss_val = dpo_loss_mean.item()
                    # Add DPO loss to total loss
                    total_loss = total_loss + dpo_loss_mean
                    
            # Backward pass & Optimizer step
            if total_loss is not None:
                total_loss.backward()
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
        else:
            # CPU/Mock mode execution
            simulated_kl = float(np.random.uniform(0.02, 0.15))
            kl_beta = kl_controller.step(simulated_kl)
            mean_kl = simulated_kl
            grpo_loss_val = float(np.random.uniform(0.1, 0.5))
            
            # In CPU mock mode, we still want to run the full DPO builder and trainer 
            # to verify correctness of the step-level DPO code pathways!
            if component_b_active:
                traces = [{"trace_text": r} for r in group_responses]
                contrastive_pairs = contrastive_builder.build_step_pairs(
                    question=question_text,
                    traces=traces,
                    ground_truth=ground_truth,
                    dataset_name=dataset_name
                )
                
                if contrastive_pairs:
                    # Let's run a single DPO computation to test it end-to-end
                    pair = contrastive_pairs[0]
                    # We run it with gradient tracking disabled to keep it fast
                    with torch.no_grad():
                        mock_dpo_loss = dpo_trainer.compute_dpo_loss(
                            model=model,
                            ref_model=None,
                            tokenizer=tokenizer,
                            prompt=pair["prompt"],
                            chosen=pair["chosen"],
                            rejected=pair["rejected"]
                        )
                        dpo_loss_val = float(mock_dpo_loss.item())
                        
        # 5. Logging and checkpointing
        if step % rl_cfg.get("logging_steps", 5) == 0:
            metrics_dict = {
                "mean_reward": mean_reward,
                "mean_success": mean_success,
                "kl_divergence": mean_kl,
                "kl_beta": kl_beta,
                "curriculum_min": curriculum.min_difficulty,
                "curriculum_max": curriculum.max_difficulty,
                "component_b_active": 1.0 if component_b_active else 0.0,
                "grpo_loss": grpo_loss_val,
                "dpo_loss": dpo_loss_val
            }
            log_metrics(step, metrics_dict, phase="rl", component="GRPO")
            
        if step % rl_cfg.get("save_steps", 50) == 0:
            metadata = {
                "global_step": step,
                "mean_reward": mean_reward,
                "mean_success": mean_success,
                "curriculum_max": curriculum.max_difficulty
            }
            # PEFT model save
            if ref_model is not None:
                manager.save(model, tokenizer, step, metadata)
            else:
                logger.info(f"[CPU Mock Save] Checkpoint-{step} recorded.")
            manager.cleanup(keep_n=3)
            
    # Save final model
    final_model_dir = os.path.join(output_dir, "final")
    logger.info(f"Saving final CRAFT RL model weights to {final_model_dir}...")
    if ref_model is not None:
        model.save_pretrained(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)
    logger.info("CRAFT RL Loop training successfully completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: CRAFT RL Loop")
    parser.add_argument("--config", type=str, default="phi3_mini", help="Model config name")
    parser.add_argument("--hardware", type=str, default="kaggle", help="Hardware profile name")
    parser.add_argument("--output", type=str, default="checkpoints/rl", help="Output directory")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint if exists")
    args = parser.parse_args()

    train_rl(
        config_name=args.config,
        hardware_name=args.hardware,
        output_dir=args.output,
        resume=args.resume
    )
