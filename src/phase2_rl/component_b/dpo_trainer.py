import torch
import torch.nn.functional as F
from loguru import logger

class StepDPOTrainer:
    """
    Computes step-level Direct Preference Optimization (DPO) losses.
    Supervises the model to prefer mathematically/logically sound reasoning steps
    over flawed steps under identical previous contexts.
    """
    def __init__(self, beta=0.1):
        self.beta = beta
        logger.info(f"StepDPOTrainer initialized with beta={self.beta}")

    def compute_step_logps(self, model, tokenizer, prompt: str, target_step: str) -> torch.Tensor:
        """
        Computes the log probabilities of the tokens in the target_step
        conditioned on the prompt prefix.
        """
        device = model.device
        
        # Tokenize the full sequence and prompt only sequence
        full_text = prompt + target_step
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        full_ids = full_inputs.input_ids
        prompt_len = prompt_inputs.input_ids.shape[1]
        
        # Forward pass to get logits
        with torch.set_grad_enabled(model.training):
            outputs = model(full_ids)
            logits = outputs.logits
            
        # Shift logits and labels for causal LM loss computation
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_ids[..., 1:].contiguous()
        
        # We only calculate log probs for the tokens corresponding to the target_step
        # target_step tokens start after the prompt_len - 1 in shift_logits/labels
        step_logits = shift_logits[:, prompt_len - 1:]
        step_labels = shift_labels[:, prompt_len - 1:]
        
        # Calculate log-softmax over vocabulary
        log_probs = F.log_softmax(step_logits, dim=-1)
        
        # Gather log probabilities for the actual labels
        per_token_logps = torch.gather(log_probs, dim=-1, index=step_labels.unsqueeze(-1)).squeeze(-1)
        
        # Sum over token sequence
        return per_token_logps.sum(dim=-1)

    def compute_dpo_loss(self, model, ref_model, tokenizer, prompt: str, chosen: str, rejected: str) -> torch.Tensor:
        """
        Computes step-level DPO loss:
        L_DPO = -log_sigmoid(beta * ( (logps_policy_chosen - logps_ref_chosen) - (logps_policy_rejected - logps_ref_rejected) ))
        """
        # Policy log probabilities
        policy_chosen_logps = self.compute_step_logps(model, tokenizer, prompt, chosen)
        policy_rejected_logps = self.compute_step_logps(model, tokenizer, prompt, rejected)
        
        # Reference model log probabilities
        if ref_model is not None:
            with torch.no_grad():
                ref_chosen_logps = self.compute_step_logps(ref_model, tokenizer, prompt, chosen)
                ref_rejected_logps = self.compute_step_logps(ref_model, tokenizer, prompt, rejected)
        else:
            # Fallback mock reference log probabilities for CPU tests
            ref_chosen_logps = policy_chosen_logps.detach() - 0.2
            ref_rejected_logps = policy_rejected_logps.detach() + 0.2
            
        # DPO equation calculations
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps
        
        loss = -F.logsigmoid(self.beta * (policy_ratio - ref_ratio))
        return loss.mean()
