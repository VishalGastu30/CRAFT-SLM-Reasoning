import torch
import torch.nn.functional as F
from loguru import logger

class StepDPOTrainer:
    """
    Computes step-level Direct Preference Optimization (DPO) losses.
    """
    def __init__(self):
        logger.info("StepDPOTrainer initialized.")

    def compute_step_logps(self, model, tokenizer, prompt: str, target_step: str) -> torch.Tensor:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        full_text = prompt + target_step
        full_inputs = tokenizer(full_text, return_tensors="pt").to(device)
        prompt_inputs = tokenizer(prompt, return_tensors="pt").to(device)

        full_ids = full_inputs.input_ids
        prompt_len = prompt_inputs.input_ids.shape[1]

        with torch.set_grad_enabled(model.training):
            outputs = model(full_ids)
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = full_ids[..., 1:].contiguous()

        step_logits = shift_logits[:, prompt_len - 1:]
        step_labels = shift_labels[:, prompt_len - 1:]

        if step_labels.shape[1] == 0:
            return torch.tensor(0.0, device=device)

        log_probs = F.log_softmax(step_logits, dim=-1)
        per_token_logps = torch.gather(log_probs, dim=-1, index=step_labels.unsqueeze(-1)).squeeze(-1)

        return per_token_logps.mean(dim=-1)

    def compute_dpo_loss(self, model, ref_model, tokenizer, prompt: str, chosen: str, rejected: str, beta: float) -> torch.Tensor:
        policy_chosen_logps = self.compute_step_logps(model, tokenizer, prompt, chosen)
        policy_rejected_logps = self.compute_step_logps(model, tokenizer, prompt, rejected)
        
        if ref_model is not None:
            with torch.no_grad():
                ref_chosen_logps = self.compute_step_logps(ref_model, tokenizer, prompt, chosen)
                ref_rejected_logps = self.compute_step_logps(ref_model, tokenizer, prompt, rejected)
        else:
            ref_chosen_logps = policy_chosen_logps.detach() - 0.2
            ref_rejected_logps = policy_rejected_logps.detach() + 0.2
            
        policy_ratio = policy_chosen_logps - policy_rejected_logps
        ref_ratio = ref_chosen_logps - ref_rejected_logps
        
        loss = -F.logsigmoid(beta * (policy_ratio - ref_ratio))
        return loss.mean()

    def compute_loss(self, policy_model, ref_model, tokenizer, contrastive_pairs: list, beta: float, device: str = "cuda") -> torch.Tensor:
        import random
        if len(contrastive_pairs) > 1:
            contrastive_pairs = random.sample(contrastive_pairs, 1)
            
        dpo_losses = []
        for pair in contrastive_pairs:
            loss = self.compute_dpo_loss(
                model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                prompt=pair["prompt"],
                chosen=pair["chosen"],
                rejected=pair["rejected"],
                beta=beta
            )
            dpo_losses.append(loss)
            
        if not dpo_losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        return torch.stack(dpo_losses).mean()