import torch
from loguru import logger

class TraceGenerator:
    def __init__(self, model, tokenizer, device="cuda", n_traces=4, temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_traces = n_traces
        self.temperature = temperature

    def generate_traces(self, question: str) -> list:
        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        traces = []
        for _ in range(self.n_traces):
            try:
                with torch.no_grad():
                    out = self.model.generate(
                        **inputs, max_new_tokens=256, do_sample=True,
                        temperature=self.temperature, repetition_penalty=1.1,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                resp = self.tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                traces.append({"trace_text": resp})
            except Exception as e:
                logger.error(f"Trace gen error: {e}")
        return traces