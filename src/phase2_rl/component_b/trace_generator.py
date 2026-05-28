import torch
from loguru import logger

class TraceGenerator:
    """
    Generates multiple reasoning traces (paths) for a single question
    under temperature-based sampling, enabling diversity for contrastive analysis.
    """
    def __init__(self, model, tokenizer, n_traces=6, temperature=0.8):
        self.model = model
        self.tokenizer = tokenizer
        self.n_traces = n_traces
        self.temperature = temperature

    def generate_traces(self, question: str) -> list:
        """
        Generates diverse reasoning traces for a given question.
        Returns:
            list of dict: [{"trace_text": str, "tokens": list}]
        """
        if self.model is None or self.tokenizer is None:
            # Fallback mock generations if no active model loaded
            mock_traces = []
            for i in range(self.n_traces):
                is_correct = (i % 2 == 0)
                ans = "36" if is_correct else "32"
                text = f"Step 1: 15/100 * 240 = {ans}.\nFinal Answer: {ans}"
                mock_traces.append({
                    "trace_text": text,
                    "is_mock": True
                })
            return mock_traces

        prompt = f"<|user|>\n{question}\n<|assistant|>\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        
        traces = []
        for _ in range(self.n_traces):
            try:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                traces.append({
                    "trace_text": response,
                    "is_mock": False
                })
            except Exception as e:
                logger.error(f"Error generating trace: {e}")
                
        return traces
