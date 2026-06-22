import torch
from loguru import logger

class NLIVerifier:
    """
    NLI Verifier class to evaluate logical consistency and progression
    in non-mathematical reasoning chains (e.g., StrategyQA).
    Uses a small DeBERTa model to predict step transition entailment.
    """
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-small", device=None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.is_loaded = False
        
    def _load_model(self):
        if self.is_loaded:
            return
            
        try:
            logger.info(f"Loading local NLI cross-encoder model: {self.model_name}...")
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info("NLI model successfully loaded.")
        except Exception as e:
            logger.error(f"Failed to load DeBERTa NLI model: {e}. Falling back to rule-based.")
            self.is_loaded = False

    def predict_entailment(self, premise: str, hypothesis: str) -> float:
        self._load_model()
        
        if not self.is_loaded:
            contradict_words = {"never", "cannot", "impossible", "not", "false"}
            p_words = set(premise.lower().split())
            h_words = set(hypothesis.lower().split())
            overlap = len(p_words.intersection(h_words)) / max(1, len(h_words))
            if any(w in h_words for w in contradict_words) and not any(w in p_words for w in contradict_words):
                return 0.2
            return 0.5 + 0.5 * overlap
            
        try:
            features = self.tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**features).logits
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            contradiction_prob = probs[0]
            entailment_prob = probs[1]
            neutral_prob = probs[2]
            
            score = entailment_prob * 1.0 + neutral_prob * 0.5 - contradiction_prob * 0.5
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Error predicting entailment: {e}")
            return 0.5

    def score_logical_chain(self, steps: list) -> float:
        if len(steps) <= 1:
            return 1.0
            
        transition_scores = []
        for i in range(len(steps) - 1):
            score = self.predict_entailment(steps[i], steps[i+1])
            transition_scores.append(score)
            
        return sum(transition_scores) / len(transition_scores)