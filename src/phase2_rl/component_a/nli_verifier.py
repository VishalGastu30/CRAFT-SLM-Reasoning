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
        
        # We load DeBERTa lazily to avoid heavy start-up costs in dry-runs
        
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
            logger.error(f"Failed to load DeBERTa NLI model: {e}. Falling back to rule-based mock NLI scoring.")
            self.is_loaded = False

    def predict_entailment(self, premise: str, hypothesis: str) -> float:
        """
        Predicts entailment score (0.0 to 1.0) from premise to hypothesis.
        Labels for DeBERTa cross-encoder/nli-deberta-v3-small:
        0: contradiction, 1: entailment, 2: neutral
        """
        self._load_model()
        
        if not self.is_loaded:
            # Fallback mock logic if DeBERTa failed to load:
            # Contradiction keywords
            contradict_words = {"never", "cannot", "impossible", "not", "false"}
            p_words = set(premise.lower().split())
            h_words = set(hypothesis.lower().split())
            
            # Simple heuristic
            overlap = len(p_words.intersection(h_words)) / max(1, len(h_words))
            if any(w in h_words for w in contradict_words) and not any(w in p_words for w in contradict_words):
                return 0.2  # contradiction
            return 0.5 + 0.5 * overlap  # soft entailment
            
        try:
            features = self.tokenizer(premise, hypothesis, padding=True, truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**features).logits
                
            # Convert to probabilities
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # probs order for cross-encoder/nli-deberta-v3-small: [contradiction, entailment, neutral]
            contradiction_prob = probs[0]
            entailment_prob = probs[1]
            neutral_prob = probs[2]
            
            # Entailment is positive, contradiction is strongly negative, neutral is slightly positive
            score = entailment_prob * 1.0 + neutral_prob * 0.5 - contradiction_prob * 0.5
            return float(max(0.0, min(1.0, score)))
        except Exception as e:
            logger.error(f"Error predicting entailment: {e}")
            return 0.5

    def score_logical_chain(self, steps: list) -> float:
        """
        Scores transition flow between successive steps: Step 1 -> Step 2 -> Step 3.
        Calculates mean transition score. Returns 1.0 if only 1 step exists.
        """
        if len(steps) <= 1:
            return 1.0
            
        transition_scores = []
        for i in range(len(steps) - 1):
            premise = steps[i]
            hypothesis = steps[i+1]
            score = self.predict_entailment(premise, hypothesis)
            transition_scores.append(score)
            
        return sum(transition_scores) / len(transition_scores)
