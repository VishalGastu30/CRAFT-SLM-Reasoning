from typing import List, Dict, Any
from .step_parser import StepParser
from src.phase2_rl.component_a.reward_scorer import RewardScorer

class ContrastiveBuilder:
    """
    Builds contrastive preference pairs at the step level.
    Identifies step-by-step deviations where a model chose a wrong
    reasoning branch, and pairs it with a successful branch under the same prefix context.
    """
    def __init__(self):
        self.parser = StepParser()
        self.scorer = RewardScorer()

    def build_step_pairs(self, question: str, traces: List[Dict[str, Any]], ground_truth: str, dataset_name: str = "") -> List[Dict[str, Any]]:
        """
        Compares multiple traces to build DPO-ready positive/negative step pairs.
        Each trace is expected to be a dict: {"trace_text": str}
        Returns:
            list of dict: [{"prompt": str, "chosen_step": str, "rejected_step": str}]
        """
        parsed_traces = []
        for t in traces:
            text = t["trace_text"]
            steps = self.parser.parse_trace(text)
            
            # Score each step individually in this trace
            step_rewards = []
            for s in steps:
                step_text = s["content"]
                # Evaluate step math correctness
                from src.phase2_rl.component_a.execution_verifier import verify_step_arithmetic
                correct_count, total_count = verify_step_arithmetic(step_text)
                if total_count == 0:
                    step_rewards.append(1.0) # Assume non-math text steps are correct
                else:
                    step_rewards.append(1.0 if correct_count == total_count else 0.0)
                
            parsed_traces.append({
                "trace_text": text,
                "steps": steps,
                "rewards": step_rewards
            })

        contrastive_pairs = []
        
        # Cross-compare traces to find step-level bifurcations
        for i in range(len(parsed_traces)):
            for j in range(i + 1, len(parsed_traces)):
                t1 = parsed_traces[i]
                t2 = parsed_traces[j]
                
                # Check up to the minimum step length between the two traces
                min_steps = min(len(t1["steps"]), len(t2["steps"]))
                
                prefix_context = f"<|user|>\n{question}\n<|assistant|>\n"
                
                for step_idx in range(min_steps):
                    s1 = t1["steps"][step_idx]["content"]
                    s2 = t2["steps"][step_idx]["content"]
                    
                    r1 = t1["rewards"][step_idx]
                    r2 = t2["rewards"][step_idx]
                    
                    # If steps differ and one is correct while the other is incorrect
                    if s1 != s2 and r1 != r2:
                        if r1 > r2:
                            chosen = s1
                            rejected = s2
                        else:
                            chosen = s2
                            rejected = s1
                            
                        # Build prompt context of preceding steps
                        # This forms the DPO prompt: prompt + previous steps
                        dpo_prompt = prefix_context
                        for prev_idx in range(step_idx):
                            dpo_prompt += f"Step {prev_idx + 1}: {t1['steps'][prev_idx]['content']}\n"
                            
                        dpo_prompt += f"Step {step_idx + 1}: "
                        
                        contrastive_pairs.append({
                            "prompt": dpo_prompt,
                            "chosen": chosen,
                            "rejected": rejected
                        })
                        
                        # Break to avoid adding multiple contrastive pairs from the same trace bifurcation point
                        break
                        
        return contrastive_pairs

    def build_pairs(self, question_data: dict, response_texts: List[str], rewards: List[float]) -> List[Dict[str, Any]]:
        """
        Wrapper compatibility method matching the new craft_rl_loop interface.
        """
        question = question_data.get("question", question_data.get("problem", ""))
        ground_truth = question_data.get("answer", "")
        dataset_name = question_data.get("dataset", "")
        
        # Build traces list
        traces = [{"trace_text": text} for text in response_texts]
        
        return self.build_step_pairs(
            question=question,
            traces=traces,
            ground_truth=ground_truth,
            dataset_name=dataset_name
        )
