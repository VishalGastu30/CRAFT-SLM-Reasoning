from typing import List, Dict, Any
from .step_parser import StepParser
from ..component_a.reward_scorer import RewardScorer

SYSTEM_PROMPT = (
    "You are a reasoning assistant. Solve the problem step by step. "
    "You MUST format your explanation as a sequence of steps starting with "
    "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: [value]' "
    "where [value] is the short final answer (number, yes/no, or letter)."
)

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
        parsed_traces = []
        for t in traces:
            text = t["trace_text"]
            steps = self.parser.parse_trace(text)
            
            step_rewards = []
            for s in steps:
                step_text = s["content"]
                from ..component_a.execution_verifier import verify_step_arithmetic
                correct_count, total_count = verify_step_arithmetic(step_text)
                if total_count == 0:
                    step_rewards.append(1.0)
                else:
                    step_rewards.append(1.0 if correct_count == total_count else 0.0)
                
            parsed_traces.append({
                "trace_text": text,
                "steps": steps,
                "rewards": step_rewards
            })

        contrastive_pairs = []
        
        for i in range(len(parsed_traces)):
            for j in range(i + 1, len(parsed_traces)):
                t1 = parsed_traces[i]
                t2 = parsed_traces[j]
                
                min_steps = min(len(t1["steps"]), len(t2["steps"]))
                
                prefix_context = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{question}\n<|assistant|>\n"
                
                for step_idx in range(min_steps):
                    s1 = t1["steps"][step_idx]["content"]
                    s2 = t2["steps"][step_idx]["content"]
                    
                    r1 = t1["rewards"][step_idx]
                    r2 = t2["rewards"][step_idx]
                    
                    if s1 != s2 and r1 != r2:
                        if r1 > r2:
                            chosen = s1
                            rejected = s2
                        else:
                            chosen = s2
                            rejected = s1
                            
                        dpo_prompt = prefix_context
                        for prev_idx in range(step_idx):
                            dpo_prompt += f"Step {prev_idx + 1}: {t1['steps'][prev_idx]['content']}\n"
                            
                        dpo_prompt += f"Step {step_idx + 1}: "
                        
                        contrastive_pairs.append({
                            "prompt": dpo_prompt,
                            "chosen": chosen,
                            "rejected": rejected
                        })
                        break
                        
        return contrastive_pairs

    def build_pairs(self, question_data: dict, response_texts: List[str], rewards: List[float]) -> List[Dict[str, Any]]:
        question = question_data.get("question", question_data.get("problem", ""))
        ground_truth = question_data.get("answer", "")
        dataset_name = question_data.get("dataset", "")
        
        traces = [{"trace_text": text} for text in response_texts]
        return self.build_step_pairs(
            question=question,
            traces=traces,
            ground_truth=ground_truth,
            dataset_name=dataset_name
        )