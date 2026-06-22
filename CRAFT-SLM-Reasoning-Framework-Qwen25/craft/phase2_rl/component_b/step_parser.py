import re
from typing import List, Dict, Any

class StepParser:
    """
    Parses reasoning traces to extract step-by-step reasoning structures.
    """
    def __init__(self, step_pattern=r"(?:^|\n)(Step \d+)\s*[:\-]\s*"):
        self.step_pattern = step_pattern

    def parse_trace(self, trace_text: str) -> List[Dict[str, Any]]:
        if not trace_text:
            return []
            
        matches = list(re.finditer(self.step_pattern, trace_text))
        
        if not matches:
            lines = [l.strip() for l in trace_text.split("\n") if len(l.strip()) > 3]
            steps = []
            char_cursor = 0
            for idx, line in enumerate(lines):
                if "final answer" in line.lower():
                    continue
                start_idx = trace_text.find(line, char_cursor)
                end_idx = start_idx + len(line)
                steps.append({
                    "step_num": idx + 1,
                    "label": f"Step {idx + 1}",
                    "content": line,
                    "start": start_idx,
                    "end": end_idx
                })
                char_cursor = end_idx
            return steps

        steps = []
        for i in range(len(matches)):
            current_match = matches[i]
            label = current_match.group(1)
            step_num = int(re.findall(r"\d+", label)[0])
            start_content = current_match.end()
            
            if i < len(matches) - 1:
                end_content = matches[i+1].start()
            else:
                end_content = len(trace_text)
                
            content_block = trace_text[start_content:end_content].strip()
            final_ans_idx = content_block.lower().find("final answer")
            if final_ans_idx != -1:
                content_block = content_block[:final_ans_idx].strip()
                end_content = start_content + final_ans_idx
                
            steps.append({
                "step_num": step_num,
                "label": label,
                "content": content_block,
                "start": current_match.start(),
                "end": end_content
            })
            
        return steps