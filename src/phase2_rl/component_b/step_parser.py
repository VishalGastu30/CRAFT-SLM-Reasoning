import re
from typing import List, Dict, Any

class StepParser:
    """
    Parses reasoning traces to extract step-by-step reasoning structures.
    Tracks step numbering, text content, and character index boundaries.
    """
    def __init__(self, step_pattern=r"(?:^|\n)(Step \d+)\s*[:\-]\s*"):
        self.step_pattern = step_pattern

    def parse_trace(self, trace_text: str) -> List[Dict[str, Any]]:
        """
        Parses the trace text and extracts each step block.
        Returns:
            list of dict: [{"step_num": int, "label": str, "content": str, "start": int, "end": int}]
        """
        if not trace_text:
            return []
            
        # Find all matches of "Step N:"
        matches = list(re.finditer(self.step_pattern, trace_text))
        
        if not matches:
            # Fallback if no step pattern is found (e.g. raw lines)
            lines = [l.strip() for l in trace_text.split("\n") if len(l.strip()) > 3]
            steps = []
            char_cursor = 0
            for idx, line in enumerate(lines):
                # Ignore lines starting with "Final Answer" as a step
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
            label = current_match.group(1) # e.g. "Step 1"
            
            # Step number integer extraction
            step_num = int(re.findall(r"\d+", label)[0])
            
            # Find boundaries of the step's body content
            start_content = current_match.end()
            
            # The end of this step is the start of the next match, or the end of trace
            if i < len(matches) - 1:
                end_content = matches[i+1].start()
            else:
                end_content = len(trace_text)
                
            # If the final step has "Final Answer" trailing, exclude it from the step content
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
