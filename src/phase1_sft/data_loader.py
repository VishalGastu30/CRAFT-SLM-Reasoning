import re
from datasets import load_dataset, concatenate_datasets
from loguru import logger

# System prompt enforcing structured reasoning steps
SYSTEM_PROMPT = (
    "You are a reasoning assistant. Solve the math problem step by step. "
    "You MUST format your explanation as a sequence of steps starting with "
    "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: [value]' "
    "where [value] is the short final answer."
)

def format_gsm8k_target(answer_text: str) -> str:
    """
    Parses GSM8K standard answers containing steps and "#### X" final answers
    and reformats them into:
    Step 1: ...
    Step 2: ...
    Final Answer: X
    """
    if "####" not in answer_text:
        return f"Step 1: Solve the problem.\nFinal Answer: {answer_text.strip()}"
        
    parts = answer_text.split("####")
    steps_text = parts[0].strip()
    final_ans = parts[1].strip()
    
    # Split text into sentences or lines to construct pseudo-steps
    sentences = re.split(r'\.\s+', steps_text)
    formatted_steps = []
    
    step_num = 1
    current_step_text = ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        # Group sentences to avoid making too many 1-sentence steps
        if len(current_step_text) > 0:
            current_step_text += ". " + sent
        else:
            current_step_text = sent
            
        if len(current_step_text.split()) > 15 or sent == sentences[-1]:
            formatted_steps.append(f"Step {step_num}: {current_step_text}.")
            step_num += 1
            current_step_text = ""
            
    # Combine into steps and add final answer
    steps_str = "\n".join(formatted_steps)
    return f"{steps_str}\nFinal Answer: {final_ans}"

def format_aqua_target(rationale: str, options: str, correct_letter: str) -> str:
    """
    Converts AQuA-RAT rationales, options, and letter answers into structured steps.
    """
    # Clean up option list
    options_clean = options.strip("[]").replace("'", "").replace('"', "")
    
    # Reformat rationale into steps
    sentences = re.split(r'\.\s+', rationale.strip())
    formatted_steps = []
    step_num = 1
    current_step_text = ""
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current_step_text) > 0:
            current_step_text += ". " + sent
        else:
            current_step_text = sent
            
        if len(current_step_text.split()) > 15 or sent == sentences[-1]:
            formatted_steps.append(f"Step {step_num}: {current_step_text}.")
            step_num += 1
            current_step_text = ""
            
    steps_str = "\n".join(formatted_steps)
    
    # Append choice options clarification if necessary, then the final letter
    return (
        f"{steps_str}\n"
        f"Evaluating options: {options_clean}.\n"
        f"Final Answer: {correct_letter}"
    )

def create_training_dataset(tokenizer, gsm_fraction=0.6, aqua_fraction=0.4, seed=42):
    """
    Loads, cleans, formats, and combines GSM8K and AQuA-RAT.
    """
    logger.info("Loading training datasets for SFT warm-up...")
    
    # 1. Load and format GSM8K
    gsm8k_dataset = None
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        # Sample based on fraction
        num_gsm = int(len(gsm8k) * gsm_fraction)
        gsm8k_sample = gsm8k.shuffle(seed=seed).select(range(min(num_gsm, len(gsm8k))))
        
        def process_gsm(example):
            formatted_target = format_gsm8k_target(example["answer"])
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {
                "text": prompt + formatted_target + tokenizer.eos_token,
                "dataset": "gsm8k"
            }
            
        gsm8k_dataset = gsm8k_sample.map(process_gsm, remove_columns=gsm8k.column_names)
        logger.info(f"Loaded and formatted {len(gsm8k_dataset)} GSM8K records.")
    except Exception as e:
        logger.error(f"Failed to load/format GSM8K: {e}")
        
    # 2. Load and format AQuA-RAT
    aqua_dataset = None
    try:
        # AQuA-RAT split is train
        aqua = load_dataset("aqua_rat", split="train")
        num_aqua = int(len(aqua) * aqua_fraction)
        aqua_sample = aqua.shuffle(seed=seed).select(range(min(num_aqua, len(aqua))))
        
        def process_aqua(example):
            formatted_target = format_aqua_target(
                example["rationale"], 
                str(example["options"]), 
                example["correct"]
            )
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {
                "text": prompt + formatted_target + tokenizer.eos_token,
                "dataset": "aqua"
            }
            
        aqua_dataset = aqua_sample.map(process_aqua, remove_columns=aqua.column_names)
        logger.info(f"Loaded and formatted {len(aqua_dataset)} AQuA-RAT records.")
    except Exception as e:
        logger.error(f"Failed to load/format AQuA-RAT: {e}")
        
    # Combine datasets
    datasets_to_combine = []
    if gsm8k_dataset:
        datasets_to_combine.append(gsm8k_dataset)
    if aqua_dataset:
        datasets_to_combine.append(aqua_dataset)
        
    if not datasets_to_combine:
        raise ValueError("Both GSM8K and AQuA-RAT datasets failed to load.")
        
    combined_dataset = concatenate_datasets(datasets_to_combine)
    # Final shuffle
    combined_dataset = combined_dataset.shuffle(seed=seed)
    
    logger.info(f"Combined dataset contains {len(combined_dataset)} training records.")
    return combined_dataset
