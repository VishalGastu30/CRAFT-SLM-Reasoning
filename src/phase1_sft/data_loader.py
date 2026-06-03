import re
from datasets import load_dataset, concatenate_datasets
from loguru import logger

# Generic system prompt – works for all tasks
SYSTEM_PROMPT = (
    "You are a reasoning assistant. Solve the problem step by step. "
    "You MUST format your explanation as a sequence of steps starting with "
    "'Step 1:', 'Step 2:', etc. You MUST end your response with 'Final Answer: [value]' "
    "where [value] is the short final answer (number, yes/no, or letter)."
)

def format_gsm8k_target(answer_text: str) -> str:
    """Reformat GSM8K answer (with '#### X') into steps + 'Final Answer: X'."""
    if "####" not in answer_text:
        return f"Step 1: Solve the problem.\nFinal Answer: {answer_text.strip()}"

    parts = answer_text.split("####")
    steps_text = parts[0].strip()
    final_ans = parts[1].strip()

    sentences = re.split(r'\.\s+', steps_text)
    formatted_steps = []
    step_num = 1
    current_step_text = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_step_text:
            current_step_text += ". " + sent
        else:
            current_step_text = sent

        if len(current_step_text.split()) > 15 or sent == sentences[-1]:
            formatted_steps.append(f"Step {step_num}: {current_step_text}.")
            step_num += 1
            current_step_text = ""

    steps_str = "\n".join(formatted_steps)
    return f"{steps_str}\nFinal Answer: {final_ans}"

def format_aqua_target(rationale: str, options: str, correct_letter: str) -> str:
    """Reformat AQuA‑RAT (multiple choice) into steps + 'Final Answer: letter'."""
    options_clean = options.strip("[]").replace("'", "").replace('"', "")

    sentences = re.split(r'\.\s+', rationale.strip())
    formatted_steps = []
    step_num = 1
    current_step_text = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if current_step_text:
            current_step_text += ". " + sent
        else:
            current_step_text = sent

        if len(current_step_text.split()) > 15 or sent == sentences[-1]:
            formatted_steps.append(f"Step {step_num}: {current_step_text}.")
            step_num += 1
            current_step_text = ""

    steps_str = "\n".join(formatted_steps)
    return (
        f"{steps_str}\n"
        f"Evaluating options: {options_clean}.\n"
        f"Final Answer: {correct_letter}"
    )

def format_strategyqa_target(question: str, answer: str) -> str:
    """Reformat StrategyQA (yes/no) into steps + 'Final Answer: yes/no'."""
    # Create a simple step‑by‑step reasoning template
    # (We rely on the model to learn proper reasoning from examples)
    steps = (
        "Step 1: Read the question carefully.\n"
        "Step 2: Identify the key facts needed to answer.\n"
        "Step 3: Determine whether the statement is true or false.\n"
        f"Final Answer: {answer.lower()}"
    )
    return steps

def format_mmlu_target(question: str, choices: list, correct_letter: str) -> str:
    """Reformat MMLU (multiple choice) into steps + 'Final Answer: letter'."""
    choice_lines = "\n".join([f"{chr(65+i)}. {c}" for i, c in enumerate(choices)])
    steps = (
        "Step 1: Read the question and the options.\n"
        f"Step 2: Options:\n{choice_lines}\n"
        "Step 3: Evaluate each option.\n"
        f"Final Answer: {correct_letter}"
    )
    return steps

def create_training_dataset(
    tokenizer,
    gsm8k_fraction=0.40,
    aqua_fraction=0.20,
    strategyqa_fraction=0.20,
    mmlu_fraction=0.20,
    seed=42
):
    """
    Loads, formats, and combines GSM8K, AQuA‑RAT, StrategyQA, and MMLU.
    All are converted to the same step‑by‑step + 'Final Answer:' format.
    """
    logger.info("Loading training datasets for SFT warm‑up (4 datasets)...")
    all_datasets = []

    # ---- GSM8K ----
    try:
        gsm8k = load_dataset("gsm8k", "main", split="train")
        num_gsm = int(len(gsm8k) * gsm8k_fraction)
        gsm_sample = gsm8k.shuffle(seed=seed).select(range(min(num_gsm, len(gsm8k))))

        def process_gsm(example):
            target = format_gsm8k_target(example["answer"])
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {"text": prompt + target + tokenizer.eos_token, "dataset": "gsm8k"}

        gsm_dataset = gsm_sample.map(process_gsm, remove_columns=gsm8k.column_names)
        all_datasets.append(gsm_dataset)
        logger.info(f"Loaded {len(gsm_dataset)} GSM8K records")
    except Exception as e:
        logger.error(f"GSM8K load failed: {e}")

    # ---- AQuA‑RAT ----
    try:
        aqua = load_dataset("aqua_rat", split="train")
        num_aqua = int(len(aqua) * aqua_fraction)
        aqua_sample = aqua.shuffle(seed=seed).select(range(min(num_aqua, len(aqua))))

        def process_aqua(example):
            target = format_aqua_target(
                example["rationale"],
                str(example["options"]),
                example["correct"]
            )
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {"text": prompt + target + tokenizer.eos_token, "dataset": "aqua"}

        aqua_dataset = aqua_sample.map(process_aqua, remove_columns=aqua.column_names)
        all_datasets.append(aqua_dataset)
        logger.info(f"Loaded {len(aqua_dataset)} AQuA‑RAT records")
    except Exception as e:
        logger.error(f"AQuA‑RAT load failed: {e}")

    # ---- StrategyQA ----
    try:
        # Use the alternative source that is more stable
        sqa = load_dataset("ChilleD/StrategyQA", split="train")
        num_sqa = int(len(sqa) * strategyqa_fraction)
        sqa_sample = sqa.shuffle(seed=seed).select(range(min(num_sqa, len(sqa))))

        def process_strategyqa(example):
            answer = "yes" if example["answer"] else "no"
            target = format_strategyqa_target(example["question"], answer)
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {"text": prompt + target + tokenizer.eos_token, "dataset": "strategyqa"}

        sqa_dataset = sqa_sample.map(process_strategyqa, remove_columns=sqa.column_names)
        all_datasets.append(sqa_dataset)
        logger.info(f"Loaded {len(sqa_dataset)} StrategyQA records")
    except Exception as e:
        logger.error(f"StrategyQA load failed: {e}")

    # ---- MMLU ----
    try:
        mmlu = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        num_mmlu = int(len(mmlu) * mmlu_fraction)
        # Sample evenly across subjects (take every N-th)
        total = len(mmlu)
        step = max(1, total // num_mmlu)
        indices = list(range(0, total, step))[:num_mmlu]
        mmlu_sample = mmlu.select(indices)

        def process_mmlu(example):
            choices = example["choices"]
            correct_letter = ["A", "B", "C", "D"][example["answer"]]
            target = format_mmlu_target(example["question"], choices, correct_letter)
            prompt = f"<|system|>\n{SYSTEM_PROMPT}\n<|user|>\n{example['question']}\n<|assistant|>\n"
            return {"text": prompt + target + tokenizer.eos_token, "dataset": "mmlu"}

        mmlu_dataset = mmlu_sample.map(process_mmlu, remove_columns=mmlu.column_names)
        all_datasets.append(mmlu_dataset)
        logger.info(f"Loaded {len(mmlu_dataset)} MMLU records")
    except Exception as e:
        logger.error(f"MMLU load failed: {e}")

    if not all_datasets:
        raise ValueError("No datasets could be loaded. Check internet and HuggingFace access.")

    combined = concatenate_datasets(all_datasets)
    combined = combined.shuffle(seed=seed)
    logger.info(f"Total SFT training records: {len(combined)}")
    return combined