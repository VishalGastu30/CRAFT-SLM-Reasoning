import os
import time
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from loguru import logger

app = FastAPI(title="CRAFT Inference Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Configuration =====
MOCK_MODE = False  # Set to True to test without model
MODEL_PATH = os.path.join("craft_output", "CRAFT_Q4_K_M.gguf")  # Unified name
model_instance = None
model_arch = None  # will store 'phi3' or 'qwen2'

# ===== Pydantic Models =====
class InferenceRequest(BaseModel):
    question: str
    temperature: float = 0.8
    max_tokens: int = 512

class InferenceResponse(BaseModel):
    question: str
    response: str
    steps: List[str]
    final_answer: str
    latency_ms: float
    is_mock: bool

# ===== Helper Functions =====
def extract_final_answer(response_text: str) -> str:
    if not response_text:
        return ""
    final_match = re.search(r'Final Answer:\s*([^\n]+)', response_text, re.IGNORECASE)
    if final_match:
        return final_match.group(1).strip()
    gsm_match = re.search(r'####\s*([^\s]+)', response_text)
    if gsm_match:
        return gsm_match.group(1)
    return ""

def parse_response(response_text: str):
    first_final = re.search(r'Final Answer:\s*([^\n]+)', response_text, re.IGNORECASE)
    if first_final:
        final_answer = first_final.group(1).strip()
        main_text = response_text[:first_final.start()].strip()
    else:
        final_answer = ""
        main_text = response_text

    if not final_answer:
        gsm = re.search(r'####\s*([^\s]+)', response_text)
        if gsm:
            final_answer = gsm.group(1)

    steps = []
    markers = list(re.finditer(r'(?:^|\n)\s*Step\s*(\d+)\s*[:\-\.]\s*', main_text, re.IGNORECASE))
    if markers:
        for idx, m in enumerate(markers):
            start = m.end()
            end = markers[idx+1].start() if idx+1 < len(markers) else len(main_text)
            content = main_text[start:end].strip()
            if content:
                steps.append(content)
    else:
        if main_text.strip():
            steps = [main_text.strip()]

    return {"steps": steps, "final_answer": final_answer}

def load_gguf_model():
    global model_instance, model_arch
    if model_instance is not None:
        return model_instance

    if not os.path.exists(MODEL_PATH):
        error_msg = f"Model file not found at: {MODEL_PATH}\n"
        error_msg += f"Current working directory: {os.getcwd()}\n"
        if os.path.exists("craft_output"):
            error_msg += f"Files in craft_output: {os.listdir('craft_output')}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        from llama_cpp import Llama
        logger.info(f"Loading model from {MODEL_PATH} ...")
        model_instance = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=0,
            verbose=False
        )

        # Detect architecture from metadata
        try:
            arch = model_instance.metadata.get("general.architecture", "").lower()
            if "phi" in arch:
                model_arch = "phi3"
            elif "qwen" in arch:
                model_arch = "qwen2"
            else:
                model_arch = "unknown"
            logger.info(f"Detected model architecture: {model_arch}")
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")
            model_arch = "unknown"

        logger.info("GGUF model loaded successfully.")
        return model_instance

    except ImportError:
        logger.error("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

def get_mock_response(question: str):
    time.sleep(0.5)
    q = question.lower()
    if "penguin" in q:
        steps = [
            "Penguins are aquatic flightless birds adapted to cold marine environments.",
            "The Sahara desert is a hyper-arid desert with extreme heat exceeding 40°C.",
            "Penguins lack mechanisms to dissipate extreme heat.",
            "Without access to cold water, a penguin cannot survive."
        ]
        response = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)]) + "\nFinal Answer: no"
        final = "no"
    elif "15%" in q or "240" in q:
        steps = [
            "15% means 15 per hundred.",
            "15% of 240 = (15/100) * 240.",
            "15 * 240 = 3600.",
            "3600 / 100 = 36."
        ]
        response = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)]) + "\nFinal Answer: 36"
        final = "36"
    else:
        steps = [
            "Parse the user's reasoning request carefully.",
            "Construct logical chains to formulate a reasoning path.",
            "Verify the correctness of the logical assertions.",
            "Present the final answer with confidence."
        ]
        response = "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(steps)]) + "\nFinal Answer: yes"
        final = "yes"
    return response, steps, final

# ===== Endpoints =====
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "mock_mode": MOCK_MODE,
        "model_path": MODEL_PATH,
        "model_exists": os.path.exists(MODEL_PATH),
        "architecture": model_arch or "unknown"
    }

@app.post("/infer", response_model=InferenceResponse)
async def infer(request: InferenceRequest):
    start_time = time.time()

    if MOCK_MODE:
        response_text, steps, final_ans = get_mock_response(request.question)
        latency = (time.time() - start_time) * 1000
        return InferenceResponse(
            question=request.question,
            response=response_text,
            steps=steps,
            final_answer=final_ans,
            latency_ms=round(latency, 2),
            is_mock=True
        )

    try:
        llm = load_gguf_model()
        if llm is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        # Build prompt based on architecture
        system_prompt = (
            "You are CRAFT, a reasoning assistant. Solve problems step by step. "
            "Format steps as 'Step 1:', 'Step 2:', etc. End with 'Final Answer: [value]'."
        )

        if model_arch == "phi3":
            prompt = (
                f"<|system|>\n{system_prompt}\n<|end|>\n"
                f"<|user|>\n{request.question}\n<|end|>\n"
                f"<|assistant|>\n"
            )
            stop_tokens = ["<|end|>", "<|user|>", "<|system|>"]
        elif model_arch == "qwen2":
            prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{request.question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            stop_tokens = ["<|im_end|>", "<|im_start|>"]
        else:
            # Fallback: generic format (try Phi-3 style, but might not work)
            prompt = (
                f"<|system|>\n{system_prompt}\n<|end|>\n"
                f"<|user|>\n{request.question}\n<|end|>\n"
                f"<|assistant|>\n"
            )
            stop_tokens = ["<|end|>", "<|user|>", "<|system|>"]
            logger.warning("Unknown architecture, using Phi-3 style prompt.")

        output = llm(
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=stop_tokens,
            echo=False
        )

        response_text = output["choices"][0]["text"].strip()
        parsed = parse_response(response_text)
        steps = parsed["steps"]
        final_ans = parsed["final_answer"] if parsed["final_answer"] else extract_final_answer(response_text)

        if not steps and response_text:
            steps = [response_text]

        latency = (time.time() - start_time) * 1000

        return InferenceResponse(
            question=request.question,
            response=response_text,
            steps=steps,
            final_answer=final_ans,
            latency_ms=round(latency, 2),
            is_mock=False
        )

    except FileNotFoundError as fnf:
        logger.error(f"Model file missing: {fnf}")
        raise HTTPException(status_code=503, detail=f"Model not found: {str(fnf)}")
    except ImportError as imp:
        logger.error(f"Missing dependency: {imp}")
        raise HTTPException(status_code=500, detail="llama-cpp-python not installed. Run: pip install llama-cpp-python")
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")