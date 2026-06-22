import os
import shutil
import tarfile
import argparse
import platform
from loguru import logger


class DeliveryPackager:
    """
    Packages the final CRAFT application into a single consolidated,
    deployable delivery bundle. Works on Windows and Linux/macOS.
    """

    def __init__(self, delivery_dir="craft_delivery"):
        self.delivery_dir = delivery_dir
        self.is_windows = platform.system().lower() == "windows"
        os.makedirs(self.delivery_dir, exist_ok=True)

    def package_all(self, gguf_path):
        """
        Build the complete deployment package.
        """
        logger.info("Building deployment package...")

        # Copy inference server and UI
        self._copy_source_files()

        # Copy the quantized model
        self._copy_model(gguf_path)

        # Create helper files
        self._create_launch_scripts()
        self._create_requirements()
        self._create_readme()

        logger.info("Packaging completed successfully.")

        # Create compressed archive
        archive_name = self.delivery_dir + ".tar.gz"

        try:
            with tarfile.open(archive_name, "w:gz") as tar:
                tar.add(self.delivery_dir,
                        arcname=os.path.basename(self.delivery_dir))

            logger.info(f"Created archive: {archive_name}")

        except Exception as e:
            logger.warning(f"Could not create archive: {e}")

        return self.delivery_dir

    def _copy_source_files(self):
        """Copy inference server and UI files from source to delivery bundle."""
        # Create required directories
        dirs = ["src/inference", "src/ui", "craft_output", "docs"]
        for d in dirs:
            os.makedirs(os.path.join(self.delivery_dir, d), exist_ok=True)

        # ---- Copy inference server ----
        inference_sources = [
            "craft/inference_server.py",
            "src/inference/inference_server.py",
            "inference_server.py"
        ]
        inference_copied = False
        for src_path in inference_sources:
            if os.path.exists(src_path):
                dst_path = os.path.join(self.delivery_dir, "src/inference/inference_server.py")
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied inference server from {src_path}")
                inference_copied = True
                break

        if not inference_copied:
            logger.warning("Inference server not found. Creating placeholder.")
            placeholder = '''import os
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
'''
            with open(os.path.join(self.delivery_dir, "src/inference/inference_server.py"), "w") as f:
                f.write(placeholder)
                logger.info("Created inference_server.py")

        # ---- Create Enhanced UI Files ----
        self._create_ui_files()

    def _create_ui_files(self):
        """Create enhanced chatbot UI files."""
        ui_dir = os.path.join(self.delivery_dir, "src/ui")
        os.makedirs(ui_dir, exist_ok=True)

        # ---- index.html ----
        index_html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CRAFT — Reasoning Assistant</title>
    <link rel="stylesheet" href="styles.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
</head>
<body>
    <div class="app-container">
        <!-- Sidebar -->
        <aside class="sidebar">
            <div class="logo">
                <span class="logo-icon">⚡</span>
                <span class="logo-text">CRAFT</span>
            </div>
            <div class="model-info">
                <span class="status-dot active"></span>
                <span class="model-name">Phi-3-Mini</span>
                <span class="model-badge">GGUF</span>
            </div>
            <div class="sidebar-actions">
                <button class="btn-new-chat" onclick="resetChat()">
                    <span>+</span> New Chat
                </button>
            </div>
            <div class="sidebar-footer">
                <div class="stats">
                    <span id="latency-display">0ms</span>
                    <span id="mode-display">Mock Mode</span>
                </div>
            </div>
        </aside>

        <!-- Main Chat Area -->
        <main class="chat-container">
            <header class="chat-header">
                <h1>CRAFT Reasoning Assistant</h1>
                <div class="header-actions">
                    <label class="toggle-label">
                        <span>Mock Mode</span>
                        <input type="checkbox" id="mock-toggle" checked>
                        <span class="toggle-slider"></span>
                    </label>
                    <select id="temp-select">
                        <option value="0.2">0.2 (Focused)</option>
                        <option value="0.5">0.5 (Balanced)</option>
                        <option value="0.8" selected>0.8 (Creative)</option>
                        <option value="1.2">1.2 (Exploratory)</option>
                    </select>
                </div>
            </header>

            <!-- Messages Area -->
            <div class="messages-area" id="messages-area">
                <div class="welcome-message">
                    <div class="welcome-icon">🤖</div>
                    <h2>Welcome to CRAFT</h2>
                    <p>Ask me any reasoning or math question, and I'll provide step-by-step thinking.</p>
                    <div class="suggestions">
                        <button class="suggestion-chip" onclick="sendQuestion('What is 15% of 240?')">15% of 240</button>
                        <button class="suggestion-chip" onclick="sendQuestion('Would a penguin survive in the Sahara desert?')">Penguin in Sahara</button>
                        <button class="suggestion-chip" onclick="sendQuestion('If a train travels at 60 mph for 2.5 hours, how far does it go?')">Train distance</button>
                        <button class="suggestion-chip" onclick="sendQuestion('Is a tomato a fruit or a vegetable?')">Tomato classification</button>
                    </div>
                </div>
            </div>

            <!-- Input Area -->
            <div class="input-area">
                <div class="input-wrapper">
                    <textarea id="question-input" rows="1" placeholder="Ask a reasoning question..." onkeydown="handleKeyDown(event)"></textarea>
                    <button id="send-btn" onclick="sendMessage()">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                        </svg>
                    </button>
                </div>
                <div class="input-footer">
                    <span>Press Enter to send, Shift+Enter for new line</span>
                    <span id="char-count">0</span>
                </div>
            </div>
        </main>
    </div>

    <!-- Loading Overlay -->
    <div id="loading-overlay" class="loading-overlay hidden">
        <div class="loading-spinner"></div>
        <span>Thinking...</span>
    </div>

    <script src="app.js"></script>
</body>
</html>'''

        # ---- styles.css ----
        styles_css = '''/* ===== CRAFT Chat UI - Modern Dark Theme ===== */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #14141e;
    --bg-card: #1a1a2e;
    --bg-input: #1e1e32;
    --bg-hover: #252540;
    
    --text-primary: #e8e8f0;
    --text-secondary: #8a8aa0;
    --text-muted: #5a5a70;
    
    --accent: #6c5ce7;
    --accent-light: #a29bfe;
    --accent-dark: #4a3db8;
    --accent-gradient: linear-gradient(135deg, #6c5ce7, #00cec9);
    
    --success: #00b894;
    --warning: #fdcb6e;
    --error: #e17055;
    
    --border-color: #2a2a42;
    --shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
    --radius: 12px;
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: var(--bg-primary);
    color: var(--text-primary);
    height: 100vh;
    overflow: hidden;
}
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

.app-container {
    display: flex;
    height: 100vh;
    background: var(--bg-primary);
}

.sidebar {
    width: 260px;
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-color);
    display: flex;
    flex-direction: column;
    padding: 20px 16px;
    flex-shrink: 0;
}
.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding-bottom: 20px;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 16px;
}
.logo-icon {
    font-size: 28px;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.logo-text {
    font-size: 22px;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.model-info {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    background: var(--bg-card);
    border-radius: var(--radius);
    margin-bottom: 16px;
}
.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-muted);
    flex-shrink: 0;
}
.status-dot.active {
    background: var(--success);
    box-shadow: 0 0 12px var(--success);
}
.model-name { font-size: 13px; font-weight: 500; flex: 1; }
.model-badge {
    font-size: 10px;
    background: var(--bg-hover);
    padding: 2px 8px;
    border-radius: 10px;
    color: var(--text-secondary);
    text-transform: uppercase;
}
.btn-new-chat {
    width: 100%;
    padding: 10px 16px;
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius);
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
}
.btn-new-chat:hover {
    background: var(--accent-dark);
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(108, 92, 231, 0.3);
}
.sidebar-footer {
    margin-top: auto;
    padding-top: 16px;
    border-top: 1px solid var(--border-color);
}
.stats {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: var(--text-muted);
}

.chat-container {
    flex: 1;
    display: flex;
    flex-direction: column;
    background: var(--bg-primary);
}
.chat-header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
    background: var(--bg-secondary);
}
.chat-header h1 {
    font-size: 18px;
    font-weight: 600;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.header-actions {
    display: flex;
    align-items: center;
    gap: 16px;
}
.toggle-label {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    color: var(--text-secondary);
    cursor: pointer;
}
.toggle-label input[type="checkbox"] { display: none; }
.toggle-slider {
    width: 36px;
    height: 20px;
    background: var(--bg-hover);
    border-radius: 10px;
    position: relative;
    transition: var(--transition);
}
.toggle-slider::after {
    content: '';
    position: absolute;
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    top: 2px;
    left: 2px;
    transition: var(--transition);
}
.toggle-label input:checked + .toggle-slider { background: var(--accent); }
.toggle-label input:checked + .toggle-slider::after { left: 18px; }
select {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: 6px 12px;
    border-radius: var(--radius);
    font-size: 12px;
    cursor: pointer;
    outline: none;
}
select:focus { border-color: var(--accent); }

.messages-area {
    flex: 1;
    overflow-y: auto;
    padding: 20px 24px;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.welcome-message {
    text-align: center;
    padding: 40px 20px;
    max-width: 600px;
    margin: auto;
}
.welcome-icon { font-size: 64px; margin-bottom: 16px; }
.welcome-message h2 { font-size: 28px; font-weight: 700; margin-bottom: 8px; }
.welcome-message p { color: var(--text-secondary); margin-bottom: 24px; }
.suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    justify-content: center;
}
.suggestion-chip {
    padding: 8px 16px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    font-size: 13px;
    color: var(--text-secondary);
    cursor: pointer;
    transition: var(--transition);
}
.suggestion-chip:hover {
    background: var(--bg-hover);
    border-color: var(--accent);
    color: var(--text-primary);
}

.message {
    display: flex;
    gap: 12px;
    max-width: 85%;
    animation: fadeIn 0.4s ease;
}
.message.user { align-self: flex-end; flex-direction: row-reverse; }
.message.assistant { align-self: flex-start; }
.message-avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
}
.message.user .message-avatar { background: var(--accent); }
.message.assistant .message-avatar { background: var(--bg-hover); }
.message-content {
    background: var(--bg-card);
    padding: 12px 16px;
    border-radius: var(--radius);
    line-height: 1.6;
    font-size: 14px;
}
.message.user .message-content {
    background: var(--accent);
    border-bottom-right-radius: 4px;
}
.message.assistant .message-content {
    border-bottom-left-radius: 4px;
}
.message-content .step {
    display: block;
    padding: 4px 0;
    border-bottom: 1px solid var(--border-color);
    margin-bottom: 4px;
}
.message-content .step:last-child { border-bottom: none; }
.message-content .step-number {
    font-weight: 600;
    color: var(--accent-light);
}
.message-content .final-answer {
    margin-top: 8px;
    padding: 8px 12px;
    background: rgba(108, 92, 231, 0.15);
    border-radius: 8px;
    border-left: 3px solid var(--accent);
    font-weight: 600;
}
.message-content .final-answer span { color: var(--accent-light); }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.input-area {
    padding: 16px 24px;
    border-top: 1px solid var(--border-color);
    background: var(--bg-secondary);
    flex-shrink: 0;
}
.input-wrapper {
    display: flex;
    gap: 12px;
    background: var(--bg-card);
    border-radius: var(--radius);
    border: 1px solid var(--border-color);
    padding: 4px 4px 4px 16px;
    transition: var(--transition);
}
.input-wrapper:focus-within {
    border-color: var(--accent);
    box-shadow: 0 0 20px rgba(108, 92, 231, 0.15);
}
#question-input {
    flex: 1;
    background: transparent;
    border: none;
    color: var(--text-primary);
    font-size: 14px;
    font-family: inherit;
    padding: 10px 0;
    resize: none;
    outline: none;
    max-height: 120px;
    line-height: 1.5;
}
#question-input::placeholder { color: var(--text-muted); }
#send-btn {
    width: 44px;
    height: 44px;
    border-radius: var(--radius);
    border: none;
    background: var(--accent);
    color: white;
    cursor: pointer;
    transition: var(--transition);
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
#send-btn:hover {
    background: var(--accent-dark);
    transform: scale(1.05);
}
#send-btn:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
}
.input-footer {
    display: flex;
    justify-content: space-between;
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
    padding: 0 4px;
}

.loading-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.6);
    backdrop-filter: blur(4px);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 999;
}
.loading-overlay.hidden { display: none; }
.loading-spinner {
    width: 48px;
    height: 48px;
    border: 4px solid var(--bg-card);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}
.loading-overlay span {
    margin-top: 16px;
    color: var(--text-secondary);
    font-size: 14px;
}
@keyframes spin {
    to { transform: rotate(360deg); }
}
@media (max-width: 768px) {
    .sidebar { display: none; }
    .chat-header { flex-wrap: wrap; gap: 8px; padding: 12px 16px; }
    .header-actions { flex-wrap: wrap; }
    .messages-area { padding: 12px 16px; }
    .message { max-width: 95%; }
    .input-area { padding: 12px 16px; }
}'''

        # ---- app.js (CORRECTED - NO TRAILING SLASH) ----
        app_js = '''// ===== CRAFT Chat Application =====
document.addEventListener('DOMContentLoaded', () => {
    // CRITICAL: No trailing slash - ensures requests go to /infer and /health
    const API_BASE = 'http://localhost:8000';
    let isProcessing = false;

    const messagesArea = document.getElementById('messages-area');
    const inputEl = document.getElementById('question-input');
    const sendBtn = document.getElementById('send-btn');
    const loadingOverlay = document.getElementById('loading-overlay');
    const mockToggle = document.getElementById('mock-toggle');
    const tempSelect = document.getElementById('temp-select');
    const latencyDisplay = document.getElementById('latency-display');
    const modeDisplay = document.getElementById('mode-display');
    const charCount = document.getElementById('char-count');

    inputEl.addEventListener('input', () => {
        charCount.textContent = `${inputEl.value.length}`;
        inputEl.style.height = 'auto';
        inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
    });

    window.handleKeyDown = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    };

    window.sendQuestion = (question) => {
        inputEl.value = question;
        charCount.textContent = `${question.length}`;
        sendMessage();
    };

    window.resetChat = () => {
        const welcome = messagesArea.querySelector('.welcome-message');
        messagesArea.innerHTML = '';
        if (welcome) messagesArea.appendChild(welcome);
        else {
            messagesArea.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">🤖</div>
                    <h2>Welcome to CRAFT</h2>
                    <p>Ask me any reasoning or math question, and I'll provide step-by-step thinking.</p>
                    <div class="suggestions">
                        <button class="suggestion-chip" onclick="sendQuestion('What is 15% of 240?')">15% of 240</button>
                        <button class="suggestion-chip" onclick="sendQuestion('Would a penguin survive in the Sahara desert?')">Penguin in Sahara</button>
                        <button class="suggestion-chip" onclick="sendQuestion('If a train travels at 60 mph for 2.5 hours, how far does it go?')">Train distance</button>
                        <button class="suggestion-chip" onclick="sendQuestion('Is a tomato a fruit or a vegetable?')">Tomato classification</button>
                    </div>
                </div>
            `;
        }
        inputEl.value = '';
        charCount.textContent = '0';
        inputEl.style.height = 'auto';
    };

    window.sendMessage = async () => {
        const question = inputEl.value.trim();
        if (!question || isProcessing) return;

        const welcome = messagesArea.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        addMessage('user', question);
        inputEl.value = '';
        charCount.textContent = '0';
        inputEl.style.height = 'auto';

        isProcessing = true;
        sendBtn.disabled = true;
        loadingOverlay.classList.remove('hidden');

        try {
            const payload = {
                question: question,
                temperature: parseFloat(tempSelect.value),
                max_tokens: 512
            };

            // Correct URL: API_BASE + '/infer' → http://localhost:8000/infer
            const response = await fetch(`${API_BASE}/infer`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                throw new Error(`API returned status ${response.status}`);
            }

            const data = await response.json();

            latencyDisplay.textContent = `${data.latency_ms}ms`;
            modeDisplay.textContent = data.is_mock ? 'Mock Mode' : 'GGUF Model';

            addMessage('assistant', data.response, data.steps, data.final_answer);

        } catch (error) {
            console.error(error);
            addMessage('assistant', `⚠️ Error: ${error.message}\\n\\nPlease make sure the inference server is running at ${API_BASE}`);
        } finally {
            isProcessing = false;
            sendBtn.disabled = false;
            loadingOverlay.classList.add('hidden');
        }
    };

    function addMessage(role, content, steps = null, finalAnswer = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? '👤' : '🤖';

        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';

        if (role === 'assistant' && steps && steps.length > 0) {
            steps.forEach((step, index) => {
                const stepDiv = document.createElement('div');
                stepDiv.className = 'step';
                stepDiv.innerHTML = `<span class="step-number">Step ${index + 1}:</span> ${step}`;
                contentDiv.appendChild(stepDiv);
            });
            if (finalAnswer) {
                const faDiv = document.createElement('div');
                faDiv.className = 'final-answer';
                faDiv.innerHTML = `🎯 Final Answer: <span>${finalAnswer}</span>`;
                contentDiv.appendChild(faDiv);
            }
        } else {
            contentDiv.textContent = content;
        }

        messageDiv.appendChild(avatar);
        messageDiv.appendChild(contentDiv);
        messagesArea.appendChild(messageDiv);
        messagesArea.scrollTop = messagesArea.scrollHeight;
        messageDiv.style.animation = 'none';
        requestAnimationFrame(() => {
            messageDiv.style.animation = 'fadeIn 0.4s ease';
        });
    }

    async function checkHealth() {
        try {
            // Correct URL: API_BASE + '/health' → http://localhost:8000/health
            const response = await fetch(`${API_BASE}/health`);
            if (response.ok) {
                const data = await response.json();
                const dot = document.querySelector('.status-dot');
                dot.className = 'status-dot active';
                modeDisplay.textContent = data.mock_mode ? 'Mock Mode' : 'GGUF Model';
            }
        } catch (e) {
            console.log('Server not reachable');
        }
    }
    checkHealth();
    setInterval(checkHealth, 10000);
});'''

        # Write files
        with open(os.path.join(ui_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(index_html)
        with open(os.path.join(ui_dir, "styles.css"), "w", encoding="utf-8") as f:
            f.write(styles_css)
        with open(os.path.join(ui_dir, "app.js"), "w", encoding="utf-8") as f:
            f.write(app_js)

        logger.info("Created enhanced chat UI files")

    def _copy_model(self, gguf_path):
        """Copy the quantized model into the delivery bundle."""
        if gguf_path and os.path.exists(gguf_path):
            logger.info(f"Copying GGUF model from {gguf_path}...")
            dest_model_path = os.path.join(self.delivery_dir, "craft_output", os.path.basename(gguf_path))
            shutil.copy2(gguf_path, dest_model_path)
            logger.info(f"Model copied to {dest_model_path}")
        else:
            logger.warning(f"GGUF model not found at {gguf_path}. Creating placeholder.")
            with open(os.path.join(self.delivery_dir, "craft_output", "craft_model_Q4_K_M.gguf.placeholder"), "w") as f:
                f.write("Download the quantized GGUF model and place it here.\n")
                f.write("Expected filename: craft_phi3_Q4_K_M.gguf\n")

    def _create_launch_scripts(self):
        """Create launch scripts for both Windows and Unix."""
        launch_sh = """#!/bin/bash
echo "=== Starting CRAFT Inference Server ==="
echo "Installing dependencies..."
pip install -q fastapi uvicorn llama-cpp-python

echo "Starting FastAPI server on port 8000..."
python -m uvicorn src.inference.inference_server:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!
echo "Server started with PID: $SERVER_PID"
echo "Open src/ui/index.html in your browser"
echo "Press Ctrl+C to stop the server"
wait $SERVER_PID
"""
        with open(os.path.join(self.delivery_dir, "launch.sh"), "w") as f:
            f.write(launch_sh.strip())
        if not self.is_windows:
            os.chmod(os.path.join(self.delivery_dir, "launch.sh"), 0o755)

        launch_bat = """@echo off
echo === Starting CRAFT Inference Server ===
echo Installing dependencies...
pip install -q fastapi uvicorn llama-cpp-python

echo Starting FastAPI server on port 8000...
start /B python -m uvicorn src.inference.inference_server:app --host 127.0.0.1 --port 8000
echo Server started!
echo Open src/ui/index.html in your browser
echo Press any key to stop the server...
pause >nul
taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*"
"""
        with open(os.path.join(self.delivery_dir, "launch.bat"), "w") as f:
            f.write(launch_bat.strip())

        logger.info("Created launch scripts: launch.sh (Unix) and launch.bat (Windows).")

    def _create_requirements(self):
        """Create a requirements_inference.txt for the inference server."""
        req_content = """fastapi>=0.104.0
uvicorn>=0.24.0
llama-cpp-python>=0.2.0
loguru>=0.7.0
pydantic>=2.0.0
"""
        req_path = os.path.join(self.delivery_dir, "requirements_inference.txt")
        with open(req_path, "w") as f:
            f.write(req_content.strip())
        logger.info("Created requirements_inference.txt")

    def _create_readme(self):
        """Create a README.md with instructions."""
        readme = """# CRAFT Deployment Package

## Contents
- `craft_output/` - Quantized GGUF model
- `src/inference/` - FastAPI inference server
- `src/ui/` - Web dashboard
- `launch.sh` / `launch.bat` - One-click startup scripts

## Quick Start

### On Linux/macOS
```bash
./launch.sh"""