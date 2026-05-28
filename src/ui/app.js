document.addEventListener("DOMContentLoaded", () => {
  const API_BASE = "http://localhost:8000";
  
  // DOM Elements
  const connectionStatus = document.getElementById("connection-status");
  const questionInput = document.getElementById("question-input");
  const tempSelect = document.getElementById("temperature-select");
  const modelModeSelect = document.getElementById("model-mode-select");
  const runBtn = document.getElementById("run-btn");
  const playgroundLoader = document.getElementById("playground-loader");
  const stepsContainer = document.getElementById("steps-container");
  const finalAnswerContainer = document.getElementById("final-answer-container");
  const finalAnswerVal = document.getElementById("final-answer-val");
  const latencyVal = document.getElementById("latency-val");
  const modeVal = document.getElementById("mode-val");

  // Check connection status on startup
  async function checkConnection() {
    try {
      const response = await fetch(`${API_BASE}/health`);
      if (response.ok) {
        const data = await response.json();
        if (data.mock_mode) {
          connectionStatus.textContent = "CONNECTED (MOCK MODE)";
          connectionStatus.style.borderColor = "var(--color-warning)";
          connectionStatus.classList.add("active");
          modeVal.textContent = "FastAPI Mock";
        } else {
          connectionStatus.textContent = "CONNECTED (REAL GGUF)";
          connectionStatus.style.borderColor = "var(--color-success)";
          connectionStatus.classList.add("active");
          modeVal.textContent = "Phi-3-Mini (GGUF)";
        }
      } else {
        showOfflineStatus();
      }
    } catch (e) {
      showOfflineStatus();
    }
  }

  function showOfflineStatus() {
    connectionStatus.textContent = "OFFLINE (FastAPI server disconnected)";
    connectionStatus.style.borderColor = "var(--color-warning)";
    connectionStatus.classList.remove("active");
    modeVal.textContent = "Disconnected";
  }

  // Initial connection check
  checkConnection();
  // Poll connection status every 5 seconds
  setInterval(checkConnection, 5000);

  // Quick preset questions mapping
  questionInput.addEventListener("focus", () => {
    if (!questionInput.value) {
      // Suggest a template question
      questionInput.placeholder = "Try: 'What is 15% of 240?' or 'Would a penguin survive in the Sahara desert?'";
    }
  });

  // Run Inference logic
  runBtn.addEventListener("click", async () => {
    const question = questionInput.value.trim();
    if (!question) {
      alert("Please enter a question or reasoning prompt!");
      return;
    }

    // Set UI to loading state
    runBtn.disabled = true;
    playgroundLoader.style.display = "block";
    stepsContainer.innerHTML = "";
    finalAnswerContainer.style.display = "none";
    latencyVal.textContent = "calculating...";
    
    try {
      const payload = {
        question: question,
        temperature: parseFloat(tempSelect.value),
        max_tokens: 512
      };

      const response = await fetch(`${API_BASE}/infer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error(`Inference API returned status ${response.status}`);
      }

      const data = await response.json();
      
      // Update latency and execution metrics
      latencyVal.textContent = `${data.latency_ms} ms`;
      modeVal.textContent = data.is_mock ? "FastAPI Mock" : "Phi-3-Mini (GGUF)";
      
      // Render steps with transition delays
      if (data.steps && data.steps.length > 0) {
        stepsContainer.innerHTML = "";
        data.steps.forEach((step, index) => {
          setTimeout(() => {
            const card = document.createElement("div");
            card.className = "step-card";
            
            const stepNum = document.createElement("span");
            stepNum.className = "step-number";
            stepNum.textContent = `Step ${index + 1}`;
            
            const stepContent = document.createElement("p");
            stepContent.className = "step-content";
            stepContent.textContent = step;
            
            card.appendChild(stepNum);
            card.appendChild(stepContent);
            stepsContainer.appendChild(card);
            
            // Scroll to bottom of panel
            stepsContainer.scrollTop = stepsContainer.scrollHeight;
          }, index * 400); // 400ms progressive render effect
        });

        // Show final answer box after all steps are rendered
        setTimeout(() => {
          finalAnswerVal.textContent = data.final_answer || "N/A";
          finalAnswerContainer.style.display = "block";
        }, data.steps.length * 400);
      } else {
        stepsContainer.innerHTML = `<div class="step-card"><p class="step-content">${data.response}</p></div>`;
        finalAnswerVal.textContent = data.final_answer || "N/A";
        finalAnswerContainer.style.display = "block";
      }

    } catch (error) {
      console.error(error);
      stepsContainer.innerHTML = `
        <div class="step-card" style="border-color: rgba(224, 36, 36, 0.3); background: rgba(224, 36, 36, 0.05);">
          <span class="step-number" style="color: #E02424;">Connection Error</span>
          <p class="step-content">Failed to connect to local edge server. Please ensure that you have run: <br><code>uvicorn src.inference.inference_server:app --port 8000</code></p>
        </div>
      `;
      latencyVal.textContent = "Error";
    } finally {
      runBtn.disabled = false;
      playgroundLoader.style.display = "none";
    }
  });
});
