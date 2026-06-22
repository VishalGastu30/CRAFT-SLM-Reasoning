// ===== CRAFT Chat Application =====
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
            addMessage('assistant', `⚠️ Error: ${error.message}\n\nPlease make sure the inference server is running at ${API_BASE}`);
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
});