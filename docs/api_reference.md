# API Reference

## Navigation
- [Main README](../README.md)
- [Technical Stack](technical_stack.md)
- [Architecture](architecture.md)
- [Implementation Details](implementation_details.md)
- [Installation Guide](installation.md)
- [User Guide](user_guide.md)
- [Features](features.md)
- [Benchmark Results](benchmark_results.md)
- [API Reference](api_reference.md)
- [Agentic AI Development (ax.md)](ax.md)

---

## Overview
CRAFT's deployment module hosts a local asynchronous FastAPI server. The server interfaces with quantized GGUF binaries using `llama-cpp-python` and exposes API endpoints for reasoning generation and system diagnostics.

By default, the server is hosted at: `http://127.0.0.1:8000`

---

## Endpoints

### 1. GET /health
Checks server responsiveness and outputs configuration metadata.

* **Method:** `GET`
* **Path:** `/health`
* **Headers:** `Content-Type: application/json`
* **Response Model:**
  * **`status`** (string): Server health indicator.
  * **`mock_mode`** (boolean): True if server is in mock evaluation mode.
  * **`model_path`** (string): Filesystem path to the GGUF model binary.
  * **`model_exists`** (boolean): True if the binary exists at `model_path`.
  * **`architecture`** (string): Detected model architecture (`"phi3"`, `"qwen2"`, or `"unknown"`).

#### Example Request
```bash
curl -X GET http://127.0.0.1:8000/health
```

#### Example Response
```json
{
  "status": "healthy",
  "mock_mode": false,
  "model_path": "craft_output/CRAFT_Q4_K_M.gguf",
  "model_exists": true,
  "architecture": "qwen2"
}
```

---

### 2. POST /infer
Submits a reasoning prompt, runs the model, parses step-by-step reasoning sequences, and extracts the final answer.

* **Method:** `POST`
* **Path:** `/infer`
* **Headers:** `Content-Type: application/json`

#### Request Schema (Pydantic model: `InferenceRequest`)
| Field | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`question`** | string | *Required* | The problem statement, question, or logical prompt to solve. |
| **`temperature`** | float | `0.8` | Sampling temperature regulating output randomness. |
| **`max_tokens`** | integer | `512` | Token limit for model output generation. |

#### Response Schema (Pydantic model: `InferenceResponse`)
| Field | Type | Description |
| :--- | :--- | :--- |
| **`question`** | string | The original query submitted. |
| **`response`** | string | Raw, unparsed text response from the model. |
| **`steps`** | list of strings | Parsed list containing individual reasoning steps. |
| **`final_answer`**| string | The extracted short final answer. |
| **`latency_ms`** | float | Latency in milliseconds. |
| **`is_mock`** | boolean | Indicates if the response was mocked. |

#### Example Request
```bash
curl -X POST http://127.0.0.1:8000/infer \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Solve: Weng earns $12/hour for 5 hours. How much did he earn?",
    "temperature": 0.5,
    "max_tokens": 256
  }'
```

#### Example Response
```json
{
  "question": "Solve: Weng earns $12/hour for 5 hours. How much did he earn?",
  "response": "Step 1: Multiply Weng's hourly rate by the number of hours worked: $12 * 5.\nStep 2: Calculate the product: 12 * 5 = 60.\nFinal Answer: 60",
  "steps": [
    "Multiply Weng's hourly rate by the number of hours worked: $12 * 5.",
    "Calculate the product: 12 * 5 = 60."
  ],
  "final_answer": "60",
  "latency_ms": 184.32,
  "is_mock": false
}
```
