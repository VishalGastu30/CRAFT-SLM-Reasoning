#!/bin/bash
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