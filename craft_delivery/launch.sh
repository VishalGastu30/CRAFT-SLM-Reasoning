#!/bin/bash
echo "=== Starting CRAFT Inference Server & Interactive Dashboard ==="
echo "Installing minimal python dependencies..."
pip install -r requirements_inference.txt

echo "Starting FastAPI Backend Server on port 8000..."
python -m uvicorn src.inference.inference_server:app --host 127.0.0.1 --port 8000 &
SERVER_PID=$!

echo "FastAPI Server launched with PID: $SERVER_PID"
echo "--------------------------------------------------------"
echo "Dashboard UI is now active!"
echo "Please open src/ui/index.html in your web browser."
echo "--------------------------------------------------------"
echo "Press Ctrl+C to terminate the server."
wait $SERVER_PID