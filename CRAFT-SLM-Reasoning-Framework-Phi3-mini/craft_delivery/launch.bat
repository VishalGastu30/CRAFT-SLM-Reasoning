@echo off
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