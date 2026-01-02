#!/usr/bin/env bash

# Function to handle script termination
cleanup() {
    echo "Stopping backend..."
    if [ -n "$BACKEND_PID" ]; then
        kill $BACKEND_PID
    fi
    exit
}

# Trap SIGINT (Ctrl+C) to run cleanup
trap cleanup SIGINT

echo "Installing/Updating uv..."
pip install uv

echo "Starting Backend..."
# Run uvicorn in the background
uv run uvicorn backend.app.main:app --reload &
BACKEND_PID=$!

echo "Backend started with PID $BACKEND_PID"

echo "Starting Streamlit..."
uv run --with streamlit streamlit run main.py --server.port=8501 --server.address=0.0.0.0 --server.enableWebsocketCompression=false --server.enableCORS=false
