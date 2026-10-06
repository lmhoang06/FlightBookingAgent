#!/usr/bin/env bash
# Quick-start script for Streamlit application
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Starting Streamlit FlightBookingAgent Playground..."

if [ -f "$PROJECT_ROOT/venv/bin/streamlit" ]; then
    STREAMLIT_BIN="$PROJECT_ROOT/venv/bin/streamlit"
else
    STREAMLIT_BIN="streamlit"
fi

"$STREAMLIT_BIN" run "$PROJECT_ROOT/app.py" --server.port 8501 --server.headless true
