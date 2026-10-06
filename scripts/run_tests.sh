#!/usr/bin/env bash
# Test execution script for FlightBookingAgent tools and agents
set -e

# Project root directory (one level above scripts/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=========================================================="
echo "  Running FlightBookingAgent Full Test Suite (Tools + Agents)"
echo "=========================================================="

# Ensure virtualenv site-packages and project root are on PYTHONPATH
if [ -d "$PROJECT_ROOT/venv/lib/python3.13/site-packages" ]; then
    export PYTHONPATH="$PROJECT_ROOT/venv/lib/python3.13/site-packages:$PROJECT_ROOT:$PYTHONPATH"
    PYTEST_BIN="$PROJECT_ROOT/venv/bin/pytest"
else
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    PYTEST_BIN="pytest"
fi

# Run all tests with pytest
"$PYTEST_BIN" -v "$PROJECT_ROOT/tests" "$@"

echo "=========================================================="
echo "  All FlightBookingAgent Tests Passed Successfully!        "
echo "=========================================================="
