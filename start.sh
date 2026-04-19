#!/bin/bash

# =============================================================================
# Agentic AI OS — Unified Startup Script
# This script starts both the FastAPI Backend and the Vite Frontend.
# =============================================================================

# ── Environment Setup ─────────────────────────────────────────────────────────

# Add common Mac/Homebrew paths if they are missing
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

# Colors for logging
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}🚀 Starting Agentic AI OS...${NC}"

# ── Port Cleanup ─────────────────────────────────────────────────────────────

echo -e "${YELLOW}🔍 Checking for existing services on ports 8000 and 5173...${NC}"
# Kill processes on backend port (8000)
BACKEND_PID=$(lsof -t -i:8000)
if [ ! -z "$BACKEND_PID" ]; then
    echo -e "${YELLOW}⚠️ Closing existing backend (PID: $BACKEND_PID)${NC}"
    kill -9 $BACKEND_PID 2>/dev/null
fi

# Kill processes on frontend port (5173)
FRONTEND_PID=$(lsof -t -i:5173)
if [ ! -z "$FRONTEND_PID" ]; then
    echo -e "${YELLOW}⚠️ Closing existing frontend (PID: $FRONTEND_PID)${NC}"
    kill -9 $FRONTEND_PID 2>/dev/null
fi
sleep 1

# ── Pre-flight Checks ────────────────────────────────────────────────────────

# Check for Python virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Error: .venv directory not found.${NC}"
    echo "Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Check for Node.js (required for UI)
if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}⚠️ Warning: Node.js not found in PATH.${NC}"
    echo "The UI requires Node.js to run. Please install it from https://nodejs.org/"
    # We continue so the API can at least start
else
    echo -e "${GREEN}✅ Node.js found: $(node -v)${NC}"
    # Check for node_modules in UI
    if [ ! -d "ui/node_modules" ]; then
        echo -e "${YELLOW}⚠️ Warning: ui/node_modules not found.${NC}"
        echo "Please run 'cd ui && npm install' before starting."
    fi
fi

# ── Start Services ───────────────────────────────────────────────────────────

# Function to kill child processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down services...${NC}"
    kill $API_PID $UI_PID 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# 1. Start FastAPI Backend
echo -e "${CYAN}📡 Starting Backend API on port 8000...${NC}"
.venv/bin/python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

# 2. Start Vite Frontend (if node is available)
if command -v npm &> /dev/null; then
    echo -e "${CYAN}🎨 Starting Frontend UI...${NC}"
    cd ui
    npm run dev &
    UI_PID=$!
    cd ..
else
    echo -e "${RED}❌ Skipping UI startup because npm is missing.${NC}"
fi

echo -e "${GREEN}✨ Services are initializing!${NC}"
echo -e "Backend: ${CYAN}http://localhost:8000${NC}"
echo -e "Frontend: ${CYAN}http://localhost:5173${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop all services.${NC}"

# Keep script running
wait
