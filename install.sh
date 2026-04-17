#!/usr/bin/env bash
# =============================================================================
# Agentic AI OS — Ubuntu 24.04 LTS Full Installation Script
# =============================================================================
# Usage: sudo ./install.sh
# Installs: Python 3.12, Node 20, Redis 7, Docker CE, Ollama, ChromaDB,
#           and all project dependencies.
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Colors & Formatting ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FILE="/var/log/agentic-os-install.log"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()     { echo -e "${GREEN}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE" >&2; }
section() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════${NC}"; \
            echo -e "${CYAN}${BOLD}  $*${NC}"; \
            echo -e "${CYAN}${BOLD}══════════════════════════════════════════${NC}\n"; }

# ── Guards ────────────────────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  error "This script must be run as root. Use: sudo ./install.sh"
  exit 1
fi

if ! grep -q "Ubuntu 24" /etc/os-release 2>/dev/null; then
  warn "This script is optimized for Ubuntu 24.04 LTS. Proceeding anyway..."
fi

touch "$LOG_FILE"
chmod 644 "$LOG_FILE"

echo -e "${BOLD}${BLUE}"
cat << 'EOF'
  ___                    _   _         _    ___    ___  ____
 / _ \   __ _   ___ _ __ | |_(_) ___  / \  |_ _|  / _ \/ ___|
| | | | / _` | / _ \ '_ \| __| |/ __| / _ \  | |  | | | \___ \
| |_| || (_| ||  __/ | | | |_| | (__ / ___ \ | |  | |_| |___) |
 \__\_\ \__, | \___|_| |_|\__|_|\___/_/   \_\___|  \___/|____/
        |___/
                 Agentic AI OS — Installer v1.0.0
EOF
echo -e "${NC}"

log "Installation started. Logging to $LOG_FILE"
log "Install directory: $INSTALL_DIR"

# ── System Update ─────────────────────────────────────────────────────────────
section "Updating System Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y -qq \
  curl wget git build-essential software-properties-common \
  apt-transport-https ca-certificates gnupg lsb-release \
  unzip zip jq htop ncdu net-tools \
  libssl-dev libffi-dev zlib1g-dev \
  libsqlite3-dev libpq-dev \
  ffmpeg libsm6 libxext6
log "System packages updated ✓"

# ── Python 3.12 ───────────────────────────────────────────────────────────────
section "Installing Python 3.12"
if python3.12 --version &>/dev/null; then
  log "Python 3.12 already installed: $(python3.12 --version)"
else
  add-apt-repository -y ppa:deadsnakes/ppa
  apt-get update -qq
  apt-get install -y -qq \
    python3.12 python3.12-dev python3.12-venv python3.12-distutils
  curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12
  update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 10
  log "Python 3.12 installed: $(python3.12 --version) ✓"
fi

# ── Node.js 20 LTS ────────────────────────────────────────────────────────────
section "Installing Node.js 20 LTS"
if node --version 2>/dev/null | grep -q "v20"; then
  log "Node 20 already installed: $(node --version)"
else
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
  npm install -g npm@latest
  log "Node.js installed: $(node --version) ✓"
  log "npm installed: $(npm --version) ✓"
fi

# ── Install pnpm (optional, fast package manager) ─────────────────────────────
npm install -g pnpm@latest || warn "pnpm install failed, using npm"
log "Node ecosystem ready ✓"

# ── Redis 7 ───────────────────────────────────────────────────────────────────
section "Installing Redis 7"
if redis-cli --version &>/dev/null; then
  log "Redis already installed: $(redis-cli --version)"
else
  curl -fsSL https://packages.redis.io/gpg | \
    gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
  echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] \
    https://packages.redis.io/deb $(lsb_release -cs) main" | \
    tee /etc/apt/sources.list.d/redis.list
  apt-get update -qq
  apt-get install -y -qq redis
  systemctl enable redis-server
  systemctl start redis-server
  log "Redis installed: $(redis-cli --version) ✓"
fi

# ── Docker CE ─────────────────────────────────────────────────────────────────
section "Installing Docker CE"
if docker --version &>/dev/null; then
  log "Docker already installed: $(docker --version)"
else
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) \
    signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu \
    $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
    tee /etc/apt/sources.list.d/docker.list > /dev/null
  apt-get update -qq
  apt-get install -y -qq \
    docker-ce docker-ce-cli containerd.io \
    docker-buildx-plugin docker-compose-plugin
  systemctl enable docker
  systemctl start docker
  # Add current user to docker group
  SUDO_USER="${SUDO_USER:-$(logname 2>/dev/null || echo ubuntu)}"
  usermod -aG docker "$SUDO_USER" || true
  log "Docker installed: $(docker --version) ✓"
fi

# ── Ollama ────────────────────────────────────────────────────────────────────
section "Installing Ollama"
if ollama --version &>/dev/null; then
  log "Ollama already installed: $(ollama --version)"
else
  curl -fsSL https://ollama.com/install.sh | sh
  systemctl enable ollama || true
  systemctl start ollama || ollama serve &>/dev/null &
  sleep 5
  log "Ollama installed ✓"
fi

# Pull default models
section "Pulling Ollama Models"
log "Pulling llama3.2:3b (fast, small model for dev)..."
ollama pull llama3.2:3b || warn "Failed to pull llama3.2:3b"
log "Pulling mistral:7b..."
ollama pull mistral:7b || warn "Failed to pull mistral:7b"
log "Pulling nomic-embed-text (for embeddings)..."
ollama pull nomic-embed-text || warn "Failed to pull nomic-embed-text"
log "Ollama models ready ✓"

# ── Python Virtual Environment & Dependencies ─────────────────────────────────
section "Setting Up Python Virtual Environment"
cd "$INSTALL_DIR"
if [[ ! -d ".venv" ]]; then
  python3.12 -m venv .venv
  log "Virtual environment created ✓"
fi

source .venv/bin/activate
pip install --upgrade pip wheel setuptools -q

log "Installing Python dependencies (this may take a few minutes)..."
pip install -r requirements.txt -q || {
  error "Failed to install some Python packages. Check $LOG_FILE for details."
}

# Install Playwright browsers
python -m playwright install chromium firefox || warn "Playwright browser install failed"
log "Python dependencies installed ✓"

# ── Node.js Dependencies ──────────────────────────────────────────────────────
section "Installing Node.js Dependencies"
cd "$INSTALL_DIR"
npm install || warn "Root npm install failed"
if [[ -d "ui" ]]; then
  cd ui && npm install && cd ..
  log "UI dependencies installed ✓"
fi

# ── Environment File ──────────────────────────────────────────────────────────
section "Creating Environment Configuration"
if [[ ! -f "$INSTALL_DIR/.env" ]]; then
  cat > "$INSTALL_DIR/.env" << 'ENVEOF'
# =============================================================================
# Agentic AI OS — Environment Configuration
# DO NOT commit this file to version control
# =============================================================================

# App
APP_NAME=agentic-os
ENVIRONMENT=development
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=CHANGE_ME_TO_A_RANDOM_64_CHAR_STRING

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=2
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# Redis
REDIS_URL=redis://:agenticsecret@localhost:6379/0
REDIS_PASSWORD=agenticsecret
REDIS_MAX_CONNECTIONS=20

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8001
CHROMA_COLLECTION_PREFIX=agentic

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_DEFAULT_MODEL=llama3.2:3b
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_TIMEOUT=120

# Security
JWT_SECRET_KEY=CHANGE_ME_TO_ANOTHER_RANDOM_SECRET
JWT_ALGORITHM=HS256
JWT_EXPIRE_MINUTES=1440

# Guardrails
MAX_TOKENS_PER_REQUEST=8192
RATE_LIMIT_REQUESTS_PER_MINUTE=60
SANDBOX_TIMEOUT_SECONDS=30
ENABLE_CONTENT_FILTER=true

# Grafana
GRAFANA_PASSWORD=admin123

# Optional — External APIs (leave empty if not using)
SERPAPI_KEY=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
ENVEOF
  log ".env created — IMPORTANT: Update SECRET_KEY and JWT_SECRET_KEY before production use ✓"
else
  log ".env already exists, skipping ✓"
fi

# ── Systemd Services ──────────────────────────────────────────────────────────
section "Installing Systemd Services"
cat > /etc/systemd/system/agentic-api.service << SVCEOF
[Unit]
Description=Agentic AI OS — FastAPI Backend
After=network.target redis.service ollama.service
Requires=redis.service

[Service]
Type=exec
User=${SUDO_USER:-ubuntu}
WorkingDirectory=$INSTALL_DIR
Environment=PATH=$INSTALL_DIR/.venv/bin:/usr/local/bin:/usr/bin:/bin
ExecStart=$INSTALL_DIR/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal
SyslogIdentifier=agentic-api

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable agentic-api || true
log "Systemd services configured ✓"

# ── Firewall Configuration ────────────────────────────────────────────────────
section "Configuring Firewall"
if command -v ufw &>/dev/null; then
  ufw allow 8000/tcp comment "Agentic API" || true
  ufw allow 3000/tcp comment "Agentic UI" || true
  ufw allow 11434/tcp comment "Ollama" || true
  log "UFW rules configured ✓"
fi

# ── Create Log Directory ──────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR/logs"
chown -R "${SUDO_USER:-ubuntu}:${SUDO_USER:-ubuntu}" "$INSTALL_DIR/logs" || true

# ── Verify Installation ───────────────────────────────────────────────────────
section "Verifying Installation"
verify() {
  if eval "$2" &>/dev/null; then
    log "  ✓ $1"
  else
    warn "  ✗ $1 — may need manual attention"
  fi
}

verify "Python 3.12"      "python3.12 --version"
verify "pip"              "pip --version"
verify "Node.js 20"       "node --version | grep -q v20"
verify "npm"              "npm --version"
verify "Redis"            "redis-cli ping | grep -q PONG"
verify "Docker"           "docker info"
verify "Docker Compose"   "docker compose version"
verify "Ollama"           "curl -sf http://localhost:11434/api/tags"
verify "Python venv"      "test -d $INSTALL_DIR/.venv"
verify ".env file"        "test -f $INSTALL_DIR/.env"

# ── Summary ───────────────────────────────────────────────────────────────────
section "Installation Complete!"
echo -e "${GREEN}${BOLD}"
cat << 'EOF'

  ╔═══════════════════════════════════════════════════════════╗
  ║           Agentic AI OS Successfully Installed!           ║
  ╠═══════════════════════════════════════════════════════════╣
  ║                                                           ║
  ║  Next Steps:                                              ║
  ║                                                           ║
  ║  1. Edit .env and set your SECRET_KEY values             ║
  ║                                                           ║
  ║  2. Start the full stack:                                 ║
  ║     docker compose up -d                                  ║
  ║                                                           ║
  ║  3. Activate Python environment:                          ║
  ║     source .venv/bin/activate                             ║
  ║                                                           ║
  ║  4. Start the API server:                                 ║
  ║     uvicorn api.main:app --reload                         ║
  ║                                                           ║
  ║  5. Start the UI:                                         ║
  ║     cd ui && npm run dev                                  ║
  ║                                                           ║
  ║  API:        http://localhost:8000                        ║
  ║  API Docs:   http://localhost:8000/docs                   ║
  ║  ChromaDB:   http://localhost:8001                        ║
  ║  Grafana:    http://localhost:3001                        ║
  ║  Ollama:     http://localhost:11434                       ║
  ║                                                           ║
  ║  Log file: /var/log/agentic-os-install.log               ║
  ╚═══════════════════════════════════════════════════════════╝

EOF
echo -e "${NC}"

log "Installation completed successfully at $(date)"
log "Full log available at: $LOG_FILE"

# Remind about group membership
warn "NOTE: Log out and back in (or run 'newgrp docker') for Docker group permissions to take effect."
