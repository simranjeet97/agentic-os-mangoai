#!/usr/bin/env bash
# =============================================================================
# Agentic AI OS — Ubuntu 24.04 LTS Full VM Installation Script
# =============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Colors & Formatting ───────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

LOG_FILE="/var/log/agentic-os-install.log"
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log()     { echo -e "${GREEN}[INFO]${NC}  $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE"; }
error()   { echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') — $*" | tee -a "$LOG_FILE" >&2; }
section() { echo -e "\n${CYAN}${BOLD}══════════════════════════════════════════${NC}"; \
            echo -e "${CYAN}${BOLD}  $*${NC}"; \
            echo -e "${CYAN}${BOLD}══════════════════════════════════════════${NC}\n"; }

if [[ $EUID -ne 0 ]]; then
  error "This script must be run as root. Use: sudo ./install.sh"
  exit 1
fi

touch "$LOG_FILE"
chmod 644 "$LOG_FILE"

echo -e "${BOLD}${CYAN}"
cat << 'EOF'
  ___                    _   _         _    ___    ___  ____
 / _ \   __ _   ___ _ __ | |_(_) ___  / \  |_ _|  / _ \/ ___|
| | | | / _` | / _ \ '_ \| __| |/ __| / _ \  | |  | | | \___ \
| |_| || (_| ||  __/ | | | |_| | (__ / ___ \ | |  | |_| |___) |
 \__\_\ \__, | \___|_| |_|\__|_|\___/_/   \_\___|  \___/|____/
        |___/
                 Agentic AI OS — VM Installer v2.0
EOF
echo -e "${NC}"

SUDO_USER="${SUDO_USER:-$(logname 2>/dev/null || echo ubuntu)}"

section "Updating System Packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get upgrade -y -qq
apt-get install -y -qq \
  curl wget git build-essential software-properties-common \
  apt-transport-https ca-certificates gnupg lsb-release \
  unzip zip jq htop ncdu net-tools \
  libssl-dev libffi-dev zlib1g-dev \
  libsqlite3-dev libpq-dev \
  ffmpeg libsm6 libxext6 python3.12 python3.12-venv python3.12-dev docker.io redis

systemctl enable redis docker
systemctl start redis docker

usermod -aG docker "$SUDO_USER" || true

section "Installing Node.js 20 LTS"
if ! node --version 2>/dev/null | grep -q "v20"; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y -qq nodejs
  npm install -g npm@latest pnpm@latest
fi

section "Installing Ollama & Models"
if ! ollama --version &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
  systemctl enable ollama || true
  systemctl start ollama || ollama serve &>/dev/null &
  sleep 5
fi

log "Pulling llama3.2:3b..."
ollama pull llama3.2:3b || true
log "Pulling mistral:7b..."
ollama pull mistral:7b || true
log "Pulling nomic-embed-text..."
ollama pull nomic-embed-text || true

section "Setting Up Virtual Environment & Dependencies"
cd "$INSTALL_DIR"
if [[ ! -d ".venv" ]]; then
  python3.12 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
python -m playwright install chromium firefox > /dev/null 2>&1 || true

section "Installing Node.js Dependencies"
if [[ -d "ui" ]]; then
    cd "$INSTALL_DIR/ui"
    npm install > /dev/null 2>&1
    cd "$INSTALL_DIR"
fi

section "Setting Up Configuration"
mkdir -p "$INSTALL_DIR/config"
chown -R "$SUDO_USER:$SUDO_USER" "$INSTALL_DIR/config"

section "Deploying Systemd Services"
if [[ -d "scripts/systemd" ]]; then
    for svc in scripts/systemd/*.service; do
        if [ -f "$svc" ]; then
            filename=$(basename "$svc")
            sed -e "s|__INSTALL_DIR__|$INSTALL_DIR|g" \
                -e "s|__USER__|$SUDO_USER|g" \
                "$svc" > "/etc/systemd/system/$filename"
            log "Deployed service $filename"
        fi
    done
fi

systemctl daemon-reload
SERVICES=("agentos-sandbox" "agentos-ollama" "agentos-memory" "agentos-api" "agentos-ui")
for s in "${SERVICES[@]}"; do
    systemctl enable "$s" || warn "Could not enable $s"
    systemctl start "$s" || warn "Could not start $s"
done

section "Testing Installation"
chmod +x scripts/agentos-health.sh scripts/agentos-boot.sh || true
su - "$SUDO_USER" -c "bash $INSTALL_DIR/scripts/agentos-health.sh" || true

section "Installation Complete!"
su - "$SUDO_USER" -c "bash $INSTALL_DIR/scripts/agentos-boot.sh" || true
