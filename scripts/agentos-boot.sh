#!/usr/bin/env bash
# =============================================================================
# Agentic AI OS — Boot Sequence Animation
# =============================================================================

clear

CYAN='\033[0;36m'
BLUE='\033[1;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}"
cat << 'EOF'
  ___                    _   _         _    ___    ___  ____
 / _ \   __ _   ___ _ __ | |_(_) ___  / \  |_ _|  / _ \/ ___|
| | | | / _` | / _ \ '_ \| __| |/ __| / _ \  | |  | | | \___ \
| |_| || (_| ||  __/ | | | |_| | (__ / ___ \ | |  | |_| |___) |
 \__\_\ \__, | \___|_| |_|\__|_|\___/_/   \_\___|  \___/|____/
        |___/
                 Agentic AI OS — Boot Sequence
EOF
echo -e "${NC}"

sleep 1

SERVICES=("agentos-memory" "agentos-ollama" "agentos-sandbox" "agentos-api" "agentos-ui")

for svc in "${SERVICES[@]}"; do
    echo -ne "Starting ${CYAN}${svc}${NC}..."
    sleep 0.4
    if systemctl is-active --quiet "$svc"; then
        echo -e "\rStarting ${CYAN}${svc}${NC}... [ ${GREEN}OK${NC} ]"
    else
        echo -e "\rStarting ${CYAN}${svc}${NC}... [ ${RED}FAILED${NC} ]"
    fi
done

echo ""
echo -e "${GREEN}Welcome to Agentic AI OS! System is fully operational.${NC}"
echo ""
