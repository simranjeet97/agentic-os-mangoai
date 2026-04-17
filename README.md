# 🧠 Agentic AI OS

> A production-grade, self-hosted Agentic AI Operating System powered by LangGraph, FastAPI, Ollama, and a secure multi-agent orchestration engine — running on Ubuntu 24 inside a VM.

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![Node 20](https://img.shields.io/badge/Node-20-green?logo=node.js)](https://nodejs.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-orange)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-teal)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Agentic AI OS                                │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   React UI   │◄──►│  FastAPI +   │◄──►│   Agent Orchestrator │  │
│  │  (Electron)  │    │  WebSocket   │    │     (LangGraph)      │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                      │              │
│              ┌───────────────────────────────────────┤              │
│              │                                       │              │
│    ┌─────────▼──────┐   ┌──────────────┐   ┌───────▼────────────┐ │
│    │   Guardrails   │   │   Memory     │   │    Agent Modules   │ │
│    │  (Safety +     │   │  (Redis +    │   │  ┌──────────────┐  │ │
│    │  Permissions)  │   │   ChromaDB)  │   │  │   Planner    │  │ │
│    └────────────────┘   └──────────────┘   │  │   Executor   │  │ │
│                                            │  │   File I/O   │  │ │
│    ┌────────────────┐   ┌──────────────┐   │  │   Web        │  │ │
│    │  MCP Tools     │   │   Sandbox    │   │  │   System     │  │ │
│    │  (Integrations)│   │  (Docker)    │   │  │   Code       │  │ │
│    └────────────────┘   └──────────────┘   │  └──────────────┘  │ │
│                                            └────────────────────┘ │
│                                                                     │
│    ┌─────────────────────────────────────────────────────────────┐ │
│    │                 Ollama LLM Backend                          │ │
│    │   (Llama3.3, Mistral, CodeLlama, Phi-3, Gemma3 models)     │ │
│    └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agentic-os/
├── core/               # LangGraph orchestration engine
│   ├── orchestrator.py     # Main agent graph builder
│   ├── graph_nodes.py      # LangGraph node definitions
│   ├── state.py            # Shared AgentState definition
│   ├── router.py           # Intent routing logic
│   └── logging_config.py   # Centralized logging setup
│
├── guardrails/         # Safety, permissions & policy layer
│   ├── guardian.py         # Central guardrail enforcement
│   ├── permission_engine.py# RBAC + capability checks
│   ├── content_filter.py   # Harmful content detection
│   └── audit_logger.py     # Tamper-proof audit trail
│
├── memory/             # Persistence & retrieval layer
│   ├── redis_store.py      # Short-term working memory
│   ├── chroma_store.py     # Long-term vector memory
│   ├── memory_manager.py   # Unified memory interface
│   └── embeddings.py       # Embedding pipeline
│
├── agents/             # Individual agent modules
│   ├── planner/            # Task decomposition & planning
│   ├── executor/           # Action execution orchestrator
│   ├── file/               # File system operations
│   ├── web/                # Web browsing & scraping
│   ├── system/             # OS-level commands
│   └── code/               # Code generation & execution
│
├── api/                # FastAPI backend
│   ├── main.py             # App entrypoint
│   ├── routes/             # REST endpoints
│   ├── websocket/          # Real-time WebSocket handler
│   └── middleware/         # Auth, CORS, rate-limiting
│
├── ui/                 # React + Electron frontend
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   └── store/
│   └── electron/
│
├── tools/              # MCP tool integrations
│   ├── mcp_server.py       # MCP server host
│   ├── browser_tool.py     # Playwright browser tool
│   ├── search_tool.py      # SerpAPI / DuckDuckGo
│   └── code_tool.py        # Code interpreter tool
│
├── sandbox/            # Docker-based execution environment
│   ├── Dockerfile.sandbox
│   ├── sandbox_manager.py
│   └── resource_limits.py
│
├── config/             # YAML configuration files
│   ├── agents.yaml         # Agent capability definitions
│   ├── permissions.yaml    # RBAC permission matrix
│   └── guardrails.yaml     # Safety policy rules
│
├── docker-compose.yml  # Full stack orchestration
├── requirements.txt    # Python dependencies
├── package.json        # Node.js dependencies
├── install.sh          # Ubuntu 24 VM bootstrap script
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Ubuntu 24.04 LTS (VM or bare metal)
- 16GB RAM minimum (32GB recommended)
- 50GB disk space
- GPU optional (for faster Ollama inference)

### 1. Run the Installer
```bash
chmod +x install.sh
sudo ./install.sh
```

This installs: Python 3.12, Node 20, Redis 7, Docker, Ollama, ChromaDB, and all dependencies.

### 2. Start the Stack
```bash
docker-compose up -d
```

### 3. Launch the API
```bash
cd agentic-os
source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Launch the UI
```bash
cd ui
npm run dev         # Web dev mode
npm run electron    # Desktop app mode
```

---

## ⚙️ Agent Modules

| Agent | Role | Tools |
|-------|------|-------|
| **Planner** | Decomposes goals into task trees | LLM reasoning, memory recall |
| **Executor** | Dispatches tasks to specialist agents | Tool router, state tracker |
| **File Agent** | Read/write/search filesystem | Sandboxed FS access |
| **Web Agent** | Browsing, scraping, research | Playwright, SerpAPI |
| **System Agent** | OS commands, process management | Sandboxed shell, Docker |
| **Code Agent** | Generate, lint, test, execute code | Code interpreter sandbox |

---

## 🛡️ Guardrails & Safety

- **Permission Engine**: RBAC with per-agent capability scopes
- **Content Filter**: Blocks harmful prompts & outputs using policy rules
- **Sandbox Isolation**: All code/system operations run inside Docker containers
- **Audit Logging**: Tamper-evident logs of every agent action
- **Rate Limiting**: Per-user and per-agent request throttling

---

## 🧠 Memory Architecture

```
Short-Term (Redis)         Long-Term (ChromaDB)
├── Active session         ├── Episodic memory
├── Working context        ├── Semantic knowledge base
├── Task queue state       ├── User preferences
└── Tool call cache        └── Agent learning store
```

---

## 🔧 Configuration

Edit `config/agents.yaml` to define agent capabilities:
```yaml
agents:
  planner:
    model: llama3.3
    max_tokens: 4096
    tools: [memory_read, task_decompose]
```

Edit `config/permissions.yaml` for RBAC:
```yaml
roles:
  admin:
    capabilities: [file_write, shell_exec, docker_run]
  user:
    capabilities: [file_read, web_browse, code_generate]
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/agent/run` | Submit a new agent task |
| `GET` | `/api/v1/agent/status/{task_id}` | Get task status |
| `WS` | `/ws/stream/{session_id}` | Real-time agent stream |
| `GET` | `/api/v1/memory/query` | Query vector memory |
| `POST` | `/api/v1/tools/execute` | Direct tool invocation |

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-agent`
3. Commit with conventional commits: `git commit -m "feat: add new agent"`
4. Open a PR

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
