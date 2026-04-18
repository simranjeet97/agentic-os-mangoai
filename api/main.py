"""
api/main.py — FastAPI application entry point for the Agentic AI OS.

Changes from v1:
  - Lifespan warms up OrchestratorGraph (not the old AgentOrchestrator)
  - Session routes mounted at /api/v1/sessions
  - All streaming now flows through OrchestratorGraph.stream()
  - Metrics, CORS, GZip, global error handler unchanged
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, make_asgi_app

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.routes import agent_router, memory_router, tools_router, session_router, graph_router
from api.routes import tasks_router, agents_status_router, approval_router, audit_router, undo_router, chat_router
from api.auth import auth_router
from api.websocket.handler import ws_router
from core.logging_config import setup_logging
from core.orchestrator_graph import get_orchestrator_graph

limiter = Limiter(key_func=get_remote_address, default_limits=["10/minute"])


logger = structlog.get_logger(__name__)

# ── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "agentic_api_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "agentic_api_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application startup and shutdown hooks."""
    setup_logging()
    logger.info("Agentic AI OS API starting up", version=app.version)

    # Warm up OrchestratorGraph (compiles LangGraph, initialises singletons)
    try:
        get_orchestrator_graph()
        logger.info("OrchestratorGraph initialised")
    except Exception as exc:
        logger.warning("OrchestratorGraph warmup failed", error=str(exc))

    yield

    logger.info("Agentic AI OS API shutting down")


# ── App Factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic AI OS",
        description=(
            "Production-grade multi-agent AI operating system API. "
            "LangGraph orchestration · LLM intent parsing · streaming SSE · "
            "Redis + ChromaDB memory · per-user session management."
        ),
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={
            "name": "Agentic AI Team",
        },
        license_info={
            "name": "MIT"
        },
        lifespan=lifespan,
    )
    
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_env = os.getenv("CORS_ORIGINS", '["http://localhost:3000","http://localhost:5173"]')
    try:
        origins = json.loads(cors_env)
    except Exception:
        origins = ["http://localhost:3000", "http://localhost:5173"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # ── Metrics Middleware ────────────────────────────────────────────────────
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        endpoint = request.url.path
        REQUEST_COUNT.labels(request.method, endpoint, response.status_code).inc()
        REQUEST_LATENCY.labels(request.method, endpoint).observe(duration)
        response.headers["X-Response-Time"] = f"{duration:.4f}s"
        return response

    # ── Global Exception Handler ──────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "Unhandled exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(agent_router,   prefix="/api/v1/agent",    tags=["Agent"])
    app.include_router(memory_router,  prefix="/api/v1/memory",   tags=["Memory"])
    app.include_router(tools_router,   prefix="/api/v1/tools",    tags=["Tools"])
    app.include_router(session_router, prefix="/api/v1/sessions", tags=["Sessions"])
    app.include_router(graph_router,   prefix="/api/v1/graph",    tags=["Graph"])
    
    # New Routers
    app.include_router(auth_router,          prefix="/api/v1/auth",          tags=["Auth"])
    app.include_router(chat_router,          prefix="/api/v1/chat",          tags=["Chat"])
    app.include_router(tasks_router,         prefix="/api/v1/tasks",         tags=["Tasks"])
    app.include_router(agents_status_router, prefix="/api/v1/agents",        tags=["Agents"])
    app.include_router(approval_router,      prefix="/api/v1/approve",       tags=["Approval"])
    app.include_router(audit_router,         prefix="/api/v1/audit",         tags=["Audit Log"])
    app.include_router(undo_router,          prefix="/api/v1/undo",          tags=["Undo"])
    
    app.include_router(ws_router,      tags=["WebSocket"])

    # ── Prometheus ────────────────────────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        """System health — checked by Docker and load balancers."""
        import httpx

        redis_ok = False
        chroma_ok = False
        ollama_ok = False

        try:
            from memory.redis_store import RedisStore
            redis_ok = await RedisStore().ping()
        except Exception:
            pass

        try:
            from memory.chroma_store import ChromaStore
            chroma_ok = await ChromaStore().ping()
        except Exception:
            pass

        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(
                    f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags"
                )
                ollama_ok = r.status_code == 200
        except Exception:
            pass

        # Orchestrator always reports up if compiled
        orch_ok = True
        try:
            get_orchestrator_graph()
        except Exception:
            orch_ok = False

        all_critical = redis_ok and chroma_ok and orch_ok
        overall = "healthy" if all_critical else "degraded"

        return {
            "status": overall,
            "version": app.version,
            "services": {
                "redis":        "up" if redis_ok else "down",
                "chromadb":     "up" if chroma_ok else "down",
                "ollama":       "up" if ollama_ok else "down",
                "orchestrator": "up" if orch_ok else "down",
            },
        }

    # ── Root ──────────────────────────────────────────────────────────────────
    @app.get("/", tags=["Root"], include_in_schema=False)
    async def root():
        return {
            "name": "Agentic AI OS",
            "version": app.version,
            "docs": "/docs",
            "health": "/health",
            "websocket": "ws://host/ws/stream/{session_id}",
        }

    return app


app = create_app()
