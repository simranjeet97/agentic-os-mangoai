"""
api/main.py — FastAPI application entry point for the Agentic AI OS.
Includes REST endpoints, WebSocket streaming, middleware, and health checks.
"""

from __future__ import annotations

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

from api.routes import agent_router, memory_router, tools_router
from api.websocket.handler import ws_router
from core.logging_config import setup_logging
from core.orchestrator import get_orchestrator

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

    # Warm up the orchestrator (loads LLM, connects services)
    try:
        get_orchestrator()
        logger.info("Orchestrator initialized")
    except Exception as exc:
        logger.warning("Orchestrator warmup failed", error=str(exc))

    yield

    logger.info("Agentic AI OS API shutting down")


# ── App Factory ───────────────────────────────────────────────────────────────
def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Agentic AI OS",
        description=(
            "Production-grade multi-agent AI operating system API. "
            "Powered by LangGraph, Ollama, Redis, and ChromaDB."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins = os.getenv("CORS_ORIGINS", '["http://localhost:3000","http://localhost:5173"]')
    import json
    try:
        origins = json.loads(cors_origins)
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
        logger.error("Unhandled exception", path=request.url.path, error=str(exc), exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(agent_router, prefix="/api/v1/agent", tags=["Agent"])
    app.include_router(memory_router, prefix="/api/v1/memory", tags=["Memory"])
    app.include_router(tools_router, prefix="/api/v1/tools", tags=["Tools"])
    app.include_router(ws_router, tags=["WebSocket"])

    # ── Prometheus Metrics Endpoint ───────────────────────────────────────────
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # ── Health Check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health_check():
        """System health status — checked by Docker and load balancers."""
        from memory.redis_store import RedisStore
        from memory.chroma_store import ChromaStore
        import httpx

        redis_ok = await RedisStore().ping()
        chroma_ok = await ChromaStore().ping()

        ollama_ok = False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/tags")
                ollama_ok = r.status_code == 200
        except Exception:
            pass

        status = "healthy" if (redis_ok and chroma_ok) else "degraded"

        return {
            "status": status,
            "version": app.version,
            "services": {
                "redis": "up" if redis_ok else "down",
                "chromadb": "up" if chroma_ok else "down",
                "ollama": "up" if ollama_ok else "down",
            },
        }

    @app.get("/", tags=["Root"], include_in_schema=False)
    async def root():
        return {"message": "Agentic AI OS API", "docs": "/docs"}

    return app


app = create_app()
