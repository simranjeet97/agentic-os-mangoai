"""
core/logging_config.py — Centralized structured logging configuration
for the Agentic AI OS. Uses structlog + loguru for rich, contextual logs.
"""

import logging
import sys
import os
from pathlib import Path
from typing import Any

import structlog
from loguru import logger

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging() -> None:
    """Configure structlog + loguru for the entire application."""

    # ── Loguru setup ──────────────────────────────────────────────────────────
    logger.remove()  # Remove default handler

    # Console handler — colored, human-readable
    logger.add(
        sys.stdout,
        level=LOG_LEVEL,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler — JSON structured logs for production
    logger.add(
        LOG_DIR / "agentic-os.log",
        level=LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation="100 MB",
        retention="30 days",
        compression="gz",
        serialize=True,  # JSON output
        enqueue=True,    # Thread-safe async logging
    )

    # Audit log — agent actions only (never rotated away early)
    logger.add(
        LOG_DIR / "audit.log",
        level="INFO",
        filter=lambda record: "audit" in record["extra"],
        rotation="500 MB",
        retention="90 days",
        compression="gz",
        serialize=True,
        enqueue=True,
    )

    # ── structlog setup ───────────────────────────────────────────────────────
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(LOG_LEVEL)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Redirect stdlib logging through loguru
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    logger.info(
        "Logging initialized",
        log_level=LOG_LEVEL,
        log_dir=str(LOG_DIR),
    )


class _InterceptHandler(logging.Handler):
    """Bridge stdlib logging → loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def get_logger(name: str) -> Any:
    """Return a bound loguru logger for a module."""
    return logger.bind(module=name)


def get_struct_logger(name: str) -> Any:
    """Return a structlog logger for structured/contextual logging."""
    return structlog.get_logger(name)
