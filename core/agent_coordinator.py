"""
core/agent_coordinator.py — Agent-to-Agent (A2A) communication layer.

Responsibilities:
  - Broadcast and point-to-point messaging between agents
  - A2A message format with correlation IDs for request/reply pairing
  - Delegation: agent A asks agent B to complete a sub-task
  - In-process mailbox (asyncio.Queue per agent); swap for Redis pub/sub in prod
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from pydantic import BaseModel, Field

from core.logging_config import get_logger

logger = get_logger(__name__)


# ── A2A Message format ────────────────────────────────────────────────────────


class A2AMessageType(str, Enum):
    DELEGATE = "DELEGATE"          # agent A → agent B: "do this sub-task"
    RESULT = "RESULT"              # agent B → agent A: "here's what I found"
    STATUS_UPDATE = "STATUS_UPDATE"  # progress notifications
    CANCEL = "CANCEL"              # abort a delegated task
    HEARTBEAT = "HEARTBEAT"        # liveness ping
    ERROR = "ERROR"                # error report from a delegatee


class A2AMessage(BaseModel):
    """
    Agent-to-Agent message envelope.
    All fields are serialisable so messages can be forwarded over Redis/NATS later.
    """

    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Links request/reply pairs",
    )
    message_type: A2AMessageType
    sender_id: str
    recipient_id: str               # agent_id or "broadcast"
    task_id: str
    session_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    ttl_seconds: int = 300          # Message expires after this many seconds

    def is_expired(self) -> bool:
        age = (datetime.utcnow() - self.timestamp).total_seconds()
        return age > self.ttl_seconds

    def reply(
        self,
        sender_id: str,
        message_type: A2AMessageType,
        payload: dict[str, Any],
    ) -> "A2AMessage":
        """Create a correlated reply to this message."""
        return A2AMessage(
            correlation_id=self.correlation_id,  # same correlation_id
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=self.sender_id,
            task_id=self.task_id,
            session_id=self.session_id,
            payload=payload,
        )


# ── Handler type ──────────────────────────────────────────────────────────────

MessageHandler = Callable[[A2AMessage], Coroutine[Any, Any, Optional[A2AMessage]]]


# ── AgentCoordinator ──────────────────────────────────────────────────────────


class AgentCoordinator:
    """
    Central message bus for agent-to-agent communication.

    Each agent registers with a unique agent_id and supplies a handler coroutine.
    Messages are delivered via per-agent asyncio queues (in-process mailboxes).

    Usage:
        coordinator = AgentCoordinator()
        coordinator.register("file-agent-01", file_agent_handler)
        coordinator.register("web-agent-02",  web_agent_handler)

        # Delegate a subtask from agent A to agent B
        reply = await coordinator.delegate(
            from_id="orchestrator",
            to_id="web-agent-02",
            task_id="t-123",
            session_id="s-abc",
            payload={"action": "search", "query": "LangGraph tutorials"},
        )
    """

    def __init__(self) -> None:
        self._mailboxes: dict[str, asyncio.Queue[A2AMessage]] = {}
        self._handlers: dict[str, MessageHandler] = {}
        self._pending_replies: dict[str, asyncio.Future[A2AMessage]] = {}  # corr_id → future
        self._running_workers: dict[str, asyncio.Task] = {}               # agent_id → task
        self._lock = asyncio.Lock()

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, agent_id: str, handler: MessageHandler) -> None:
        """
        Register an agent and queue its message-processing worker.
        The asyncio.Task is started lazily the first time a message is sent
        (so register() is safe to call outside a running event loop).
        """
        if agent_id in self._mailboxes:
            logger.warning("Agent already registered", agent_id=agent_id)
            return
        self._mailboxes[agent_id] = asyncio.Queue()
        self._handlers[agent_id] = handler
        # Worker will be started lazily in _ensure_worker()
        logger.info("Agent registered with coordinator", agent_id=agent_id)

    def unregister(self, agent_id: str) -> None:
        """Deregister an agent and clean up its worker."""
        worker = self._running_workers.pop(agent_id, None)
        if worker:
            worker.cancel()
        self._mailboxes.pop(agent_id, None)
        self._handlers.pop(agent_id, None)
        logger.info("Agent unregistered", agent_id=agent_id)

    def registered_agents(self) -> list[str]:
        return list(self._mailboxes.keys())

    # ── Sending ───────────────────────────────────────────────────────────────

    def _ensure_worker(self, agent_id: str) -> None:
        """Start the worker task for agent_id if not already running."""
        if agent_id not in self._running_workers or self._running_workers[agent_id].done():
            task = asyncio.create_task(
                self._worker(agent_id), name=f"a2a-worker-{agent_id}"
            )
            self._running_workers[agent_id] = task

    async def send(self, message: A2AMessage) -> None:
        """
        Deliver a message to the recipient's mailbox (or everyone if 'broadcast').
        Workers are started lazily here, inside a running event loop.
        Fire-and-forget — use `delegate` if you need a reply.
        """
        if message.recipient_id == "broadcast":
            for agent_id, mailbox in self._mailboxes.items():
                if agent_id != message.sender_id:
                    self._ensure_worker(agent_id)
                    await mailbox.put(message)
            logger.debug("Broadcast sent", sender=message.sender_id)
        else:
            mailbox = self._mailboxes.get(message.recipient_id)
            if mailbox is None:
                logger.error(
                    "Recipient not registered",
                    recipient=message.recipient_id,
                    sender=message.sender_id,
                )
                raise KeyError(f"Agent '{message.recipient_id}' is not registered")
            self._ensure_worker(message.recipient_id)
            await mailbox.put(message)

    async def delegate(
        self,
        from_id: str,
        to_id: str,
        task_id: str,
        session_id: str,
        payload: dict[str, Any],
        timeout: float = 120.0,
    ) -> A2AMessage:
        """
        Send a DELEGATE message and await a correlated RESULT reply.
        Raises asyncio.TimeoutError if the delegatee doesn't respond in time.
        """
        msg = A2AMessage(
            message_type=A2AMessageType.DELEGATE,
            sender_id=from_id,
            recipient_id=to_id,
            task_id=task_id,
            session_id=session_id,
            payload=payload,
        )

        loop = asyncio.get_event_loop()
        future: asyncio.Future[A2AMessage] = loop.create_future()

        async with self._lock:
            self._pending_replies[msg.correlation_id] = future

        try:
            await self.send(msg)
            logger.info(
                "Delegation sent",
                from_id=from_id,
                to_id=to_id,
                task_id=task_id,
                correlation_id=msg.correlation_id,
            )
            reply = await asyncio.wait_for(future, timeout=timeout)
            return reply
        finally:
            async with self._lock:
                self._pending_replies.pop(msg.correlation_id, None)

    async def publish_status(
        self,
        sender_id: str,
        task_id: str,
        session_id: str,
        status: str,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Broadcast a STATUS_UPDATE message to all agents."""
        await self.send(
            A2AMessage(
                message_type=A2AMessageType.STATUS_UPDATE,
                sender_id=sender_id,
                recipient_id="broadcast",
                task_id=task_id,
                session_id=session_id,
                payload={"status": status, **(details or {})},
            )
        )

    # ── Internal worker ───────────────────────────────────────────────────────

    async def _worker(self, agent_id: str) -> None:
        """Per-agent message pump. Runs until the agent is unregistered."""
        mailbox = self._mailboxes[agent_id]
        handler = self._handlers[agent_id]

        logger.debug("A2A worker started", agent_id=agent_id)

        while True:
            try:
                message = await mailbox.get()

                if message.is_expired():
                    logger.warning("Dropping expired message", message_id=message.message_id)
                    mailbox.task_done()
                    continue

                try:
                    reply = await handler(message)
                except Exception as exc:
                    logger.error(
                        "Handler raised exception",
                        agent_id=agent_id,
                        error=str(exc),
                        exc_info=True,
                    )
                    reply = message.reply(
                        sender_id=agent_id,
                        message_type=A2AMessageType.ERROR,
                        payload={"error": str(exc)},
                    )

                # Resolve pending future if this is a correlated reply
                if reply is not None:
                    async with self._lock:
                        future = self._pending_replies.get(reply.correlation_id)
                    if future and not future.done():
                        future.set_result(reply)
                    elif reply.recipient_id != "broadcast" and reply.recipient_id in self._mailboxes:
                        await self.send(reply)

                mailbox.task_done()

            except asyncio.CancelledError:
                logger.debug("A2A worker cancelled", agent_id=agent_id)
                break
            except Exception as exc:
                logger.error("Unexpected error in A2A worker", agent_id=agent_id, error=str(exc))


# ── Singleton ─────────────────────────────────────────────────────────────────

_coordinator: Optional[AgentCoordinator] = None


def get_coordinator() -> AgentCoordinator:
    """Return the global AgentCoordinator singleton."""
    global _coordinator
    if _coordinator is None:
        _coordinator = AgentCoordinator()
    return _coordinator
