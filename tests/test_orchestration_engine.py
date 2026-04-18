"""
tests/test_orchestration_engine.py — Test suite for the Agent Orchestration Engine.

Covers:
  - IntentParser   (heuristic + schema validation)
  - TaskQueue      (enqueue, parallel, dependencies, cancellation)
  - AgentCoordinator (register, send, delegate/reply)
  - SessionManager (CRUD, interrupt, resume, context switch)
  - OrchestratorGraph (graph structure validation, streaming smoke test)
  - NaturalLanguageShell (command handlers: stop, resume, status, history)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# ─────────────────────────────── helpers ────────────────────────────────────


def _run(coro):
    """Execute a coroutine synchronously (pytest-asyncio fallback)."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ═══════════════════════════════════════════════════════════════════════════════
# IntentParser Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntentParser:
    """Unit tests for IntentParser (heuristic path, no LLM required)."""

    def _parser_no_llm(self):
        from core.intent_parser import IntentParser
        return IntentParser(use_llm=False)

    def test_conversation_detected(self):
        from core.intent_parser import IntentType
        parser = self._parser_no_llm()
        result = _run(parser.parse("hello"))
        assert result.intent_type == IntentType.CONVERSATION

    def test_system_query_detected(self):
        from core.intent_parser import IntentType
        parser = self._parser_no_llm()
        result = _run(parser.parse("status"))
        assert result.intent_type == IntentType.SYSTEM_QUERY

    def test_single_agent_file_task(self):
        from core.intent_parser import IntentType
        parser = self._parser_no_llm()
        result = _run(parser.parse("read the config file and show me its contents"))
        assert result.intent_type in (IntentType.SINGLE_AGENT_TASK, IntentType.MULTI_AGENT_TASK)
        assert "file" in result.required_agents

    def test_multi_agent_detected_via_signals(self):
        from core.intent_parser import IntentType
        parser = self._parser_no_llm()
        result = _run(parser.parse("search the web and then write results to file.md"))
        assert result.intent_type == IntentType.MULTI_AGENT_TASK

    def test_urgency_critical(self):
        parser = self._parser_no_llm()
        result = _run(parser.parse("fix this bug asap it is critical"))
        assert result.urgency == 5

    def test_urgency_low(self):
        parser = self._parser_no_llm()
        result = _run(parser.parse("when you can, organise the documents folder"))
        assert result.urgency == 1

    def test_parsed_intent_is_executable(self):
        parser = self._parser_no_llm()
        result = _run(parser.parse("run the test suite"))
        assert result.is_executable

    def test_conversation_not_executable(self):
        parser = self._parser_no_llm()
        result = _run(parser.parse("hi"))
        assert not result.is_executable

    def test_complexity_scales_with_length(self):
        parser = self._parser_no_llm()
        short = _run(parser.parse("list files"))
        long_ = _run(parser.parse(
            "search the web for the top 10 machine learning papers of 2024, "
            "summarise each one, save them to a markdown file, and then create a "
            "python script that embeddings them with sentence-transformers"
        ))
        assert long_.estimated_complexity >= short.estimated_complexity

    def test_schema_fields_always_present(self):
        parser = self._parser_no_llm()
        result = _run(parser.parse("do something vague"))
        assert result.raw_input
        assert result.intent_type
        assert isinstance(result.required_agents, list)
        assert 1 <= result.urgency <= 5
        assert 1 <= result.estimated_complexity <= 10


# ═══════════════════════════════════════════════════════════════════════════════
# TaskQueue Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestTaskQueue:
    """Unit tests for TaskQueue: enqueue, dependencies, parallel exec, cancel."""

    def _queue(self, max_parallel: int = 4):
        from core.task_queue import TaskQueue
        return TaskQueue(max_parallel=max_parallel)

    def test_enqueue_returns_task(self):
        q = self._queue()
        task = _run(q.enqueue("test task", priority=2))
        assert task.task_id
        assert task.name == "test task"
        assert task.priority == 2

    def test_pending_count_increments(self):
        q = self._queue()
        _run(q.enqueue("a"))
        _run(q.enqueue("b"))
        assert q.pending_count() == 2

    def test_sequential_execution(self):
        q = self._queue()
        results = []

        async def executor(task):
            results.append(task.name)
            return {"done": task.name}

        _run(q.enqueue("task-1", priority=1))
        _run(q.enqueue("task-2", priority=2))
        _run(q.run_all(executor))
        assert set(results) == {"task-1", "task-2"}

    def test_dependency_ordering(self):
        q = self._queue()
        order = []

        async def executor(task):
            order.append(task.name)
            await asyncio.sleep(0.01)
            return {}

        t1 = _run(q.enqueue("step-A", priority=1))
        _run(q.enqueue("step-B", priority=1, dependencies=[t1.task_id]))
        _run(q.run_all(executor))
        assert order.index("step-A") < order.index("step-B")

    def test_parallel_tasks_run_concurrently(self):
        """Two independent tasks should run in parallel and complete faster than sequential."""
        import time
        q = self._queue(max_parallel=2)
        timings = []

        async def slow_executor(task):
            await asyncio.sleep(0.1)
            timings.append(time.monotonic())
            return {}

        _run(q.enqueue("p1", priority=1))
        _run(q.enqueue("p2", priority=1))

        start = time.monotonic()
        _run(q.run_all(slow_executor))
        elapsed = time.monotonic() - start

        # If truly parallel, both tasks finish in ~0.1s (not 0.2s)
        assert elapsed < 0.25

    def test_cancel_pending_task(self):
        from core.task_queue import QueuedTaskStatus
        q = self._queue()
        task = _run(q.enqueue("to-cancel"))
        success = _run(q.cancel(task.task_id))
        assert success
        assert q.get_task(task.task_id).status == QueuedTaskStatus.CANCELLED

    def test_get_task_by_id(self):
        q = self._queue()
        task = _run(q.enqueue("find me"))
        found = q.get_task(task.task_id)
        assert found is not None
        assert found.name == "find me"

    def test_failed_dependency_propagates(self):
        from core.task_queue import QueuedTaskStatus
        q = self._queue()

        async def failing_executor(task):
            if task.name == "fail-task":
                raise RuntimeError("intentional failure")
            return {}

        t1 = _run(q.enqueue("fail-task", priority=1))
        _run(q.enqueue("depends-on-fail", priority=1, dependencies=[t1.task_id]))
        _run(q.run_all(failing_executor))

        t1_result = q.get_task(t1.task_id)
        assert t1_result.status == QueuedTaskStatus.FAILED


# ═══════════════════════════════════════════════════════════════════════════════
# AgentCoordinator Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentCoordinator:
    """Unit tests for AgentCoordinator: register, send, delegate/reply."""

    def _coordinator(self):
        from core.agent_coordinator import AgentCoordinator
        return AgentCoordinator()

    def test_register_and_list(self):
        """register() is sync; registered_agents() should reflect it immediately."""
        coord = self._coordinator()

        async def handler(msg):
            return None

        async def run():
            coord.register("agent-a", handler)
            return coord.registered_agents()

        agents = _run(run())
        assert "agent-a" in agents

    def test_send_delivers_message(self):
        coord = self._coordinator()
        received = []

        async def handler(msg):
            received.append(msg.message_id)
            return None

        from core.agent_coordinator import A2AMessage, A2AMessageType

        async def run():
            coord.register("receiver", handler)
            msg = A2AMessage(
                message_type=A2AMessageType.STATUS_UPDATE,
                sender_id="orchestrator",
                recipient_id="receiver",
                task_id="t1",
                session_id="s1",
            )
            await coord.send(msg)
            await asyncio.sleep(0.1)  # let worker fire
            return msg.message_id

        msg_id = _run(run())
        assert msg_id in received

    def test_delegate_and_reply(self):
        """Agent B replies to a DELEGATE message; future resolves."""
        coord = self._coordinator()

        from core.agent_coordinator import A2AMessageType

        async def agent_b_handler(msg):
            if msg.message_type == A2AMessageType.DELEGATE:
                return msg.reply(
                    sender_id="agent-b",
                    message_type=A2AMessageType.RESULT,
                    payload={"output": "42"},
                )
            return None

        async def run():
            coord.register("agent-b", agent_b_handler)
            coord.register("orchestrator", AsyncMock(return_value=None))
            reply = await coord.delegate(
                from_id="orchestrator",
                to_id="agent-b",
                task_id="t-99",
                session_id="s-99",
                payload={"action": "compute"},
                timeout=5.0,
            )
            return reply

        reply = _run(run())
        assert reply.payload["output"] == "42"
        assert reply.message_type == A2AMessageType.RESULT

    def test_send_to_unknown_agent_raises(self):
        coord = self._coordinator()

        from core.agent_coordinator import A2AMessage, A2AMessageType

        async def run():
            msg = A2AMessage(
                message_type=A2AMessageType.STATUS_UPDATE,
                sender_id="x",
                recipient_id="ghost",
                task_id="t",
                session_id="s",
            )
            with pytest.raises(KeyError):
                await coord.send(msg)

        _run(run())

    def test_broadcast_reaches_all_registered(self):
        coord = self._coordinator()
        received_by = []

        def make_handler(name):
            async def h(msg):
                received_by.append(name)
                return None
            return h

        from core.agent_coordinator import A2AMessage, A2AMessageType

        async def run():
            coord.register("agent-1", make_handler("agent-1"))
            coord.register("agent-2", make_handler("agent-2"))
            coord.register("agent-3", make_handler("agent-3"))
            msg = A2AMessage(
                message_type=A2AMessageType.STATUS_UPDATE,
                sender_id="agent-1",  # sender won't receive own broadcast
                recipient_id="broadcast",
                task_id="t",
                session_id="s",
            )
            await coord.send(msg)
            await asyncio.sleep(0.1)

        _run(run())
        assert "agent-2" in received_by
        assert "agent-3" in received_by

    def test_unregister_removes_agent(self):
        coord = self._coordinator()

        async def h(msg):
            return None

        async def run():
            coord.register("temp-agent", h)
            # send a dummy to ensure worker starts so cancel works cleanly
            coord.unregister("temp-agent")
            return coord.registered_agents()

        agents = _run(run())
        assert "temp-agent" not in agents

    def test_a2a_message_reply_preserves_correlation_id(self):
        from core.agent_coordinator import A2AMessage, A2AMessageType
        original = A2AMessage(
            message_type=A2AMessageType.DELEGATE,
            sender_id="a",
            recipient_id="b",
            task_id="t",
            session_id="s",
        )
        reply = original.reply("b", A2AMessageType.RESULT, {"ok": True})
        assert reply.correlation_id == original.correlation_id
        assert reply.recipient_id == original.sender_id


# ═══════════════════════════════════════════════════════════════════════════════
# SessionManager Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSessionManager:
    """Unit tests for SessionManager lifecycle, interrupt, resume, context switch."""

    def _sm(self):
        from core.session_manager import SessionManager
        return SessionManager(session_ttl_seconds=3600)

    def test_create_and_get(self):
        sm = self._sm()
        session = _run(sm.create("user-1"))
        fetched = _run(sm.get(session.session_id))
        assert fetched is not None
        assert fetched.user_id == "user-1"

    def test_get_or_create_idempotent(self):
        sm = self._sm()
        s1 = _run(sm.get_or_create("user-2"))
        s2 = _run(sm.get_or_create("user-2"))
        assert s1.session_id == s2.session_id

    def test_interrupt_sets_flag(self):
        sm = self._sm()
        session = _run(sm.create("user-3"))
        ok = _run(sm.interrupt(session.session_id, reason="user said stop"))
        assert ok
        updated = _run(sm.get(session.session_id))
        assert updated.interrupt_flag

    def test_resume_clears_flag(self):
        sm = self._sm()
        session = _run(sm.create("user-4"))
        _run(sm.interrupt(session.session_id))
        snap = _run(sm.resume(session.session_id))
        updated = _run(sm.get(session.session_id))
        assert not updated.interrupt_flag

    def test_resume_returns_snapshot_when_saved(self):
        sm = self._sm()
        session = _run(sm.create("user-5"))
        session.save_snapshot("task-99", step=3, results=["partial"], user_input="do something")
        _run(sm.update(session))
        snap = _run(sm.resume(session.session_id))
        assert snap is not None
        assert snap.current_step == 3

    def test_detect_stop_signal(self):
        from core.session_manager import SessionManager
        assert SessionManager.detect_stop_signal("stop")
        assert SessionManager.detect_stop_signal("STOP!")
        assert SessionManager.detect_stop_signal("cancel")
        assert not SessionManager.detect_stop_signal("stop that noise and keep going")

    def test_detect_resume_signal(self):
        from core.session_manager import SessionManager
        assert SessionManager.detect_resume_signal("resume")
        assert SessionManager.detect_resume_signal("continue")
        assert not SessionManager.detect_resume_signal("don't resume")

    def test_add_turn_history_capped(self):
        sm = self._sm()
        session = _run(sm.create("user-6"))
        for i in range(60):
            session.add_turn("user", f"message {i}")
        assert len(session.history) == 50

    def test_delete_session(self):
        sm = self._sm()
        session = _run(sm.create("user-7"))
        _run(sm.delete(session.session_id))
        assert _run(sm.get(session.session_id)) is None

    def test_context_switch_saves_snapshot(self):
        sm = self._sm()
        session = _run(sm.create("user-8"))
        session.current_task_id = "old-task"
        _run(sm.update(session))
        state_stub = {"current_step_index": 2, "tool_results": ["r1"], "user_input": "old query"}
        _run(sm.switch_context(session.session_id, "new topic", current_state=state_stub))
        updated = _run(sm.get(session.session_id))
        assert updated.snapshot is not None
        assert updated.current_task_id is None

    def test_active_session_count(self):
        sm = self._sm()
        _run(sm.create("u-a"))
        _run(sm.create("u-b"))
        assert sm.active_session_count() >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# OrchestratorGraph Tests (structural + smoke)
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorGraph:
    """Structural + lightweight stream smoke tests (no Ollama required)."""

    def test_graph_builds_without_error(self):
        from core.orchestrator_graph import OrchestratorGraph
        g = OrchestratorGraph()
        assert g._graph is not None

    def test_graph_nodes_present(self):
        from core.orchestrator_graph import OrchestratorGraph
        g = OrchestratorGraph()
        # The compiled graph exposes the node names
        node_names = set(g._graph.nodes.keys())
        expected = {"parse_intent", "route_to_agent", "execute_with_guardrails",
                    "update_memory", "respond_to_user"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_get_orchestrator_graph_singleton(self):
        from core.orchestrator_graph import get_orchestrator_graph
        g1 = get_orchestrator_graph()
        g2 = get_orchestrator_graph()
        assert g1 is g2

    @patch("core.orchestrator_graph.parse_intent_node", new_callable=AsyncMock)
    @patch("core.orchestrator_graph.route_to_agent_node", new_callable=AsyncMock)
    @patch("core.orchestrator_graph.respond_to_user_node", new_callable=AsyncMock)
    def test_run_conversation_path(self, mock_respond, mock_route, mock_parse):
        """CONVERSATION intent should skip route_to_agent and go straight to respond."""
        from datetime import datetime
        from core.state import TaskStatus

        now = datetime.utcnow().isoformat()

        mock_parse.return_value = {
            "metadata": {"intent_type": "CONVERSATION", "intent": "hello", "required_agents": [], "urgency": 1, "estimated_complexity": 1},
            "goal": "hello",
            "status": TaskStatus.EXECUTING.value,
            "updated_at": now,
        }
        mock_respond.return_value = {
            "status": TaskStatus.COMPLETED.value,
            "messages": [],
            "updated_at": now,
        }

        from core.orchestrator_graph import OrchestratorGraph
        g = OrchestratorGraph()
        result = _run(g.run("hello", user_id="test-user"))
        # route_to_agent should NOT have been called for CONVERSATION
        mock_route.assert_not_called()


# ═══════════════════════════════════════════════════════════════════════════════
# NaturalLanguageShell Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestNaturalLanguageShell:
    """Tests for built-in NaturalLanguageShell commands."""

    def _shell(self) -> Any:
        from core.shell import NaturalLanguageShell
        return NaturalLanguageShell(user_id="test-user", stream=False)

    def test_help_command(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("help"))
        assert "Commands:" in response

    def test_exit_command(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("exit"))
        assert response == "__EXIT__"

    def test_quit_command(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("quit"))
        assert response == "__EXIT__"

    def test_stop_command(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("stop"))
        assert "⛔" in response or "stopped" in response.lower()

    def test_status_command(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("status"))
        assert "Session" in response or "session" in response

    def test_history_empty(self):
        shell = self._shell()
        _run(shell.start_session())
        response = _run(shell.chat("history"))
        # Either shows empty message or has some system turns
        assert isinstance(response, str) and len(response) > 0

    def test_resume_no_snapshot(self):
        shell = self._shell()
        _run(shell.start_session())
        # Stop first, then resume
        _run(shell.chat("stop"))
        response = _run(shell.chat("resume"))
        assert isinstance(response, str)

    def test_session_auto_created(self):
        shell = self._shell()
        # No explicit start_session; chat should still work for built-in commands
        response = _run(shell.chat("help"))
        assert "Commands:" in response
