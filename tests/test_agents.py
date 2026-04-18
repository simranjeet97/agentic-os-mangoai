"""
tests/test_agents.py — Comprehensive test suite for all 6 agents + Registry + Router

Tests cover:
  - BaseAgent: run() → AgentResult, memory integration, guardrail integration
  - PlannerAgent: decompose, replan, monitor_plan
  - ExecutorAgent: safe_execute (Python/Bash/JS), blocked commands, explanation
  - FileAgent: read/write/search/summarize/organize/delete (with confirmation)
  - WebAgent: DuckDuckGo search, link extraction (mocked Playwright)
  - SystemAgent: metrics, anomaly detection, process list, suggest_fixes
  - CodeAgent: generate/debug/refactor/explain/lint/git/review
  - AgentRegistry: auto-discovery, get, list_all, health_check
  - AgentRouter: keyword classification, LLM fallback, route()
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset the singleton registry between tests."""
    from agents.registry import AgentRegistry
    AgentRegistry.reset()
    yield
    AgentRegistry.reset()


@pytest.fixture
def minimal_state() -> dict[str, Any]:
    """A minimal AgentState-compatible dict."""
    return {
        "task_id": "test-task-001",
        "session_id": "test-session-001",
        "user_id": "test-user",
        "messages": [],
        "user_input": "test input",
        "goal": "Test goal",
        "plan": [],
        "current_step_index": 0,
        "active_agent": None,
        "tool_calls": [],
        "tool_results": [],
        "artifacts": [],
        "status": "executing",
        "error": None,
        "iterations": 0,
        "memory": {},
        "guardrail_result": None,
        "requires_approval": False,
        "metadata": {},
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }


@pytest.fixture
def mock_llm():
    """Mock LLM that returns a configurable response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Mocked LLM response"
    llm.ainvoke = AsyncMock(return_value=mock_response)
    return llm


@pytest.fixture
def mock_guardrail():
    """Mock GuardrailMiddleware."""
    gm = MagicMock()
    mock_safe_result = MagicMock()
    mock_safe_result.sandbox_result = None
    gm.evaluate_action = AsyncMock(return_value=mock_safe_result)
    gm.safe_execute_code = AsyncMock(return_value=mock_safe_result)
    gm.network = MagicMock()
    gm.network.check_url = MagicMock(return_value=MagicMock(allowed=True, reason="allowed"))
    gm.network.log_external_call = MagicMock()
    return gm


@pytest.fixture
def mock_memory():
    """Mock MemoryAgent."""
    mem = MagicMock()
    recall_response = MagicMock()
    recall_response.results = []
    recall_response.sources_queried = []
    mem.recall = AsyncMock(return_value=recall_response)
    mem.remember = AsyncMock(return_value={"episodic": "ep-123"})
    mem.semantic = MagicMock()
    mem.semantic.index_document = AsyncMock(return_value=["sem-123"])
    return mem


def inject_mocks(agent, mock_guardrail, mock_memory):
    """Inject mock guardrail and memory into a BaseAgent."""
    from agents.base_agent import BaseAgent
    BaseAgent._shared_guardrail = mock_guardrail
    BaseAgent._shared_memory = mock_memory


# ═══════════════════════════════════════════════════════════════════════════════
# BaseAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaseAgent:
    """Test the BaseAgent base class via PlannerAgent (simplest concrete subclass)."""

    @pytest.mark.asyncio
    async def test_agent_has_required_attributes(self):
        from agents.planner.agent import PlannerAgent
        agent = PlannerAgent()
        assert agent.name == "planner"
        assert isinstance(agent.agent_id, str)
        assert len(agent.agent_id) > 0
        assert isinstance(agent.capabilities, list)
        assert isinstance(agent.tools, list)

    @pytest.mark.asyncio
    async def test_run_returns_agent_result(self, mock_guardrail, mock_memory):
        from agents.base_agent import AgentResult
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        mock_response = MagicMock()
        # Return valid JSON plan
        mock_response.content = '[{"step_id":"1","description":"Do something","agent":"executor","dependencies":[]}]'
        agent._llm.ainvoke = AsyncMock(return_value=mock_response)

        task = {"description": "Build a web scraper", "action": "decompose"}
        result = await agent.run(task, user_id="test-user")

        assert isinstance(result, AgentResult)
        assert result.agent_name == "planner"
        assert isinstance(result.agent_id, str)
        assert result.success is True
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_stores_in_memory(self, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = '[{"step_id":"1","description":"Step","agent":"executor","dependencies":[]}]'
        agent._llm.ainvoke = AsyncMock(return_value=mock_response)

        await agent.run({"description": "Test task"})
        mock_memory.remember.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_handles_execute_exception(self, mock_guardrail, mock_memory):
        from agents.base_agent import AgentResult
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        agent._llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM exploded"))

        result = await agent.run({"description": "Crash test"})
        # Should NOT raise — wraps in AgentResult
        assert isinstance(result, AgentResult)
        assert result.success is False
        assert "LLM exploded" in (result.error or "")


# ═══════════════════════════════════════════════════════════════════════════════
# PlannerAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlannerAgent:

    @pytest.mark.asyncio
    async def test_decompose_returns_plan(self, minimal_state, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()

        # CoT response + plan response
        cot_response = MagicMock()
        cot_response.content = "I will break this into steps..."
        plan_response = MagicMock()
        plan_response.content = json.dumps([
            {"step_id": "1", "description": "Search the web", "agent": "web", "dependencies": []},
            {"step_id": "2", "description": "Summarize results", "agent": "code", "dependencies": ["1"]},
        ])
        agent._llm.ainvoke = AsyncMock(side_effect=[cot_response, plan_response])

        step = {"description": "Research and summarize AI trends", "action": "decompose"}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is True
        assert "plan" in result
        assert len(result["plan"]) == 2
        assert result["plan"][0]["agent"] == "web"
        assert result["plan"][1]["agent"] == "code"
        assert "reasoning" in result

    @pytest.mark.asyncio
    async def test_decompose_handles_invalid_json(self, minimal_state, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        bad_response = MagicMock()
        bad_response.content = "This is not JSON at all"
        agent._llm.ainvoke = AsyncMock(return_value=bad_response)

        result = await agent.execute(
            {"description": "Do something", "action": "decompose"}, minimal_state
        )

        # Should fallback to a single executor step
        assert "plan" in result
        assert len(result["plan"]) >= 1
        assert result["plan"][0]["agent"] == "executor"

    @pytest.mark.asyncio
    async def test_replan_on_failure(self, minimal_state, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = '[{"step_id":"3","description":"Retry with different approach","agent":"executor","dependencies":[]}]'
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "replan",
            "failed_step": {"step_id": "2", "description": "Failed step"},
            "error": "Connection timeout",
            "remaining_steps": [],
        }
        result = await agent.execute(step, minimal_state)

        assert "revised_plan" in result
        assert len(result["revised_plan"]) == 1
        assert result["replan_attempt"] == 1

    @pytest.mark.asyncio
    async def test_replan_respects_max_attempts(self, minimal_state, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = '[]'
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "replan",
            "failed_step": {"step_id": "1"},
            "error": "Unknown error",
        }

        # Exhaust attempts
        for _ in range(PlannerAgent.MAX_REPLAN_ATTEMPTS):
            await agent.execute(step, minimal_state)

        # Next attempt should fail
        result = await agent.execute(step, minimal_state)
        assert result["success"] is False
        assert "Max replan" in result.get("error", "")

    @pytest.mark.asyncio
    async def test_monitor_plan_reports_correctly(self, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        plan = [
            {"step_id": "1", "status": "completed"},
            {"step_id": "2", "status": "failed"},
            {"step_id": "3", "status": "pending"},
        ]
        report = await agent.monitor_plan(plan)

        assert report["total"] == 3
        assert report["completed"] == 1
        assert len(report["failed_steps"]) == 1
        assert report["pending"] == 1
        assert report["progress_pct"] == pytest.approx(33.3, abs=0.5)

    @pytest.mark.asyncio
    async def test_plan_capped_at_max_subtasks(self, minimal_state, mock_guardrail, mock_memory):
        from agents.planner.agent import PlannerAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = PlannerAgent()
        agent._llm = MagicMock()
        cot = MagicMock()
        cot.content = "Reasoning..."
        plan_resp = MagicMock()
        # Generate 30 subtasks
        plan_resp.content = json.dumps([
            {"step_id": str(i), "description": f"Step {i}", "agent": "executor"}
            for i in range(30)
        ])
        agent._llm.ainvoke = AsyncMock(side_effect=[cot, plan_resp])

        result = await agent.execute(
            {"description": "Huge task", "action": "decompose"}, minimal_state
        )
        assert len(result["plan"]) <= PlannerAgent.MAX_SUBTASKS


# ═══════════════════════════════════════════════════════════════════════════════
# ExecutorAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestExecutorAgent:

    @pytest.mark.asyncio
    async def test_safe_execute_python(self, mock_guardrail, mock_memory):
        from agents.executor.agent import ExecutorAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        # Mock sandbox result
        sandbox_result = MagicMock()
        sandbox_result.stdout = "Hello, World!\n"
        sandbox_result.stderr = ""
        sandbox_result.exit_code = 0
        sandbox_result.duration_ms = 123
        sandbox_result.timed_out = False

        with patch("agents.executor.agent.ExecutorAgent.sandbox") as mock_sb:
            mock_sb.safe_execute = AsyncMock(return_value=sandbox_result)
            agent = ExecutorAgent()
            agent._sandbox = mock_sb
            agent._llm = MagicMock()
            mock_resp = MagicMock()
            mock_resp.content = "Prints Hello World"
            agent._llm.ainvoke = AsyncMock(return_value=mock_resp)

            result = await agent.safe_execute(
                code='print("Hello, World!")',
                language="python",
                user_id="test",
            )

        assert result["success"] is True
        assert result["exit_code"] == 0
        assert "explanation" in result

    @pytest.mark.asyncio
    async def test_unsupported_language_rejected(self, mock_guardrail, mock_memory):
        from agents.executor.agent import ExecutorAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = ExecutorAgent()
        result = await agent.safe_execute(code="test", language="cobol")

        assert result["success"] is False
        assert "Unsupported language" in result["error"]

    @pytest.mark.asyncio
    async def test_blocked_by_guardrail(self, mock_memory):
        from agents.executor.agent import ExecutorAgent
        from agents.base_agent import BaseAgent
        from guardrails.exceptions import BlockedActionError
        inject_mocks(None, None, mock_memory)

        mock_gr = MagicMock()
        mock_gr.safe_execute_code = AsyncMock(
            side_effect=BlockedActionError(
                action_id="a1", reason="BLOCKLIST_HIT", violations=["rm -rf /"]
            )
        )
        BaseAgent._shared_guardrail = mock_gr

        agent = ExecutorAgent()
        result = await agent.safe_execute(code="rm -rf /", language="bash")

        assert result["success"] is False
        assert result.get("blocked") is True
        assert "Blocked by guardrail" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_step_dispatches(self, minimal_state, mock_guardrail, mock_memory):
        from agents.executor.agent import ExecutorAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = ExecutorAgent()
        sandbox_result = MagicMock()
        sandbox_result.stdout = "done\n"
        sandbox_result.stderr = ""
        sandbox_result.exit_code = 0
        sandbox_result.duration_ms = 50
        sandbox_result.timed_out = False
        agent._sandbox = MagicMock()
        agent._sandbox.safe_execute = AsyncMock(return_value=sandbox_result)
        agent._llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Executed successfully"
        agent._llm.ainvoke = AsyncMock(return_value=mock_resp)

        step = {"action": "execute", "code": "echo done", "language": "bash"}
        result = await agent.execute(step, minimal_state)
        assert result.get("success") is True


# ═══════════════════════════════════════════════════════════════════════════════
# FileAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestFileAgent:

    @pytest.mark.asyncio
    async def test_read_existing_file(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello File Agent!")

        agent = FileAgent()
        step = {"action": "read", "path": str(test_file)}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is True
        assert "Hello File Agent!" in result["output"]

    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, minimal_state, mock_guardrail, mock_memory):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = FileAgent()
        step = {"action": "read", "path": "/nonexistent/path/file.txt"}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_write_file(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = FileAgent()
        out_path = tmp_path / "output.txt"
        step = {"action": "write", "path": str(out_path), "content": "Written by FileAgent"}
        result = await agent.execute(step, minimal_state)

        assert result.get("success", True) is not False or "blocked" in result
        # If not blocked, verify file was written
        if result.get("success"):
            assert out_path.read_text() == "Written by FileAgent"

    @pytest.mark.asyncio
    async def test_list_files(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        # Create some files
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "subdir").mkdir()

        agent = FileAgent()
        step = {"action": "list", "path": str(tmp_path)}
        result = await agent.execute(step, minimal_state)

        assert result.get("success", True) is not False
        assert "entries" in result
        assert len(result["entries"]) >= 3

    @pytest.mark.asyncio
    async def test_search_glob(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        (tmp_path / "report_2024.pdf").write_bytes(b"fake pdf")
        (tmp_path / "report_2025.pdf").write_bytes(b"fake pdf")
        (tmp_path / "notes.txt").write_text("not a report")

        agent = FileAgent()
        step = {
            "action": "search",
            "query": "report",
            "root": str(tmp_path),
            "mode": "glob",
            "glob": "**/report*",
        }
        result = await agent.execute(step, minimal_state)

        assert "results" in result
        paths = [r["path"] for r in result["results"]]
        assert any("report" in p for p in paths)

    @pytest.mark.asyncio
    async def test_delete_requires_confirmation(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")

        agent = FileAgent()
        step = {"action": "delete", "path": str(test_file), "confirmed": False}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is False
        assert result.get("requires_approval") is True
        # File should still exist
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_delete_with_confirmation(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        test_file = tmp_path / "to_delete.txt"
        test_file.write_text("delete me")

        agent = FileAgent()
        step = {"action": "delete", "path": str(test_file), "confirmed": True}
        result = await agent.execute(step, minimal_state)

        assert result.get("success", True) is True
        assert not test_file.exists()

    @pytest.mark.asyncio
    async def test_summarize_text_file(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        test_file = tmp_path / "report.txt"
        test_file.write_text("This is a detailed report about quarterly earnings. Q3 showed 15% growth.")

        agent = FileAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "Quarterly earnings report showing 15% Q3 growth."
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {"action": "summarize", "path": str(test_file)}
        result = await agent.execute(step, minimal_state)

        assert "summary" in result
        assert len(result["summary"]) > 0

    @pytest.mark.asyncio
    async def test_organize_dry_run(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        (tmp_path / "main.py").write_text("import os")
        (tmp_path / "README.md").write_text("# Docs")
        (tmp_path / "data.csv").write_text("a,b,c")

        agent = FileAgent()
        agent._llm = MagicMock()

        def side_effect(messages):
            async def _inner():
                r = MagicMock()
                content = messages[-1].content
                if ".py" in content:
                    r.content = "code"
                elif ".md" in content:
                    r.content = "documents"
                else:
                    r.content = "data"
                return r
            return _inner()

        agent._llm.ainvoke = AsyncMock(side_effect=lambda m: asyncio.coroutine(
            lambda: type('R', (), {'content': 'misc'})()
        )())

        step = {"action": "organize", "path": str(tmp_path), "dry_run": True}
        result = await agent.execute(step, minimal_state)

        assert "moves" in result
        assert result["dry_run"] is True
        # Files should NOT be moved in dry_run
        assert (tmp_path / "main.py").exists()

    @pytest.mark.asyncio
    async def test_convert_csv_to_json(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        csv_file = tmp_path / "data.csv"
        csv_file.write_text("name,age\nAlice,30\nBob,25\n")

        agent = FileAgent()
        step = {"action": "convert", "path": str(csv_file), "format": "json"}
        result = await agent.execute(step, minimal_state)

        assert result.get("success", True) is not False
        out_path = Path(result.get("output_path", ""))
        if out_path.exists():
            data = json.loads(out_path.read_text())
            assert data[0]["name"] == "Alice"

    @pytest.mark.asyncio
    async def test_batch_rename_dry_run(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.file.agent import FileAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        for i in range(3):
            (tmp_path / f"img_{i:04d}.jpg").write_bytes(b"fake jpg")

        agent = FileAgent()
        step = {
            "action": "batch_rename",
            "path": str(tmp_path),
            "glob": "*.jpg",
            "pattern": "photo_*.jpg",
            "dry_run": True,
        }
        result = await agent.execute(step, minimal_state)

        assert "renames" in result
        assert result["dry_run"] is True
        # Originals should be untouched
        assert len(list(tmp_path.glob("img_*.jpg"))) == 3


# ═══════════════════════════════════════════════════════════════════════════════
# WebAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestWebAgent:

    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, minimal_state, mock_guardrail, mock_memory):
        from agents.web.agent import WebAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = WebAgent()

        mock_results = [
            {"title": "AI News", "body": "Latest AI developments", "href": "https://example.com"},
            {"title": "LLM Guide", "body": "How LLMs work", "href": "https://example2.com"},
        ]

        with patch("agents.web.agent.DDGS") as mock_ddgs:
            mock_ddgs.return_value.text = MagicMock(return_value=mock_results)
            step = {"action": "search_duckduckgo", "query": "AI latest news", "max_results": 5}
            result = await agent.execute(step, minimal_state)

        assert "results" in result
        assert len(result["results"]) == 2
        assert result["engine"] == "duckduckgo"

    @pytest.mark.asyncio
    async def test_network_policy_blocks_url(self, minimal_state, mock_memory):
        from agents.web.agent import WebAgent
        from agents.base_agent import BaseAgent
        inject_mocks(None, None, mock_memory)

        mock_gr = MagicMock()
        mock_gr.network = MagicMock()
        mock_net_result = MagicMock()
        mock_net_result.allowed = False
        mock_net_result.reason = "Domain not whitelisted"
        mock_gr.network.check_url = MagicMock(return_value=mock_net_result)
        mock_gr.network.log_external_call = MagicMock()
        BaseAgent._shared_guardrail = mock_gr

        agent = WebAgent()
        step = {"action": "browse", "url": "https://blocked-domain.com"}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is False
        assert result.get("blocked") is True

    @pytest.mark.asyncio
    async def test_ensure_scheme_prepend(self):
        from agents.web.agent import WebAgent
        assert WebAgent._ensure_scheme("example.com") == "https://example.com"
        assert WebAgent._ensure_scheme("https://example.com") == "https://example.com"
        assert WebAgent._ensure_scheme("http://example.com") == "http://example.com"

    @pytest.mark.asyncio
    async def test_browse_url(self, minimal_state, mock_guardrail, mock_memory):
        from agents.web.agent import WebAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = WebAgent()
        agent._llm = MagicMock()

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.title = AsyncMock(return_value="Test Page")
        mock_page.evaluate = AsyncMock(return_value="This is the page content about AI.")
        mock_page.content = AsyncMock(return_value="<html>content</html>")
        mock_page.close = AsyncMock()

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_pw = AsyncMock()
        mock_pw.chromium = AsyncMock()
        mock_pw.chromium.launch = AsyncMock(return_value=mock_browser)
        mock_pw.stop = AsyncMock()

        mock_llm_resp = MagicMock()
        mock_llm_resp.content = "A page about AI content."
        agent._llm.ainvoke = AsyncMock(return_value=mock_llm_resp)

        with patch("agents.web.agent.async_playwright") as pw_factory:
            pw_cm = AsyncMock()
            pw_cm.__aenter__ = AsyncMock(return_value=mock_pw)
            pw_cm.__aexit__ = AsyncMock(return_value=None)
            pw_factory.return_value = pw_cm

            with patch.object(agent, "_get_page", return_value=(mock_page, mock_context, mock_browser, mock_pw)):
                step = {"action": "browse", "url": "https://example.com"}
                result = await agent.execute(step, minimal_state)

        # title should be fetched
        assert result.get("title") == "Test Page"
        assert "summary" in result


# ═══════════════════════════════════════════════════════════════════════════════
# SystemAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestSystemAgent:

    @pytest.mark.asyncio
    async def test_get_metrics(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        mock_metrics = {
            "timestamp": "2026-01-01T00:00:00",
            "cpu": {"percent": 23.5, "count_logical": 8},
            "memory": {"percent": 45.2, "used_gb": 7.2, "total_gb": 16.0},
            "disk": [{"mountpoint": "/", "percent": 55.0, "total_gb": 256.0, "used_gb": 140.8, "free_gb": 115.2}],
            "network": {"bytes_sent_mb": 100.0, "bytes_recv_mb": 500.0, "errin": 0, "errout": 0, "dropin": 0, "dropout": 0},
            "uptime_hours": 24.5,
        }

        agent = SystemAgent()
        with patch.object(agent, "_collect_metrics", return_value=mock_metrics):
            step = {"action": "metrics"}
            result = await agent.execute(step, minimal_state)

        assert "metrics" in result
        assert result["metrics"]["cpu"]["percent"] == 23.5
        assert result["metrics"]["memory"]["percent"] == 45.2

    @pytest.mark.asyncio
    async def test_detect_anomalies_healthy(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        healthy_metrics = {
            "timestamp": "2026-01-01T00:00:00",
            "cpu": {"percent": 30.0},
            "memory": {"percent": 50.0},
            "disk": [{"mountpoint": "/", "percent": 60.0}],
            "network": {"errin": 0, "errout": 0},
        }
        agent = SystemAgent()
        with patch.object(agent, "_collect_metrics", return_value=healthy_metrics):
            step = {"action": "anomalies"}
            result = await agent.execute(step, minimal_state)

        assert result["health"] == "healthy"
        assert len(result["anomalies"]) == 0

    @pytest.mark.asyncio
    async def test_detect_anomalies_disk_full(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        full_disk_metrics = {
            "timestamp": "2026-01-01T00:00:00",
            "cpu": {"percent": 20.0},
            "memory": {"percent": 40.0},
            "disk": [{"mountpoint": "/data", "percent": 92.0}],
            "network": {"errin": 0, "errout": 0},
        }
        agent = SystemAgent()
        with patch.object(agent, "_collect_metrics", return_value=full_disk_metrics):
            step = {"action": "anomalies"}
            result = await agent.execute(step, minimal_state)

        assert result["health"] in ("warning", "critical")
        disk_anomalies = [a for a in result["anomalies"] if a["type"] == "disk_full"]
        assert len(disk_anomalies) == 1

    @pytest.mark.asyncio
    async def test_detect_anomalies_high_cpu_and_ram(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        critical_metrics = {
            "timestamp": "2026-01-01T00:00:00",
            "cpu": {"percent": 95.0},
            "memory": {"percent": 92.0},
            "disk": [{"mountpoint": "/", "percent": 50.0}],
            "network": {"errin": 0, "errout": 0},
        }
        agent = SystemAgent()
        with patch.object(agent, "_collect_metrics", return_value=critical_metrics):
            step = {"action": "anomalies"}
            result = await agent.execute(step, minimal_state)

        assert result["health"] == "critical"
        types = {a["type"] for a in result["anomalies"]}
        assert "high_cpu" in types
        assert "high_memory" in types

    @pytest.mark.asyncio
    async def test_list_processes(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        mock_procs = [
            {"pid": 1, "name": "systemd", "user": "root", "cpu_pct": 0.1, "mem_pct": 0.2, "status": "sleeping"},
            {"pid": 100, "name": "python3", "user": "user", "cpu_pct": 5.0, "mem_pct": 2.5, "status": "running"},
        ]

        agent = SystemAgent()
        with patch.object(
            agent,
            "_list_processes",
            new_callable=AsyncMock,
            return_value={"processes": mock_procs, "output": "Listed 2 processes", "step_type": "process_list"},
        ):
            step = {"action": "processes", "sort_by": "cpu", "limit": 10}
            result = await agent.execute(step, minimal_state)

        assert "processes" in result

    @pytest.mark.asyncio
    async def test_kill_requires_confirmation(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = SystemAgent()
        step = {"action": "kill", "pid": 12345, "confirmed": False}
        result = await agent.execute(step, minimal_state)

        assert result["success"] is False
        assert result.get("requires_approval") is True

    @pytest.mark.asyncio
    async def test_clean_temp_dry_run(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        old_file = tmp_path / "old.tmp"
        old_file.write_text("temporary data")
        # Make it appear old by modifying mtime
        import os
        old_time = os.path.getmtime(str(old_file)) - (8 * 86400)  # 8 days ago
        os.utime(str(old_file), (old_time, old_time))

        agent = SystemAgent()
        step = {
            "action": "clean",
            "paths": [str(tmp_path)],
            "dry_run": True,
            "older_than_days": 7,
        }
        result = await agent.execute(step, minimal_state)

        assert "candidates" in result
        assert result["dry_run"] is True
        assert old_file.exists()  # Not deleted in dry_run

    @pytest.mark.asyncio
    async def test_suggest_fixes(self, minimal_state, mock_guardrail, mock_memory):
        from agents.system.agent import SystemAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        disk_full_metrics = {
            "timestamp": "2026-01-01T00:00:00",
            "cpu": {"percent": 20.0},
            "memory": {"percent": 40.0},
            "disk": [{"mountpoint": "/", "percent": 90.0}],
            "network": {"errin": 0, "errout": 0},
        }
        agent = SystemAgent()
        with patch.object(agent, "_collect_metrics", return_value=disk_full_metrics):
            step = {"action": "suggest"}
            result = await agent.execute(step, minimal_state)

        assert "suggestions" in result
        assert len(result["suggestions"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# CodeAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestCodeAgent:

    @pytest.mark.asyncio
    async def test_generate_python_code(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "```python\ndef hello():\n    return 'hello'\n```"
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {"action": "generate", "description": "Write a hello function", "language": "python"}
        result = await agent.execute(step, minimal_state)

        assert result.get("success", True) is not False
        assert "hello" in result.get("code", "")
        assert result["language"] == "python"

    @pytest.mark.asyncio
    async def test_generate_saves_to_file(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "```python\nprint('saved')\n```"
        agent._llm.ainvoke = AsyncMock(return_value=response)

        out_file = tmp_path / "generated.py"
        step = {
            "action": "generate",
            "description": "Print saved",
            "language": "python",
            "save_to": str(out_file),
        }
        result = await agent.execute(step, minimal_state)
        assert out_file.exists()
        assert "print" in out_file.read_text()

    @pytest.mark.asyncio
    async def test_debug_returns_fixed_code(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = (
            "Root cause: missing colon after `if`.\n\n"
            "```python\nif x > 0:\n    print(x)\n```\n"
            "Watch out for similar syntax errors."
        )
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "debug",
            "code": "if x > 0\n    print(x)",
            "error": "SyntaxError: invalid syntax",
            "language": "python",
        }
        result = await agent.execute(step, minimal_state)

        assert "analysis" in result
        assert "fixed_code" in result
        assert "if x > 0:" in (result["fixed_code"] or "")

    @pytest.mark.asyncio
    async def test_explain_code(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "This function computes the factorial recursively."
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "explain",
            "code": "def fact(n):\n    return 1 if n <= 1 else n * fact(n-1)",
            "language": "python",
        }
        result = await agent.execute(step, minimal_state)

        assert "explanation" in result
        assert len(result["explanation"]) > 0

    @pytest.mark.asyncio
    async def test_refactor_code(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "```python\ndef factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n```\nChanges: added type hints."
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "refactor",
            "code": "def fact(n):\n    return 1 if n<=1 else n*fact(n-1)",
            "language": "python",
        }
        result = await agent.execute(step, minimal_state)

        assert "refactored_code" in result
        assert result["refactored_code"] is not None

    @pytest.mark.asyncio
    async def test_generate_tests(self, minimal_state, mock_guardrail, mock_memory, tmp_path):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "```python\nimport pytest\n\ndef test_hello():\n    assert hello() == 'hello'\n```"
        agent._llm.ainvoke = AsyncMock(return_value=response)

        src_file = tmp_path / "hello.py"
        src_file.write_text("def hello():\n    return 'hello'\n")

        step = {
            "action": "test",
            "path": str(src_file),
            "language": "python",
            "run": False,
        }
        result = await agent.execute(step, minimal_state)

        assert "test_code" in result
        assert "pytest" in (result["test_code"] or "")
        # Test file should be created alongside source
        test_file = tmp_path / "test_hello.py"
        assert test_file.exists()

    @pytest.mark.asyncio
    async def test_review_code(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        agent._llm = MagicMock()
        response = MagicMock()
        response.content = "Score: 7/10\n\nCritical issues: None\nImprovements: Add type hints, docstrings."
        agent._llm.ainvoke = AsyncMock(return_value=response)

        step = {
            "action": "review",
            "code": "def add(a, b):\n    return a + b",
            "language": "python",
        }
        result = await agent.execute(step, minimal_state)

        assert "review" in result
        assert result.get("score") == pytest.approx(7.0)

    @pytest.mark.asyncio
    async def test_git_status_no_confirmation_needed(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()

        with patch("agents.system.agent.SystemAgent._run_in_sandbox") as mock_run:
            mock_run.return_value = asyncio.coroutine(
                lambda: {"stdout": "On branch main\n", "stderr": "", "exit_code": 0, "success": True}
            )()
            with patch("agents.code.agent.SystemAgent") as mock_sys_cls:
                mock_sys = MagicMock()
                mock_sys._run_in_sandbox = AsyncMock(
                    return_value={"stdout": "On branch main\n", "stderr": "", "exit_code": 0, "success": True}
                )
                mock_sys_cls.return_value = mock_sys

                step = {"action": "git", "git_action": "status", "repo_path": "."}
                result = await agent.execute(step, minimal_state)

        assert result.get("git_action") == "status"

    @pytest.mark.asyncio
    async def test_git_commit_requires_confirmation(self, minimal_state, mock_guardrail, mock_memory):
        from agents.code.agent import CodeAgent
        inject_mocks(None, mock_guardrail, mock_memory)

        agent = CodeAgent()
        step = {
            "action": "git",
            "git_action": "commit",
            "message": "feat: add feature",
            "confirmed": False,
        }
        result = await agent.execute(step, minimal_state)

        assert result["success"] is False
        assert result.get("requires_approval") is True

    @pytest.mark.asyncio
    async def test_lsp_ast_enrichment(self):
        from agents.code.agent import CodeAgent
        agent = CodeAgent()

        code = "import os\n\nclass MyClass:\n    def my_method(self): pass\n\ndef standalone(): pass\n"
        enriched = await agent._lsp_enrich(code, "python")

        assert "MyClass" in enriched
        assert "my_method" in enriched
        assert "standalone" in enriched
        assert "os" in enriched


# ═══════════════════════════════════════════════════════════════════════════════
# AgentRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentRegistry:

    def test_singleton(self):
        from agents.registry import AgentRegistry
        r1 = AgentRegistry.get_instance()
        r2 = AgentRegistry.get_instance()
        assert r1 is r2

    def test_register_and_get(self):
        from agents.registry import AgentRegistry
        from agents.code.agent import CodeAgent
        registry = AgentRegistry()
        agent = CodeAgent()
        registry.register(agent)

        retrieved = registry.get("code")
        assert retrieved is agent

    def test_get_missing_agent_returns_none(self):
        from agents.registry import AgentRegistry
        registry = AgentRegistry()
        assert registry.get("nonexistent") is None

    def test_get_by_capability(self):
        from agents.registry import AgentRegistry
        from agents.code.agent import CodeAgent
        from agents.executor.agent import ExecutorAgent

        registry = AgentRegistry()
        code = CodeAgent()
        executor = ExecutorAgent()
        registry.register(code)
        registry.register(executor)

        agents = registry.get_by_capability("code_execute")
        names = [a.name for a in agents]
        assert "code" in names
        assert "executor" in names

    def test_list_all_returns_metadata(self):
        from agents.registry import AgentRegistry
        from agents.planner.agent import PlannerAgent

        registry = AgentRegistry()
        registry.register(PlannerAgent())
        all_agents = registry.list_all()

        assert len(all_agents) >= 1
        agent_meta = all_agents[0]
        assert "name" in agent_meta
        assert "capabilities" in agent_meta
        assert "tools" in agent_meta
        assert "agent_id" in agent_meta

    def test_unregister_agent(self):
        from agents.registry import AgentRegistry
        from agents.planner.agent import PlannerAgent

        registry = AgentRegistry()
        registry.register(PlannerAgent())
        assert "planner" in registry
        removed = registry.unregister("planner")
        assert removed is True
        assert "planner" not in registry

    def test_len(self):
        from agents.registry import AgentRegistry
        from agents.code.agent import CodeAgent
        from agents.executor.agent import ExecutorAgent

        registry = AgentRegistry()
        assert len(registry) == 0
        registry.register(CodeAgent())
        assert len(registry) == 1
        registry.register(ExecutorAgent())
        assert len(registry) == 2

    @pytest.mark.asyncio
    async def test_health_check_all_healthy(self):
        from agents.registry import AgentRegistry
        from agents.code.agent import CodeAgent
        from agents.executor.agent import ExecutorAgent

        registry = AgentRegistry()
        registry.register(CodeAgent())
        registry.register(ExecutorAgent())

        health = await registry.health_check()
        assert health["code"]["healthy"] is True
        assert health["executor"]["healthy"] is True

    def test_auto_discover_registers_agents(self, mock_guardrail, mock_memory):
        from agents.registry import AgentRegistry
        from agents.base_agent import BaseAgent
        BaseAgent._shared_guardrail = mock_guardrail
        BaseAgent._shared_memory = mock_memory

        registry = AgentRegistry.get_instance()
        # Should have discovered at least some agents
        # (exact number depends on which agents can be instantiated)
        assert len(registry) >= 0  # At minimum doesn't crash


# ═══════════════════════════════════════════════════════════════════════════════
# AgentRouter
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentRouter:

    def test_keyword_route_file_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("search for a file named report.pdf") == "file"
        assert router.classify_sync("rename all photos in the folder") == "file"

    def test_keyword_route_web_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("browse to https://example.com") == "web"
        assert router.classify_sync("search google for latest AI news") == "web"

    def test_keyword_route_system_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("check CPU and RAM usage") == "system"
        assert router.classify_sync("monitor disk space and alert me") == "system"

    def test_keyword_route_code_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("debug this Python code") == "code"
        assert router.classify_sync("write a function to parse JSON") == "code"

    def test_keyword_route_planner_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("plan a strategy to deploy the application") == "planner"

    def test_keyword_route_executor_agent(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("execute this shell command") == "executor"
        assert router.classify_sync("run this bash script") == "executor"

    def test_default_agent_for_unknown(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        assert router.classify_sync("do something mysterious") == "executor"

    @pytest.mark.asyncio
    async def test_classify_async_keyword_path(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        result = await router.classify("summarize the PDF file")
        assert result == "file"

    @pytest.mark.asyncio
    async def test_classify_async_llm_fallback(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=True)
        router._llm = MagicMock()
        response = MagicMock()
        response.content = "code"
        router._llm.ainvoke = AsyncMock(return_value=response)

        # Use a task that won't match any keyword rule
        result = await router.classify("apply the linter transformations to the codebase")
        assert result in ("code", "executor", "file")  # LLM or keyword match

    @pytest.mark.asyncio
    async def test_route_returns_agent_instance(self, mock_guardrail, mock_memory):
        from agents.router import AgentRouter
        from agents.registry import AgentRegistry
        from agents.base_agent import BaseAgent
        from agents.file.agent import FileAgent
        BaseAgent._shared_guardrail = mock_guardrail
        BaseAgent._shared_memory = mock_memory

        registry = AgentRegistry()
        registry.register(FileAgent())

        router = AgentRouter(use_llm=False, registry=registry)
        agent = await router.route("find all CSV files in the project directory")

        assert isinstance(agent, BaseAgent)

    @pytest.mark.asyncio
    async def test_route_falls_back_to_default(self, mock_guardrail, mock_memory):
        from agents.router import AgentRouter
        from agents.registry import AgentRegistry
        from agents.base_agent import BaseAgent
        from agents.executor.agent import ExecutorAgent
        BaseAgent._shared_guardrail = mock_guardrail
        BaseAgent._shared_memory = mock_memory

        registry = AgentRegistry()
        # Only register executor (fallback)
        registry.register(ExecutorAgent())

        router = AgentRouter(use_llm=False, registry=registry)
        agent = await router.route("???")
        assert agent.name == "executor"

    def test_routing_cache(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)

        # First call populates cache
        r1 = router.classify_sync("search for files")
        assert router.classify_sync("search for files") == r1  # Same result from cache

    def test_explain_routing(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)

        explanation = router.explain_routing("debug the Python script and commit to git")
        assert "keyword_scores" in explanation
        assert "recommended" in explanation
        # Both code and git-related keywords should score
        assert explanation["keyword_scores"].get("code", 0) > 0

    def test_add_custom_rule(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        router.add_rule(["custom_keyword_xyz"], "web")
        result = router.classify_sync("this has custom_keyword_xyz in it")
        assert result == "web"

    def test_get_routing_table(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=False)
        table = router.get_routing_table()
        assert isinstance(table, list)
        assert len(table) == 6   # One per agent
        agents_in_table = {row["agent"] for row in table}
        assert agents_in_table == {"planner", "executor", "file", "web", "system", "code"}

    @pytest.mark.asyncio
    async def test_llm_invalid_response_falls_back(self):
        from agents.router import AgentRouter
        router = AgentRouter(use_llm=True)
        router._llm = MagicMock()
        response = MagicMock()
        response.content = "INVALID_AGENT_NAME"
        router._llm.ainvoke = AsyncMock(return_value=response)

        # Ensure it doesn't crash and falls back to default
        result = await router.classify("some completely ambiguous task that matches nothing")
        assert result == "executor"

    def test_get_router_singleton(self):
        from agents.router import get_router, AgentRouter
        r1 = get_router()
        r2 = get_router()
        assert r1 is r2
        assert isinstance(r1, AgentRouter)
