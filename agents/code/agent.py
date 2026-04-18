"""
agents/code/agent.py — CodeAgent

Write, debug, refactor, and test code with full git and LSP integration.

Actions:
  generate    — LLM code generation (any language)
  debug       — diagnose errors, suggest fixes, apply if requested
  refactor    — improve code quality, readability, efficiency
  explain     — explain code in plain English
  test        — generate and run unit tests
  lint        — run black/pylint/mypy/bandit (Python) or eslint (JS)
  git         — log, diff, status, add, commit, push (guarded)
  execute     — run code via sandbox (all languages)
  review      — full code review with improvement suggestions

Language support: Python, JavaScript/TypeScript, Bash, Go, Rust (generation only)
LSP: Uses tree-sitter for syntax-aware tokenization when available.
Git: All write-operations (commit, push) require guardrail approval.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

import aiofiles
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.state import AgentState
from guardrails.models import ActionType, AgentAction


# ── Prompts ───────────────────────────────────────────────────────────────────

CODEGEN_PROMPT = """\
You are an expert polyglot software engineer. Generate clean, production-quality code.
Requirements:
- Add type hints and docstrings (Python/TypeScript)
- Handle errors gracefully with specific exception types
- Write efficient, readable code following language conventions
- Do NOT include prose explanations outside code blocks
- Return ONLY the code, wrapped in a single ```{language} ... ``` block
"""

DEBUG_PROMPT = """\
You are an expert debugger. Given code and an error message:
1. Identify the root cause
2. Explain WHAT went wrong and WHY
3. Provide the FIXED code in a ```{language} ... ``` block
4. List any related issues to watch for

Be precise and technically accurate.
"""

REFACTOR_PROMPT = """\
You are a code quality expert. Refactor the given code for:
- Improved readability (clear names, structure)
- Better performance (remove redundancy, use efficient patterns)
- Stronger error handling
- Cleaner API surface

Return the refactored code in a ```{language} ... ``` block, then bullet-point the changes.
"""

EXPLAIN_PROMPT = """\
You are a code educator. Explain this code clearly:
1. What does it do (high level)?
2. How does it work (step by step)?
3. What are the key design decisions?
4. Are there any potential issues?

Target audience: senior engineer unfamiliar with this codebase.
"""

TEST_GENERATION_PROMPT = """\
You are a test engineer. Write comprehensive unit tests for the given code.
Use pytest (Python) or jest (JavaScript).
Cover:
- Happy path (at least 2 cases)
- Edge cases
- Error/exception paths
- Boundary conditions

Return ONLY the test code in a ```{language} ... ``` block.
"""

REVIEW_PROMPT = """\
You are a senior code reviewer. Review this code and provide:
1. Overall quality score (1-10)
2. Critical issues (bugs, security, correctness)
3. Improvement suggestions (performance, readability, maintainability)
4. Positive aspects
5. Specific line-level comments where applicable

Be direct, actionable, and specific.
"""

SUPPORTED_LANGUAGES = {"python", "javascript", "typescript", "bash", "shell", "go", "rust", "java", "c", "cpp"}


class CodeAgent(BaseAgent):
    name = "code"
    description = "Code generation, debugging, refactoring, testing, linting, git integration"
    capabilities = ["code_execute", "file_read", "file_write"]
    tools = [
        "generate_code", "debug_code", "refactor_code", "explain_code",
        "run_tests", "lint_code", "git_operation", "execute_code", "review_code",
        "index_codebase", "query_graph", "analyze_impact",
    ]

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._llm = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = self._get_llm("OLLAMA_CODE_MODEL", temperature=0.1)
        return self._llm

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            action = step.get("action", "generate")
            dispatch = {
                "generate": self._generate,
                "debug": self._debug,
                "refactor": self._refactor,
                "explain": self._explain,
                "test": self._generate_tests,
                "lint": self._lint,
                "git": self._git_operation,
                "execute": self._execute,
                "run": self._execute,
                "review": self._review,
                "index_codebase": self._index_codebase,
                "query_graph": self._query_graph,
                "analyze_impact": self._analyze_impact,
            }
            handler = dispatch.get(action, self._generate)
            return await handler(step, state)

        return await self._run_with_audit(step, state, _run)

    # ── GENERATE CODE ─────────────────────────────────────────────────────────

    async def _generate(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        description = step.get("description", state.get("goal", ""))
        language = step.get("language", "python").lower()
        save_to = step.get("save_to", "")

        if language not in SUPPORTED_LANGUAGES:
            return {"success": False, "error": f"Unsupported language: {language}"}

        self.logger.info("Generating code", language=language, task=description[:80])

        code = await self._llm_codegen(description, language)
        if code.get("error"):
            return {"success": False, **code}

        result: dict[str, Any] = {
            "output": code["code"],
            "code": code["code"],
            "language": language,
            "step_type": "code_generate",
        }

        # Optionally save the code to a file
        if save_to:
            try:
                out_path = Path(save_to)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(out_path, "w") as f:
                    await f.write(code["code"])
                result["saved_to"] = str(out_path)
            except Exception as exc:
                result["save_error"] = str(exc)

        return result

    # ── DEBUG ─────────────────────────────────────────────────────────────────

    async def _debug(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        error = step.get("error", step.get("stderr", ""))
        language = step.get("language", "python").lower()
        apply_fix = step.get("apply_fix", False)
        path = step.get("path", "")

        if not code:
            return {"success": False, "error": "No code provided to debug"}

        self.logger.info("Debugging code", language=language, error_snippet=error[:120])

        try:
            messages = [
                SystemMessage(content=DEBUG_PROMPT.format(language=language)),
                HumanMessage(
                    content=f"Language: {language}\n\nCode:\n```{language}\n{code[:4000]}\n```\n\nError:\n{error[:1500]}"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            analysis = response.content.strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        fixed_code = self._extract_code_block(analysis, language)
        result: dict[str, Any] = {
            "output": analysis,
            "analysis": analysis,
            "original_code": code,
            "fixed_code": fixed_code,
            "language": language,
            "step_type": "debug",
        }

        if apply_fix and fixed_code and path:
            try:
                async with aiofiles.open(path, "w") as f:
                    await f.write(fixed_code)
                result["fix_applied"] = True
                result["patched_path"] = path
            except Exception as exc:
                result["fix_error"] = str(exc)

        return result

    # ── REFACTOR ──────────────────────────────────────────────────────────────

    async def _refactor(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        language = step.get("language", "python").lower()
        goals = step.get("goals", "readability, performance, error handling")
        use_graph = step.get("use_graph", True)
        save = step.get("save", False)
        path = step.get("path", "")

        if not code:
            return {"success": False, "error": "No code provided to refactor"}

        graph_ctx = ""
        if use_graph and path:
            graph_ctx = await self._get_graph_context_for_file(path)

        try:
            messages = [
                SystemMessage(content=REFACTOR_PROMPT),
                HumanMessage(
                    content=f"Language: {language}\nGoals: {goals}\n\nCode Graph Context:\n{graph_ctx}\n\nCode:\n```{language}\n{code[:4000]}\n```"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            analysis = response.content.strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        refactored = self._extract_code_block(analysis, language)
        result: dict[str, Any] = {
            "output": analysis,
            "refactored_code": refactored,
            "original_code": code,
            "language": language,
            "step_type": "refactor",
        }

        if save and refactored and path:
            try:
                async with aiofiles.open(path, "w") as f:
                    await f.write(refactored)
                result["saved_to"] = path
            except Exception as exc:
                result["save_error"] = str(exc)

        return result

    # ── EXPLAIN ───────────────────────────────────────────────────────────────

    async def _explain(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        language = step.get("language", "python").lower()

        if not code:
            return {"success": False, "error": "No code provided to explain"}

        # LSP-powered token enrichment if tree-sitter available
        enriched = await self._lsp_enrich(code, language)

        try:
            messages = [
                SystemMessage(content=EXPLAIN_PROMPT),
                HumanMessage(
                    content=f"Language: {language}\n\nCode:\n```{language}\n{code[:5000]}\n```{enriched}"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            explanation = response.content.strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        return {
            "output": explanation,
            "explanation": explanation,
            "language": language,
            "step_type": "explain",
        }

    # ── GENERATE TESTS ────────────────────────────────────────────────────────

    async def _generate_tests(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        language = step.get("language", "python").lower()
        run_tests = step.get("run", False)
        path = step.get("path", "")

        if not code:
            return {"success": False, "error": "No code provided for test generation"}

        try:
            messages = [
                SystemMessage(content=TEST_GENERATION_PROMPT),
                HumanMessage(
                    content=f"Language: {language}\n\nCode to test:\n```{language}\n{code[:4000]}\n```"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        test_code = self._extract_code_block(raw, language)
        result: dict[str, Any] = {
            "output": test_code or raw,
            "test_code": test_code,
            "language": language,
            "step_type": "test_generate",
        }

        # Save test file alongside source
        if test_code and path:
            src_path = Path(path)
            test_path = src_path.parent / f"test_{src_path.name}"
            try:
                async with aiofiles.open(test_path, "w") as f:
                    await f.write(test_code)
                result["test_file"] = str(test_path)
            except Exception as exc:
                result["test_save_error"] = str(exc)

        # Optionally run tests
        if run_tests and test_code:
            run_result = await self._run_test_code(test_code, language)
            result["run_result"] = run_result

        return result

    # ── LINT ──────────────────────────────────────────────────────────────────

    async def _lint(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Run linting tools on a file or code snippet.
        Python: black (format) + pylint (quality) + mypy (types) + bandit (security)
        JavaScript: eslint (if available)
        """
        path = step.get("path", "")
        code = step.get("code", "")
        language = step.get("language", "python").lower()
        tools = step.get("tools", ["all"])   # all | black | pylint | mypy | bandit

        if not path and not code:
            return {"success": False, "error": "Provide path or code to lint"}

        # Write code to temp file if provided inline
        tmp_path: Optional[str] = None
        if code and not path:
            ext = {"python": ".py", "javascript": ".js", "typescript": ".ts"}.get(language, ".txt")
            with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
                f.write(code)
                tmp_path = f.name
            path = tmp_path

        results: dict[str, Any] = {}

        try:
            if language == "python":
                if "all" in tools or "black" in tools:
                    results["black"] = await self._run_tool(f"python3 -m black --check {path}")
                if "all" in tools or "pylint" in tools:
                    results["pylint"] = await self._run_tool(f"python3 -m pylint {path} --output-format=text")
                if "all" in tools or "mypy" in tools:
                    results["mypy"] = await self._run_tool(f"python3 -m mypy {path} --ignore-missing-imports")
                if "all" in tools or "bandit" in tools:
                    results["bandit"] = await self._run_tool(f"python3 -m bandit -r {path} -f text")
                if "all" in tools or "isort" in tools:
                    results["isort"] = await self._run_tool(f"python3 -m isort --check {path}")

            elif language in ("javascript", "typescript"):
                if shutil.which("eslint"):
                    results["eslint"] = await self._run_tool(f"eslint {path}")
                else:
                    results["eslint"] = {"stdout": "", "stderr": "eslint not found", "exit_code": -1}

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

        overall_clean = all(
            r.get("exit_code", 1) == 0 for r in results.values() if isinstance(r, dict)
        )

        return {
            "output": "Clean" if overall_clean else "Lint issues found",
            "clean": overall_clean,
            "results": results,
            "language": language,
            "path": path,
            "step_type": "lint",
        }

    # ── GIT OPERATIONS ────────────────────────────────────────────────────────

    async def _git_operation(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Execute git commands. Write ops (commit, push, reset) require guardrail approval.
        """
        git_action = step.get("git_action", "status")
        repo_path = step.get("repo_path", ".")
        message = step.get("message", "")
        files = step.get("files", ["."])
        branch = step.get("branch", "")
        remote = step.get("remote", "origin")
        confirmed = step.get("confirmed", False)

        WRITE_ACTIONS = {"commit", "push", "reset", "checkout", "merge", "rebase", "tag"}
        READ_ACTIONS = {"status", "log", "diff", "show", "branch", "remote", "stash-list"}

        if git_action in WRITE_ACTIONS and not confirmed:
            return {
                "success": False,
                "requires_approval": True,
                "error": f"Git {git_action} is a write operation and requires confirmed=True.",
            }

        # Build the git command
        cmd = self._build_git_command(
            git_action, repo_path, message, files, branch, remote
        )
        if not cmd:
            return {"success": False, "error": f"Unknown git action: {git_action}"}

        # Guardrail check for write ops
        if git_action in WRITE_ACTIONS:
            try:
                action = AgentAction(
                    agent_id=self.agent_id,
                    agent_type=self.name,
                    action_type=ActionType.SHELL_COMMAND,
                    command=cmd,
                    raw_input=cmd,
                )
                await self.guardrail.evaluate_action(action, user_role="agent")
            except Exception as exc:
                return {"success": False, "error": str(exc), "blocked": True}

        from agents.system.agent import SystemAgent
        sys_agent = SystemAgent(agent_id=self.agent_id)
        result = await sys_agent._run_in_sandbox(f"cd {repo_path} && {cmd}")

        return {
            **result,
            "git_action": git_action,
            "repo_path": repo_path,
            "step_type": "git",
        }

    # ── EXECUTE CODE ──────────────────────────────────────────────────────────

    async def _execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Execute code through the sandbox."""
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        language = step.get("language", "python").lower()
        timeout = int(step.get("timeout", 30))

        if not code:
            return {"success": False, "error": "No code to execute"}

        from agents.executor.agent import ExecutorAgent
        executor = ExecutorAgent(agent_id=self.agent_id)
        result = await executor.safe_execute(
            code=code,
            language=language,
            timeout=timeout,
            user_id=state.get("user_id", "system"),
        )
        result["step_type"] = "code_execute"
        return result

    # ── CODE REVIEW ───────────────────────────────────────────────────────────

    async def _review(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        code = step.get("code") or await self._read_file_if_path(step.get("path", ""))
        language = step.get("language", "python").lower()
        use_graph = step.get("use_graph", True)
        path = step.get("path", "")

        if not code:
            return {"success": False, "error": "No code provided for review"}

        graph_ctx = ""
        review_code = code[:5000]

        if use_graph and path:
            graph_ctx = await self._get_graph_context_for_file(path)
            
            # Retrieve structural skeleton from Graph instead of passing full code
            from core.code_graph import get_code_graph
            graph = get_code_graph()
            await graph.initialize()
            
            rel_path = path
            if os.path.isabs(path):
                try:
                    rel_path = os.path.relpath(path, os.getcwd())
                except ValueError:
                    pass

            symbols = [d for n, d in graph.graph.nodes(data=True) if d.get("file_path") == rel_path]
            
            if symbols:
                # Use skeleton to save tokens
                lines = code.splitlines()
                skeleton_lines = []
                for sym_data in sorted(symbols, key=lambda x: x.get('line_start', 0)):
                    start = sym_data.get('line_start', 1) - 1
                    skeleton_lines.append(f"# Symbol: {sym_data.get('name')} ({sym_data.get('type')})")
                    if start < len(lines):
                        skeleton_lines.append(lines[start])
                    skeleton_lines.append("    # ... [Implementation abstracted by Graph to save tokens] ...\n")
                
                review_code = "\n".join(skeleton_lines)
                if len(review_code) > 4000:
                    review_code = review_code[:4000] + "\n...[Truncated]"
                
                graph_ctx += "\n\nNote: Full implementation details omitted to save tokens. Use Graph Context."

        try:
            messages = [
                SystemMessage(content=REVIEW_PROMPT),
                HumanMessage(
                    content=f"Language: {language}\n\nCode Graph Context:\n{graph_ctx}\n\nCode Skeleton:\n```{language}\n{review_code}\n```"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            review = response.content.strip()
        except Exception as exc:
            return {"success": False, "error": str(exc)}

        # Extract score if present
        score_match = re.search(r"(?:score|rating)[:\s]+(\d+(?:\.\d+)?)\s*/?\s*10", review, re.I)
        score = float(score_match.group(1)) if score_match else None

        return {
            "output": review,
            "review": review,
            "score": score,
            "language": language,
            "step_type": "review",
        }

    # ── GRAPH TOOLS ───────────────────────────────────────────────────────────

    async def _index_codebase(self, step: dict[str, Any], state: AgentState) -> dict[str, Any]:
        """Trigger a codebase re-indexing."""
        force = step.get("force", False)
        from core.code_graph import get_code_graph
        graph = get_code_graph()
        await graph.index_codebase(force=force)
        return {
            "success": True,
            "message": "Codebase indexed",
            "nodes": graph.graph.number_of_nodes(),
            "edges": graph.graph.number_of_edges()
        }

    async def _query_graph(self, step: dict[str, Any], state: AgentState) -> dict[str, Any]:
        """Query for a symbol's context/neighbors."""
        symbol = step.get("symbol_id") or step.get("query")
        if not symbol:
            return {"success": False, "error": "Provide symbol_id or query"}
        
        from core.code_graph import get_code_graph
        graph = get_code_graph()
        await graph.initialize()
        
        results = graph.find_symbol(symbol)
        neighbors = []
        if results:
            target_id = results[0]["id"]
            neighbors = graph.get_neighbors(target_id)
            
        return {
            "success": True,
            "query": symbol,
            "found_symbols": results[:5],
            "neighbors": neighbors[:10],
            "step_type": "graph_query"
        }

    async def _analyze_impact(self, step: dict[str, Any], state: AgentState) -> dict[str, Any]:
        """Perform impact analysis on a symbol."""
        symbol = step.get("symbol_id")
        if not symbol:
            return {"success": False, "error": "Provide symbol_id"}
            
        from core.code_graph import get_code_graph
        graph = get_code_graph()
        await graph.initialize()
        
        impact = graph.get_impact_analysis(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "impact": impact,
            "step_type": "graph_impact"
        }

    async def _get_graph_context_for_file(self, file_path: str) -> str:
        """Helper to get structural context for a file to enrich LLM prompts."""
        from core.code_graph import get_code_graph
        graph = get_code_graph()
        await graph.initialize()
        
        rel_path = file_path
        if os.path.isabs(file_path):
            try:
                rel_path = os.path.relpath(file_path, os.getcwd())
            except ValueError:
                pass
        
        # Find module node
        mod_id = rel_path.replace("/", ".").replace(".py", "")
        # Find all symbols in this module
        symbols = [n for n, d in graph.graph.nodes(data=True) if d.get("file_path") == rel_path]
        
        if not symbols:
            return "No graph context available for this file."
            
        ctx = [f"Symbols defined in {rel_path}:"]
        for s in symbols[:20]:
            neighbors = graph.get_neighbors(s, direction="out")
            calls = [n["id"] for n in neighbors if n["relation"] == "calls"]
            ctx.append(f"- {s}: calls {', '.join(calls[:5]) if calls else 'none'}")
            
        return "\n".join(ctx)

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _llm_codegen(self, description: str, language: str) -> dict[str, Any]:
        """Call LLM to generate code."""
        try:
            messages = [
                SystemMessage(content=CODEGEN_PROMPT.format(language=language)),
                HumanMessage(content=f"Language: {language}\nTask: {description}"),
            ]
            response = await self.llm.ainvoke(messages)
            content = response.content.strip()
            code = self._extract_code_block(content, language)
            return {"code": code or content, "language": language}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _extract_code_block(text: str, language: str) -> str:
        """Extract code from LLM markdown response."""
        # Try language-specific fence first
        for lang_tag in (language, language[:2], ""):
            pattern = rf"```{lang_tag}\n?(.*?)```"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return text.strip()

    @staticmethod
    async def _read_file_if_path(path: str) -> str:
        """Read file content if a path is provided."""
        if not path:
            return ""
        p = Path(path)
        if not p.exists():
            return ""
        try:
            async with aiofiles.open(p, "r", encoding="utf-8", errors="replace") as f:
                return await f.read(50_000)
        except Exception:
            return ""

    async def _run_tool(self, cmd: str) -> dict[str, Any]:
        """Run a linting tool via subprocess and return result."""
        from agents.system.agent import SystemAgent
        sys_agent = SystemAgent(agent_id=self.agent_id)
        return await sys_agent._run_in_sandbox(cmd)

    async def _run_test_code(self, test_code: str, language: str) -> dict[str, Any]:
        """Write test code to temp file and run it."""
        ext = {"python": ".py", "javascript": ".js"}.get(language, ".txt")
        runner = "python3 -m pytest" if language == "python" else "npx jest"

        with tempfile.NamedTemporaryFile(suffix=ext, mode="w", delete=False) as f:
            f.write(test_code)
            tmp_path = f.name

        try:
            return await self._run_tool(f"{runner} {tmp_path} -v")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    async def _lsp_enrich(self, code: str, language: str) -> str:
        """
        LSP/tree-sitter enrichment: extract symbols, function names, types.
        Adds structured context to LLM prompts for better analysis.
        """
        if language != "python":
            return ""
        try:
            import ast
            tree = ast.parse(code)
            classes = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            functions = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            imports = [
                n.names[0].name if isinstance(n, ast.Import) else n.module or ""
                for n in ast.walk(tree)
                if isinstance(n, (ast.Import, ast.ImportFrom))
            ]
            ctx = []
            if classes:
                ctx.append(f"Classes: {', '.join(classes[:10])}")
            if functions:
                ctx.append(f"Functions: {', '.join(functions[:10])}")
            if imports:
                ctx.append(f"Imports: {', '.join(set(imports[:10]))}")
            return "\n\nAST context:\n" + "\n".join(ctx) if ctx else ""
        except SyntaxError:
            return ""
        except ImportError:
            return ""

    @staticmethod
    def _build_git_command(
        action: str,
        repo_path: str,
        message: str,
        files: list[str],
        branch: str,
        remote: str,
    ) -> str:
        """Build the git CLI command string."""
        commands = {
            "status": "git status",
            "log": "git log --oneline -20",
            "diff": "git diff",
            "show": "git show --stat HEAD",
            "branch": "git branch -a",
            "remote": f"git remote -v",
            "add": f"git add {' '.join(files)}",
            "commit": f'git add {" ".join(files)} && git commit -m {repr(message or "auto-commit")}',
            "push": f"git push {remote} {branch or 'HEAD'}",
            "stash-list": "git stash list",
        }
        return commands.get(action, "")
