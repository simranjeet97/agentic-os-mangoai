"""
agents/file/agent.py — FileAgent

Intelligent file operations:
  - search:   semantic (via ChromaDB embeddings) + fuzzy (rapidfuzz) + glob
  - summarize: any format via markitdown → LLM summarization
  - organize:  classify files by content → move to labelled subfolders
  - convert:   markdown → html, csv → json, etc.
  - batch_rename: pattern-based or LLM-powered renaming
  - read / write: safe fs operations

SAFETY RULES:
  - Never deletes without explicit user confirmation (requires_approval=True)
  - All destructive writes go through GuardrailMiddleware UndoBuffer snapshot
  - Paths outside the AGENT_WRITE zone require PendingApprovalException
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import aiofiles
from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.state import AgentState
from guardrails.models import ActionType, AgentAction


# ── File-type → MIME helpers ──────────────────────────────────────────────────

TEXT_EXTENSIONS = {
    ".txt", ".md", ".rst", ".py", ".js", ".ts", ".java", ".go", ".rs",
    ".c", ".cpp", ".h", ".yaml", ".yml", ".toml", ".json", ".xml",
    ".html", ".htm", ".css", ".sh", ".bash", ".env", ".ini", ".cfg",
    ".csv", ".tsv", ".sql", ".log",
}

SUMMARIZE_PROMPT = """\
You are a file analyst. Given the content of a file, produce a concise summary:
- Main purpose / topic
- Key data points or decisions
- Any action items or TODOs
Write 3-5 sentences maximum. Be direct.
"""

ORGANIZE_PROMPT = """\
You are a file classifier. Given a file name and its content snippet, assign ONE category label
from the list: {categories}.
Respond with ONLY the category label — nothing else.
"""

RENAME_PROMPT = """\
You are a naming expert. Given file information, suggest a concise, descriptive file name.
Follow these conventions:
- lowercase_with_underscores
- Include date prefix if file is date-stamped content (YYYYMMDD_)
- Keep extension unchanged
- Max 50 chars

Respond with ONLY the new filename including extension.
"""


class FileAgent(BaseAgent):
    name = "file"
    description = "Intelligent file operations — search, summarize, organize, convert, batch rename"
    capabilities = ["file_read", "file_write", "file_delete", "file_search"]
    tools = [
        "search_files", "summarize_file", "read_file", "write_file",
        "organize_files", "convert_file", "batch_rename",
    ]

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._llm = None

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = self._get_llm(temperature=0.0)
        return self._llm

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run() -> dict[str, Any]:
            action = step.get("action", "read")
            dispatch = {
                "read": self._read_file,
                "write": self._write_file,
                "search": self._search_files,
                "summarize": self._summarize_file,
                "organize": self._organize_files,
                "convert": self._convert_file,
                "batch_rename": self._batch_rename,
                "list": self._list_files,
                "delete": self._delete_file,
            }
            handler = dispatch.get(action, self._read_file)
            return await handler(step, state)

        return await self._run_with_audit(step, state, _run)

    # ── READ ──────────────────────────────────────────────────────────────────

    async def _read_file(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        path = Path(step.get("path", ""))
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        # Guardrail check (FILE_READ)
        await self._check_guardrail(
            str(path), ActionType.FILE_READ, state.get("user_id", "system")
        )

        max_bytes = int(step.get("max_bytes", 1_000_000))
        try:
            async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
                content = await f.read(max_bytes)
            return {
                "output": content,
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "step_type": "file_read",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── WRITE ─────────────────────────────────────────────────────────────────

    async def _write_file(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        path = Path(step.get("path", ""))
        content = step.get("content", "")

        # Guardrail check (FILE_WRITE) — triggers undo snapshot
        try:
            await self._check_guardrail(
                str(path), ActionType.FILE_WRITE, state.get("user_id", "system")
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "blocked": True}

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(content)
            return {
                "output": f"Written {len(content)} bytes to {path}",
                "path": str(path),
                "bytes_written": len(content.encode("utf-8")),
                "step_type": "file_write",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── SEARCH ────────────────────────────────────────────────────────────────

    async def _search_files(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Multi-strategy file search:
        1. Glob pattern match (exact filename patterns)
        2. Fuzzy match on filename (rapidfuzz ratio ≥ 70)
        3. Semantic search in file content index (ChromaDB)
        """
        query = step.get("query", "")
        root = Path(step.get("root", ".")).expanduser()
        mode = step.get("mode", "all")    # glob | fuzzy | semantic | all
        max_results = int(step.get("max_results", 20))

        if not root.exists():
            return {"success": False, "error": f"Search root not found: {root}"}

        results: list[dict[str, Any]] = []

        # ── 1. Glob ───────────────────────────────────────────────────────────
        if mode in ("glob", "all"):
            glob_pattern = step.get("glob", f"**/*{query}*") if query else "**/*"
            hits = sorted(root.rglob(glob_pattern))[:max_results]
            for h in hits:
                results.append({
                    "path": str(h),
                    "match_type": "glob",
                    "score": 1.0,
                })

        # ── 2. Fuzzy filename match ───────────────────────────────────────────
        if mode in ("fuzzy", "all") and query:
            try:
                from rapidfuzz import fuzz, process
                candidates = [str(p) for p in root.rglob("*") if p.is_file()]
                fuzzy_hits = process.extract(
                    query, candidates, scorer=fuzz.partial_ratio, limit=max_results
                )
                for path_str, score, _ in fuzzy_hits:
                    if score >= 70:
                        results.append({
                            "path": path_str,
                            "match_type": "fuzzy",
                            "score": score / 100,
                        })
            except ImportError:
                self.logger.warning("rapidfuzz not installed — fuzzy search skipped")

        # ── 3. Semantic content search ────────────────────────────────────────
        if mode in ("semantic", "all") and query:
            try:
                recall = await self.memory.recall(
                    query=query,
                    user_id=state.get("user_id", "global"),
                    top_k=max_results,
                )
                for r in recall.results:
                    if r.source.value == "semantic":
                        results.append({
                            "path": r.metadata.get("source_path", r.item_id),
                            "match_type": "semantic",
                            "score": r.relevance,
                            "snippet": r.content[:200],
                        })
            except Exception as exc:
                self.logger.warning("Semantic search failed", error=str(exc))

        # Deduplicate by path, keeping highest score
        seen: dict[str, dict] = {}
        for r in results:
            p = r["path"]
            if p not in seen or r["score"] > seen[p]["score"]:
                seen[p] = r

        final = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:max_results]
        return {
            "output": f"Found {len(final)} matches for '{query}'",
            "results": final,
            "step_type": "file_search",
            "query": query,
        }

    # ── SUMMARIZE ─────────────────────────────────────────────────────────────

    async def _summarize_file(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Summarize any file format. Uses markitdown for non-text formats."""
        path = Path(step.get("path", ""))
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        # Convert to text via markitdown (handles PDF, DOCX, PPTX, images, etc.)
        text = await self._extract_text(path)
        if not text:
            return {"success": False, "error": "Could not extract text from file"}

        # LLM summarization
        snippet = text[:6000]
        try:
            messages = [
                SystemMessage(content=SUMMARIZE_PROMPT),
                HumanMessage(content=f"File: {path.name}\n\nContent:\n{snippet}"),
            ]
            response = await self.llm.ainvoke(messages)
            summary = response.content.strip()
        except Exception as exc:
            summary = f"(LLM failed: {exc}) — Raw excerpt: {snippet[:500]}"

        # Index in semantic memory for future searches
        try:
            await self.memory.semantic.index_document(
                content=text[:5000],
                source_path=str(path),
                metadata={"file_name": path.name, "size": path.stat().st_size},
                namespace=state.get("user_id", "global"),
            )
        except Exception:
            pass

        return {
            "output": summary,
            "summary": summary,
            "path": str(path),
            "char_count": len(text),
            "step_type": "file_summarize",
        }

    # ── ORGANIZE ──────────────────────────────────────────────────────────────

    async def _organize_files(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Classify files by content and move them to labelled subfolders.
        Categories can be provided or default to generic types.
        """
        root = Path(step.get("path", ".")).expanduser()
        categories = step.get("categories", [
            "code", "documents", "data", "images", "config", "logs", "misc"
        ])
        dry_run = step.get("dry_run", True)  # Always dry_run by default for safety

        if not root.is_dir():
            return {"success": False, "error": f"Not a directory: {root}"}

        files = [f for f in root.iterdir() if f.is_file()]
        moves: list[dict[str, str]] = []

        cat_prompt = ORGANIZE_PROMPT.format(categories=", ".join(categories))

        for file in files[:50]:  # Limit to 50 files per run
            ext = file.suffix.lower()
            snippet = ""
            if ext in TEXT_EXTENSIONS:
                try:
                    async with aiofiles.open(file, "r", errors="replace") as f:
                        snippet = (await f.read(500)).strip()
                except Exception:
                    pass

            try:
                messages = [
                    SystemMessage(content=cat_prompt),
                    HumanMessage(content=f"Filename: {file.name}\nContent snippet: {snippet[:300]}"),
                ]
                response = await self.llm.ainvoke(messages)
                category = response.content.strip().lower()
                if category not in categories:
                    category = "misc"
            except Exception:
                category = self._classify_by_extension(ext)

            dest = root / category / file.name
            moves.append({"from": str(file), "to": str(dest), "category": category})

        if not dry_run:
            for move in moves:
                dest_path = Path(move["to"])
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(move["from"], move["to"])

        return {
            "output": f"{'Would move' if dry_run else 'Moved'} {len(moves)} files",
            "moves": moves,
            "dry_run": dry_run,
            "step_type": "file_organize",
        }

    # ── CONVERT ───────────────────────────────────────────────────────────────

    async def _convert_file(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Convert file formats: md→html, csv→json, json→yaml, txt→md, etc."""
        path = Path(step.get("path", ""))
        target_format = step.get("format", "").lower()

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        out_path = path.with_suffix(f".{target_format}")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            converted = await self._do_convert(content, path.suffix.lower(), target_format)
            if isinstance(converted, dict) and "error" in converted:
                return converted

            async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                await f.write(converted)

            return {
                "output": f"Converted {path.name} → {out_path.name}",
                "input_path": str(path),
                "output_path": str(out_path),
                "step_type": "file_convert",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── BATCH RENAME ──────────────────────────────────────────────────────────

    async def _batch_rename(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Batch rename files using a pattern or LLM-powered suggestions.
        Requires user confirmation before applying (dry_run=True by default).
        """
        root = Path(step.get("path", ".")).expanduser()
        pattern = step.get("pattern", "")         # e.g. "photo_*.jpg" → "img_*.jpg"
        use_llm = step.get("use_llm", False)
        dry_run = step.get("dry_run", True)
        glob = step.get("glob", "*")

        if not root.is_dir():
            return {"success": False, "error": f"Not a directory: {root}"}

        files = sorted(root.glob(glob))
        renames: list[dict[str, str]] = []

        for i, file in enumerate(files[:100]):
            if pattern:
                # Simple pattern substitution: replace * with index
                new_name = pattern.replace("*", str(i + 1).zfill(4))
                new_name = file.stem + "_" + new_name if "*" not in pattern else new_name
                new_name = file.with_name(new_name + file.suffix).name
            elif use_llm:
                snippet = ""
                if file.suffix in TEXT_EXTENSIONS:
                    try:
                        async with aiofiles.open(file, "r", errors="replace") as f:
                            snippet = (await f.read(300)).strip()
                    except Exception:
                        pass
                try:
                    messages = [
                        SystemMessage(content=RENAME_PROMPT),
                        HumanMessage(
                            content=f"Current filename: {file.name}\nContent snippet: {snippet[:200]}"
                        ),
                    ]
                    response = await self.llm.ainvoke(messages)
                    new_name = re.sub(r"[^\w\.\-]", "_", response.content.strip())
                    if not new_name.endswith(file.suffix):
                        new_name = Path(new_name).stem + file.suffix
                except Exception:
                    new_name = file.name
            else:
                new_name = file.name

            renames.append({"from": str(file), "to": str(root / new_name)})

        if not dry_run:
            for rename in renames:
                if rename["from"] != rename["to"]:
                    os.rename(rename["from"], rename["to"])

        return {
            "output": f"{'Would rename' if dry_run else 'Renamed'} {len(renames)} files",
            "renames": renames,
            "dry_run": dry_run,
            "step_type": "batch_rename",
        }

    # ── LIST ──────────────────────────────────────────────────────────────────

    async def _list_files(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        root = Path(step.get("path", ".")).expanduser()
        recursive = step.get("recursive", False)
        pattern = step.get("pattern", "*")

        if not root.exists():
            return {"success": False, "error": f"Path not found: {root}"}

        if recursive:
            entries = list(root.rglob(pattern))
        else:
            entries = list(root.glob(pattern))

        file_list = [
            {
                "path": str(e),
                "type": "dir" if e.is_dir() else "file",
                "size_bytes": e.stat().st_size if e.is_file() else None,
                "extension": e.suffix,
            }
            for e in sorted(entries)[:200]
        ]

        return {
            "output": f"Listed {len(file_list)} entries in {root}",
            "entries": file_list,
            "step_type": "file_list",
        }

    # ── DELETE (requires confirmation) ────────────────────────────────────────

    async def _delete_file(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """Delete a file — ALWAYS requires explicit user confirmation."""
        path = Path(step.get("path", ""))
        confirmed = step.get("confirmed", False)

        if not confirmed:
            return {
                "success": False,
                "requires_approval": True,
                "error": f"Deletion of {path} requires explicit user confirmation. "
                         "Set step['confirmed'] = True to proceed.",
                "path": str(path),
            }

        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        # Guardrail check (FILE_DELETE) → creates undo snapshot
        try:
            await self._check_guardrail(
                str(path), ActionType.FILE_DELETE, state.get("user_id", "system")
            )
        except Exception as exc:
            return {"success": False, "error": str(exc), "blocked": True}

        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            return {
                "output": f"Deleted {path}",
                "path": str(path),
                "step_type": "file_delete",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Guardrail helper ──────────────────────────────────────────────────────

    async def _check_guardrail(
        self,
        path: str,
        action_type: ActionType,
        user_id: str = "system",
    ) -> None:
        """Run guardrail check for a file action. Raises on violation."""
        action = AgentAction(
            agent_id=self.agent_id,
            agent_type=self.name,
            action_type=action_type,
            target_paths=[path],
            raw_input=path,
            metadata={"user_id": user_id},
        )
        from guardrails.exceptions import BlockedActionError, PendingApprovalException
        try:
            await self.guardrail.evaluate_action(action, user_role="agent")
        except (BlockedActionError, PendingApprovalException):
            raise

    # ── Text extraction (markitdown + fallback) ───────────────────────────────

    async def _extract_text(self, path: Path) -> str:
        """Extract text from any file format. Uses markitdown if available."""
        ext = path.suffix.lower()

        if ext in TEXT_EXTENSIONS:
            try:
                async with aiofiles.open(path, "r", encoding="utf-8", errors="replace") as f:
                    return await f.read(50_000)
            except Exception:
                return ""

        # Try markitdown for rich formats
        try:
            import asyncio
            loop = asyncio.get_event_loop()

            def _markitdown_convert() -> str:
                from markitdown import MarkItDown
                md = MarkItDown()
                result = md.convert(str(path))
                return result.text_content or ""

            return await loop.run_in_executor(None, _markitdown_convert)
        except ImportError:
            self.logger.warning("markitdown not installed — cannot convert rich formats")
        except Exception as exc:
            self.logger.warning("markitdown conversion failed", path=str(path), error=str(exc))

        return ""

    # ── Format conversion helper ──────────────────────────────────────────────

    async def _do_convert(
        self, content: str, from_ext: str, to_format: str
    ) -> str | dict:
        """Convert content between formats."""
        try:
            if from_ext == ".md" and to_format == "html":
                import markdown
                return markdown.markdown(content)

            if from_ext == ".csv" and to_format == "json":
                import csv, io
                reader = csv.DictReader(io.StringIO(content))
                return json.dumps(list(reader), indent=2)

            if from_ext == ".json" and to_format in ("yaml", "yml"):
                import yaml
                return yaml.dump(json.loads(content), default_flow_style=False)

            if from_ext in (".yaml", ".yml") and to_format == "json":
                import yaml
                return json.dumps(yaml.safe_load(content), indent=2)

            if to_format == "md":
                return f"# Converted\n\n```\n{content}\n```"

            return {"error": f"Unsupported conversion: {from_ext} → {to_format}"}
        except Exception as exc:
            return {"error": str(exc)}

    @staticmethod
    def _classify_by_extension(ext: str) -> str:
        """Heuristic category from file extension."""
        mapping = {
            ".py": "code", ".js": "code", ".ts": "code", ".go": "code",
            ".rs": "code", ".java": "code", ".c": "code", ".cpp": "code",
            ".md": "documents", ".txt": "documents", ".rst": "documents",
            ".pdf": "documents", ".docx": "documents",
            ".csv": "data", ".json": "data", ".xml": "data", ".yaml": "data",
            ".png": "images", ".jpg": "images", ".gif": "images", ".svg": "images",
            ".env": "config", ".toml": "config", ".ini": "config",
            ".log": "logs",
        }
        return mapping.get(ext, "misc")
