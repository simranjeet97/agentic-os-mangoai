"""
agents/web/agent.py — Web Agent.
Handles web browsing, scraping, and search using Playwright + DuckDuckGo.
"""

from __future__ import annotations

import asyncio
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import AsyncDDGS

from agents.base_agent import BaseAgent
from core.state import AgentState

ALLOWED_SCHEMES = {"http", "https"}
MAX_PAGE_CHARS = 20_000  # Trim large pages
REQUEST_TIMEOUT = 30.0


class WebAgent(BaseAgent):
    name = "web"
    description = "Web browsing, search, and content extraction"

    async def execute(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        async def _run():
            action = step.get("action", "search")
            query = step.get("query") or step.get("description", "")
            url = step.get("url", "")

            if action == "search" or not url:
                return await self._search(query)
            elif action == "fetch" and url:
                return await self._fetch(url)
            elif action == "browse" and url:
                return await self._browse_with_playwright(url)
            else:
                return await self._search(query)

        return await self._run_with_audit(step, state, _run)

    async def _search(self, query: str, max_results: int = 5) -> dict[str, Any]:
        """DuckDuckGo search — no API key needed."""
        self.logger.info("Web search", query=query[:80])
        try:
            async with AsyncDDGS() as ddgs:
                results = await ddgs.atext(query, max_results=max_results)
            return {
                "output": results,
                "query": query,
                "result_count": len(results),
                "step_type": "web_search",
            }
        except Exception as exc:
            self.logger.warning("DuckDuckGo search failed", error=str(exc))
            return {"success": False, "error": str(exc), "step_type": "web_search"}

    async def _fetch(self, url: str) -> dict[str, Any]:
        """Fetch and parse a URL using httpx."""
        parsed = urlparse(url)
        if parsed.scheme not in ALLOWED_SCHEMES:
            return {"success": False, "error": f"Scheme not allowed: {parsed.scheme}"}

        self.logger.info("Fetching URL", url=url)
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "AgenticOS/1.0 (research bot)"},
                )
                resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "lxml")
            # Remove scripts, styles
            for tag in soup(["script", "style", "nav", "footer", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            text = text[:MAX_PAGE_CHARS]

            return {
                "output": text,
                "url": url,
                "status_code": resp.status_code,
                "title": soup.title.string if soup.title else "",
                "step_type": "web_fetch",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc), "step_type": "web_fetch"}

    async def _browse_with_playwright(self, url: str) -> dict[str, Any]:
        """Full browser rendering using Playwright (JS-heavy sites)."""
        try:
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()
                page.set_default_timeout(REQUEST_TIMEOUT * 1000)
                await page.goto(url, wait_until="networkidle")
                content = await page.inner_text("body")
                title = await page.title()
                await browser.close()

            return {
                "output": content[:MAX_PAGE_CHARS],
                "url": url,
                "title": title,
                "step_type": "web_browse",
            }
        except Exception as exc:
            self.logger.warning("Playwright browse failed, falling back to fetch", error=str(exc))
            return await self._fetch(url)
