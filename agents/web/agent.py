"""
agents/web/agent.py — WebAgent

Uses Playwright headless browser for:
  - Browse URLs  (navigate, extract text/links/tables)
  - Extract structured data  (CSS selectors, JSON-LD, tables)
  - Fill forms  (find inputs by label, type, submit)
  - Screenshot pages  (full-page or element)
  - Google / DuckDuckGo search  (returns organic results)
  - Click elements  (by text, CSS selector, aria-label)

ALL external requests pass through NetworkPolicy (GuardrailMiddleware step 4).
Browser sessions are created fresh per task and closed on completion.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base_agent import BaseAgent
from core.state import AgentState


# ── Prompts ───────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """\
You are a web data extractor. Given the HTML text of a webpage, extract the
requested information as structured JSON.
- Return ONLY valid JSON.
- If the data is tabular, use a list of objects with consistent keys.
- Omit navigation menus, footers, ads, and boilerplate.
"""

SUMMARIZE_PAGE_PROMPT = """\
You are a web researcher. Given the text content of a webpage, provide a concise summary:
- Main topic or purpose of the page
- Key facts or data points (bullet list)
- Any notable links or calls to action
Keep it under 200 words.
"""


class WebAgent(BaseAgent):
    name = "web"
    description = "Playwright-powered headless browser: browse, scrape, fill forms, screenshot, search"
    capabilities = ["web_browse", "web_extract", "web_screenshot", "web_search"]
    tools = [
        "browse_url", "extract_data", "fill_form", "take_screenshot",
        "search_google", "search_duckduckgo", "click_element", "get_links",
    ]

    SCREENSHOT_DIR = Path("artifacts/screenshots")
    USER_AGENT = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    def __init__(self, agent_id: Optional[str] = None) -> None:
        super().__init__(agent_id)
        self._llm = None
        self._playwright = None
        self._browser = None
        self.SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

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
            action = step.get("action", "browse")
            dispatch = {
                "browse": self._browse_url,
                "extract": self._extract_data,
                "fill_form": self._fill_form,
                "screenshot": self._screenshot,
                "search": self._search,
                "click": self._click_element,
                "links": self._get_links,
                "search_google": self._search_google,
                "search_duckduckgo": self._search_duckduckgo,
            }
            handler = dispatch.get(action, self._browse_url)
            return await handler(step, state)

        return await self._run_with_audit(step, state, _run)

    # ── BROWSE URL ────────────────────────────────────────────────────────────

    async def _browse_url(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        url = step.get("url", "")
        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)

        # NetworkPolicy check
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason", "Blocked by NetworkPolicy")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                title = await page.title()
                text_content = await page.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                )
                html = await page.content()

                # Summarize via LLM
                summary = await self._summarize_page(title, text_content[:4000])

                return {
                    "output": summary,
                    "url": url,
                    "title": title,
                    "text_length": len(text_content),
                    "summary": summary,
                    "raw_text": text_content[:3000],
                    "step_type": "browse",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc), "url": url}

    # ── EXTRACT STRUCTURED DATA ───────────────────────────────────────────────

    async def _extract_data(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        url = step.get("url", "")
        selector = step.get("selector", "")
        extract_type = step.get("extract_type", "text")  # text | table | json_ld | links
        what = step.get("what", "")  # natural language description for LLM extraction

        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

                if selector:
                    elements = await page.query_selector_all(selector)
                    data = [await el.inner_text() for el in elements]
                elif extract_type == "table":
                    data = await self._extract_tables(page)
                elif extract_type == "json_ld":
                    data = await self._extract_json_ld(page)
                elif what:
                    html = await page.content()
                    data = await self._llm_extract(html, what)
                else:
                    text = await page.evaluate(
                        "() => document.body ? document.body.innerText : ''"
                    )
                    data = text[:5000]

                return {
                    "output": data,
                    "url": url,
                    "extract_type": extract_type,
                    "selector": selector,
                    "step_type": "extract",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc), "url": url}

    # ── FILL FORM ─────────────────────────────────────────────────────────────

    async def _fill_form(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        """
        Fill a web form and optionally submit it.
        fields: {label_or_selector: value}
        """
        url = step.get("url", "")
        fields: dict[str, str] = step.get("fields", {})
        submit_selector = step.get("submit", "")
        capture_result = step.get("capture_result", True)

        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

                filled: list[str] = []
                for field, value in fields.items():
                    try:
                        # Try label first, then selector
                        locator = page.get_by_label(field)
                        count = await locator.count()
                        if count == 0:
                            locator = page.locator(field)
                        await locator.first.fill(str(value))
                        filled.append(field)
                    except Exception as fe:
                        self.logger.warning("Could not fill field", field=field, error=str(fe))

                result_text = ""
                if submit_selector:
                    await page.locator(submit_selector).first.click()
                    await page.wait_for_load_state("domcontentloaded", timeout=10_000)
                    if capture_result:
                        result_text = await page.evaluate(
                            "() => document.body ? document.body.innerText : ''"
                        )

                return {
                    "output": f"Filled {len(filled)} fields, submitted: {bool(submit_selector)}",
                    "url": url,
                    "filled_fields": filled,
                    "result_text": result_text[:2000],
                    "step_type": "fill_form",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── SCREENSHOT ────────────────────────────────────────────────────────────

    async def _screenshot(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        url = step.get("url", "")
        full_page = step.get("full_page", True)
        selector = step.get("selector", "")        # element screenshot
        return_base64 = step.get("base64", False)

        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_domain = re.sub(r"[^\w]", "_", url.split("//")[-1].split("/")[0])
        out_path = self.SCREENSHOT_DIR / f"{safe_domain}_{ts}.png"

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)

                if selector:
                    el = page.locator(selector).first
                    screenshot_bytes = await el.screenshot()
                else:
                    screenshot_bytes = await page.screenshot(full_page=full_page)

                out_path.write_bytes(screenshot_bytes)
                result = {
                    "output": f"Screenshot saved: {out_path}",
                    "path": str(out_path),
                    "url": url,
                    "step_type": "screenshot",
                }
                if return_base64:
                    result["base64"] = base64.b64encode(screenshot_bytes).decode()
                return result
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── SEARCH (dispatches to Google or DuckDuckGo) ───────────────────────────

    async def _search(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        engine = step.get("engine", "duckduckgo").lower()
        if engine == "google":
            return await self._search_google(step, state)
        return await self._search_duckduckgo(step, state)

    async def _search_duckduckgo(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        query = step.get("query", "")
        max_results = int(step.get("max_results", 10))

        if not query:
            return {"success": False, "error": "No search query provided"}

        try:
            from duckduckgo_search import DDGS
            
            results = []
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    loop = asyncio.get_event_loop()
                    def _do_search():
                        with DDGS(timeout=25) as ddgs:
                            return list(ddgs.text(query, max_results=max_results))
                    
                    results = await loop.run_in_executor(None, _do_search)
                    if results:
                        break
                except Exception as e:
                    err_str = str(e).lower()
                    if ("timeout" in err_str or "parsing" in err_str) and attempt < max_retries - 1:
                        self.logger.warning(f"DuckDuckGo search error (attempt {attempt+1}/{max_retries}), retrying...", error=str(e))
                        await asyncio.sleep(1.5)
                        continue
                    raise e

            return {
                "output": f"Found {len(results)} results for '{query}'",
                "results": results,
                "query": query,
                "engine": "duckduckgo",
                "step_type": "search",
            }
        except ImportError:
            self.logger.warning("duckduckgo_search not installed — falling back to browser search")
        except Exception as exc:
            self.logger.error("DuckDuckGo API search failed critically", error=str(exc))

        # Fallback: browser-based DDG search
        return await self._browser_search(
            f"https://duckduckgo.com/?q={query.replace(' ', '+')}", state
        )

    async def _search_google(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        query = step.get("query", "")
        max_results = int(step.get("max_results", 10))

        if not query:
            return {"success": False, "error": "No search query provided"}

        url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={max_results}"
        return await self._browser_search(url, state, query=query)

    async def _browser_search(
        self,
        url: str,
        state: AgentState,
        query: str = "",
    ) -> dict[str, Any]:
        """Perform a search via browser and extract result snippets."""
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                text = await page.evaluate(
                    "() => document.body ? document.body.innerText : ''"
                )
                links = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('a[href]')).slice(0,30).map(a=>({text:a.innerText.trim(),href:a.href}))"
                )
                return {
                    "output": text[:3000],
                    "links": links,
                    "url": url,
                    "query": query,
                    "step_type": "search",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── CLICK ELEMENT ─────────────────────────────────────────────────────────

    async def _click_element(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        url = step.get("url", "")
        selector = step.get("selector", "")
        text = step.get("text", "")    # click by visible text

        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
                if text:
                    await page.get_by_text(text).first.click()
                elif selector:
                    await page.locator(selector).first.click()
                else:
                    return {"success": False, "error": "Provide selector or text to click"}

                await page.wait_for_load_state("domcontentloaded", timeout=8_000)
                new_url = page.url
                new_title = await page.title()

                return {
                    "output": f"Clicked element, navigated to {new_url}",
                    "new_url": new_url,
                    "new_title": new_title,
                    "step_type": "click",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── GET LINKS ─────────────────────────────────────────────────────────────

    async def _get_links(
        self,
        step: dict[str, Any],
        state: AgentState,
    ) -> dict[str, Any]:
        url = step.get("url", "")
        if not url:
            return {"success": False, "error": "No URL provided"}

        url = self._ensure_scheme(url)
        net_result = self._check_network(url, state.get("user_id", "system"))
        if not net_result.get("allowed", True):
            return {"success": False, "blocked": True, "error": net_result.get("reason")}

        try:
            page, context, browser, playwright = await self._get_page()
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
                links = await page.evaluate(
                    "() => Array.from(document.querySelectorAll('a[href]')).map(a=>({"
                    "text: a.innerText.trim().slice(0,200), href: a.href})).filter(l=>l.href)"
                )
                return {
                    "output": f"Found {len(links)} links on {url}",
                    "links": links[:100],
                    "url": url,
                    "step_type": "links",
                }
            finally:
                await self._close_session(page, context, browser, playwright)
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ── Browser session management ────────────────────────────────────────────

    async def _get_page(self):
        """Create a fresh single-use browser session."""
        from playwright.async_api import async_playwright
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=self.USER_AGENT,
            viewport={"width": 1280, "height": 800},
        )
        page = await context.new_page()
        return page, context, browser, playwright

    @staticmethod
    async def _close_session(page, context, browser, playwright) -> None:
        """Clean up browser resources."""
        try:
            await page.close()
            await context.close()
            await browser.close()
            await playwright.stop()
        except Exception:
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_network(self, url: str, user_id: str) -> dict[str, Any]:
        """Run NetworkPolicy check. Returns {allowed, reason}."""
        try:
            result = self.guardrail.network.check_url(url)
            self.guardrail.network.log_external_call(
                url=url, agent_id=self.agent_id,
                outcome="allowed" if result.allowed else "blocked",
            )
            return {"allowed": result.allowed, "reason": result.reason}
        except Exception as exc:
            self.logger.warning("NetworkPolicy check failed", error=str(exc))
            return {"allowed": True, "reason": "policy_check_failed"}

    async def _summarize_page(self, title: str, text: str) -> str:
        """LLM summarization of page content."""
        try:
            messages = [
                SystemMessage(content=SUMMARIZE_PAGE_PROMPT),
                HumanMessage(content=f"Title: {title}\n\nContent:\n{text[:3000]}"),
            ]
            response = await self.llm.ainvoke(messages)
            return response.content.strip()
        except Exception as exc:
            return f"(Summary unavailable: {exc})"

    async def _llm_extract(self, html: str, what: str) -> Any:
        """Use LLM to extract structured data from HTML."""
        # Strip HTML tags
        clean = re.sub(r"<[^>]+>", " ", html)
        clean = re.sub(r"\s+", " ", clean)[:5000]

        try:
            messages = [
                SystemMessage(content=EXTRACT_PROMPT),
                HumanMessage(
                    content=f"Extract: {what}\n\nPage content:\n{clean}"
                ),
            ]
            response = await self.llm.ainvoke(messages)
            raw = response.content.strip()
            match = re.search(r"```(?:json)?(.*?)```", raw, re.DOTALL)
            return json.loads(match.group(1) if match else raw)
        except Exception:
            return clean

    @staticmethod
    async def _extract_tables(page) -> list[list[str]]:
        """Extract all tables from the current page as lists of rows."""
        return await page.evaluate("""
            () => Array.from(document.querySelectorAll('table')).map(t =>
              Array.from(t.querySelectorAll('tr')).map(tr =>
                Array.from(tr.querySelectorAll('td,th')).map(td => td.innerText.trim())
              )
            )
        """)

    @staticmethod
    async def _extract_json_ld(page) -> list[dict]:
        """Extract JSON-LD structured data from the page."""
        scripts = await page.evaluate("""
            () => Array.from(document.querySelectorAll('script[type="application/ld+json"]'))
              .map(s => s.textContent)
        """)
        results = []
        for s in scripts:
            try:
                results.append(json.loads(s))
            except Exception:
                pass
        return results

    @staticmethod
    def _ensure_scheme(url: str) -> str:
        if not url.startswith(("http://", "https://")):
            return f"https://{url}"
        return url
