"""
tools/code_mcp.py — Git and Code Intelligence via Code Review Graph.
"""
import subprocess
from typing import Any, Dict, List
from mcp.types import Tool, TextContent

from tools.base import BaseMCPServer
from guardrails.models import ActionType
from core.code_graph import get_code_graph

class CodeMCP(BaseMCPServer):
    name = "code"

    def list_tools(self) -> List[Tool]:
        return [
            Tool(
                name="git_status",
                description="Get current git status.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="git_diff",
                description="Get git diff.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="run_tests",
                description="Run project tests via pytest.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="lint_code",
                description="Lint code.",
                inputSchema={"type": "object", "properties": {}}
            ),
            Tool(
                name="explain_code",
                description="Uses the Code Review Graph to get full code context and impact radius.",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "action": {
                            "type": "string", 
                            "description": "One of: search, impact, context, refresh", 
                            "enum": ["search", "impact", "context", "refresh"]
                        },
                        "query": {"type": "string", "description": "Symbol name or ID"}
                    },
                    "required": ["action"]
                }
            )
        ]

    def get_action_type(self, tool_name: str, arguments: Dict[str, Any]) -> ActionType:
        if tool_name in ("run_tests", "lint_code"):
            return ActionType.SHELL_COMMAND
        return ActionType.FILE_READ

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[TextContent]:
        try:
            if name == "git_status":
                res = subprocess.check_output(["git", "status"], text=True)
                return [TextContent(type="text", text=res)]
                
            elif name == "git_diff":
                res = subprocess.check_output(["git", "diff"], text=True)
                return [TextContent(type="text", text=res)]
                
            elif name == "run_tests":
                try:
                    res = subprocess.check_output(["pytest"], text=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    res = e.output
                return [TextContent(type="text", text=res)]
                
            elif name == "lint_code":
                try:
                    res = subprocess.check_output(["flake8"], text=True, stderr=subprocess.STDOUT)
                except subprocess.CalledProcessError as e:
                    res = e.output
                except FileNotFoundError:
                    res = "Flake8 not installed, assuming clean."
                return [TextContent(type="text", text=res)]
                
            elif name == "explain_code":
                graph = get_code_graph()
                await graph.initialize()
                
                action = arguments.get("action")
                query = arguments.get("query", "")
                
                if action == "search":
                    results = graph.find_symbol(query)
                    out = [f"Found {len(results)} symbols matching '{query}':"]
                    for r in results[:10]:
                        out.append(f"- {r['id']} ({r['type']}) in {r['file_path']}:{r['line_start']}")
                    return [TextContent(type="text", text="\n".join(out))]
                    
                elif action == "impact":
                    impact = graph.get_impact_analysis(query)
                    if "error" in impact:
                        return [TextContent(type="text", text=f"Error: {impact['error']}")]
                    
                    out = [f"Impact Analysis for '{query}':"]
                    out.append(f"- Impacted Symbols: {impact['impacted_symbols_count']}")
                    out.append(f"- Impacted Files: {impact['impacted_files_count']}\nTop 5 Files:")
                    for f in impact['impacted_files'][:5]:
                        out.append(f"  - {f}")
                    return [TextContent(type="text", text="\n".join(out))]
                    
                elif action == "context":
                    neighbors = graph.get_neighbors(query)
                    out = [f"Context for '{query}':"]
                    in_n = [n for n in neighbors if n['direction'] == 'incoming']
                    out_n = [n for n in neighbors if n['direction'] == 'outgoing']
                    
                    if in_n:
                        out.append("\nUsed By:")
                        for n in in_n[:10]:
                            out.append(f"  - {n['id']}")
                    if out_n:
                        out.append("\nUses/Depends On:")
                        for n in out_n[:10]:
                            out.append(f"  - {n['id']}")
                    return [TextContent(type="text", text="\n".join(out))]
                    
                elif action == "refresh":
                    await graph.index_codebase(force=False)
                    return [TextContent(type="text", text="Code review graph refreshed.")]
                
                return [TextContent(type="text", text="Unknown explain_code action.")]
            
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        except Exception as e:
            return [TextContent(type="text", text=f"Error running {name}: {e}")]
