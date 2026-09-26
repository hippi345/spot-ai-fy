"""Agent vs MCP Spotify tool name sets (for parity tests)."""

from __future__ import annotations

import ast
from pathlib import Path

from spot_backend.spotify_tools import OLLAMA_TOOLS

# Intentional differences between agent-exposed tools and MCP stdio tools.
MCP_AGENT_TOOL_ALLOWLIST_DIFF: frozenset[str] = frozenset()

_MCP_SERVER_PATH = Path(__file__).with_name("mcp_server.py")


def agent_tool_names() -> set[str]:
    names: set[str] = set()
    for entry in OLLAMA_TOOLS:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if isinstance(fn, dict) and isinstance(fn.get("name"), str):
            names.add(fn["name"])
    return names


def mcp_tool_names() -> set[str]:
    """Parse mcp_server.py so tests do not import the MCP runtime (optional dep version)."""
    tree = ast.parse(_MCP_SERVER_PATH.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name.startswith("spotify_"):
            names.add(node.name)
    return names
