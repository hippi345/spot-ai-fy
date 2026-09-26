from __future__ import annotations

from spot_backend.tool_registry import (
    MCP_AGENT_TOOL_ALLOWLIST_DIFF,
    agent_tool_names,
    mcp_tool_names,
)


def test_agent_and_mcp_tool_sets_match() -> None:
    agent = agent_tool_names()
    mcp = mcp_tool_names()
    missing_from_mcp = agent - mcp - MCP_AGENT_TOOL_ALLOWLIST_DIFF
    missing_from_agent = mcp - agent - MCP_AGENT_TOOL_ALLOWLIST_DIFF
    assert not missing_from_mcp, f"agent tools missing from MCP: {sorted(missing_from_mcp)}"
    assert not missing_from_agent, f"MCP tools missing from agent: {sorted(missing_from_agent)}"
