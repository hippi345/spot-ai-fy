"""Shared types for the Spot-AI-fy agent eval suite."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallRecord(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""


class AgentTurnResult(BaseModel):
    """Structured output from one agent turn (mock replay or live LLM)."""

    reply: str = ""
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    duration_s: float = 0.0
    shortcut: bool = False
    llm_invoked: bool = False
    provider: str = ""
    acted_uris: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


class EvalCaseInput(BaseModel):
    """Case input: user prompt plus execution mode (passed to the dataset task)."""

    prompt: str
    mode: Literal["mock", "live"] = "mock"
    history: list[dict[str, str]] = Field(default_factory=list)
    conversation_id: str = "eval-session"
    mock_script: list[dict[str, Any]] = Field(default_factory=list)
    mock_spotify_profile: str = "default"
    expect_shortcut: bool = False
    forbid_llm: bool = False
    eval_fixture: dict[str, Any] | None = None


class EvalCaseMetadata(BaseModel):
    """Per-case expectations (stored in Dataset metadata)."""

    group: str = "unknown"
    ci: bool = False
    description: str = ""
    expected_tools: list[str] = Field(default_factory=list)
    expected_tool_prefix: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    expect_shortcut: bool = False
    forbid_llm: bool = False
    max_latency_s: float | None = None
    accepted_uris: list[str] = Field(default_factory=list)
    mock_script: list[dict[str, Any]] = Field(default_factory=list)
    mock_spotify_profile: str = "default"
    expect_grounded: bool = False
    expect_tool_error: bool = False
    mood_request: bool = False
    library_index_arm: str = ""  # "on" | "off" | "" (unset)
