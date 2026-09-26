"""User-visible chat fallback copy shared by the agent and frontend."""

from __future__ import annotations

STOCK_NO_ASSISTANT_HINT = (
    "The model returned no assistant text and no tool calls (Ollama may stream reasoning "
    "without a final answer, or the run was cut short). Try: (1) a shorter, one-step question, "
    "(2) another Ollama tag if this one misbehaves with tools, (3) Gemini in Spot-AI-fy, or "
    "(4) concrete examples in backend/AGENT_CONTEXT.md."
)

FRIENDLY_SPOTIFY_GUIDANCE = (
    "I can help with Spotify things like playing music, searching, playlists, and queue "
    "management — try something like “play my workout playlist”, “search for Taylor Swift”, "
    "or “what’s playing?”"
)


def is_unpersisted_assistant_fallback(text: str) -> bool:
    """True when assistant text should not be stored in chat history."""
    t = (text or "").strip()
    if not t:
        return True
    if t == STOCK_NO_ASSISTANT_HINT or t == FRIENDLY_SPOTIFY_GUIDANCE:
        return True
    if t.startswith("The model returned no assistant text"):
        return True
    if t.startswith("No response from model."):
        return True
    return False


def friendly_reply_for_empty_model_output(user_text: str) -> str:
    """Return a helpful reply when the model produced no usable assistant text."""
    _ = user_text
    return FRIENDLY_SPOTIFY_GUIDANCE
