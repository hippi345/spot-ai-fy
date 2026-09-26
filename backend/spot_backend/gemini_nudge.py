"""Gemini 'no tools yet' nudge policy."""

from __future__ import annotations

from spot_backend.prompt_intent import prompt_is_pure_how_to


def should_send_gemini_tool_nudge(
    *,
    user_text: str,
    had_tool_results: bool,
    action_claim_reprompted: bool,
    tool_nudge_used: bool,
    wants_spotify_data: bool,
) -> bool:
    if had_tool_results or action_claim_reprompted or tool_nudge_used:
        return False
    if not wants_spotify_data:
        return False
    if prompt_is_pure_how_to(user_text):
        return False
    return True
