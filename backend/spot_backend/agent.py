"""Spot-AI-fy: chat agent (Ollama with SSE-friendly streaming, Gemini, Spotify tools)."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from typing import Any

import httpx

from spot_backend.config import Settings, get_settings
from spot_backend.context_loader import load_optional_agent_context_markdown
from spot_backend.llm_prefs import read_effective_llm_provider, read_effective_ollama_model
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner

_SYSTEM = """You are a Spotify assistant with tools to read the user's library and control playback.
Rules:
- Use tools instead of guessing Spotify IDs. spotify_artist_albums accepts a catalog artist id or an artist name (resolved via search).
- Do not claim failures "usually mean permissions" or lack of Spotify access unless the tool JSON shows Spotify HTTP 403 (or 401) in the error field. If reconnect_spotify_unnecessary: true, wrong or empty playlist_id/tracks — fix arguments and retry; do NOT ask the user to reconnect.
- For spotify_add_tracks_to_playlist: if `reauth_may_resolve` is true (scope/token wording from Spotify in tool JSON), you may suggest Sign out → Connect and quote spotify_api_message. If `sign_out_not_recommended` is true (no reauth_may_resolve on 403, or validation/404 errors), do not suggest signing out — fix playlist_id and tracks. HTTP 401 always warrants re-auth. Do not claim the playlist id is "verified" or that you changed collaborative mode unless tool outputs prove it.
- Treat `granted_scopes` / `missing_scopes` in tool JSON as proof. If `stale_scopes_need_reauth` is true or `missing_scopes` is non-empty, state the concrete cause to the user ("your Spotify login is missing playlist-modify-public") and direct them to Sign out → Connect to re-consent — do NOT claim the playlist is wrong or not owned. If `scopes_appear_sufficient` is true on a 403, DO NOT open with Sign out — say scopes look fine and the failure is unusual; invite them to retry or share the exact playlist name, rather than pivoting or inventing permission stories. spotify_me now returns `granted_scopes`; use it to reason.
- After spotify_add_tracks_to_playlist fails: fix playlist_id (exact `id` from spotify_create_playlist in this chat) and track arguments, then retry. If spotify_create_playlist already returned an `id` in this conversation, that proves playlist-write scope — do not suggest sign-out for a later add failure unless `reauth_may_resolve` is true (Spotify message matched scope/token). If reconnect_spotify_unnecessary is true, never suggest reconnect. For HTTP 401 only, sign-in again is appropriate. Never open with "usually permissions" for validation errors or 404/400.
- If tool JSON includes assistant_guidance, follow it for user-facing wording. Never tell the user the playlist is inaccessible, locked, or that you cannot access it after a failed add — that misreads API/tool errors; retry add with correct arguments or quote the API message.
- If do_not_claim_ownership_issue is true on spotify_add_tracks_to_playlist, never tell the user they lack permission or that the playlist is not owned by them based on the title — call spotify_user_playlists to match name→id or reuse create response `id`, then retry add with tracks from search.
- After spotify_add_tracks_to_playlist fails, do not offer to create a new or replacement playlist unless the user explicitly asked for a new one; retry with paginated spotify_user_playlists (exact name→id), spotify_get_playlist + spotify_me if 403 persists, then add again.
- If spotify_playlist_tracks or spotify_get_playlist fails for one playlist id, do not claim you cannot read or play any of the user's playlists — pick another id from spotify_user_playlists (or paginate) and retry; one 403 is not proof all playlists are blocked.
- If tool JSON includes do_not_generalize_to_all_playlists or assistant_guidance_playlist_read, obey it: never tell the user you cannot determine which playlists have ten tracks, cannot play any playlist, or that there is a blanket 'recurring permission error' on their whole library. If user_playlists succeeded, you can list playlists; for play, use spotify_start_resume_playback with context_uri spotify:playlist:<id> on ids from that list when track listing fails.
- If read_403_ambiguous is true on spotify_playlist_tracks or spotify_get_playlist, do not open with Sign out or blame collaborative mode — follow the hint (owner vs me, then scopes). If playlist_not_owned_by_user on playlist_tracks, the list is followed-not-owned — use context_uri playback, not sign-out.
- spotify_add_tracks_to_playlist needs a writable playlist_id: use the exact `id` from spotify_create_playlist or an id from spotify_user_playlists (never a playlist id from someone else's search/catalog). A human-readable playlist name is not proof of the correct id. If tool JSON has playlist_not_owned_by_user, the list is followed or another user's — create a new playlist or pick an id where get_playlist.owner.id matches spotify_me.id; signing out will not fix that.
- For requests like "play John Mayer", search (artist/track), pick sensible results, then start playback with spotify:track: URIs (or context_uri for an album/playlist).
- When building tracks for spotify_add_tracks_to_playlist, ONLY use uri/id values from spotify_search results where type == "track" (tracks.items[i].uri or tracks.items[i].id). Do not use an artist id or album id as a track id — Spotify will accept the write as an empty ghost row. The add tool now verifies existence; if it returns `skipped_uris` or `rejected_uris`, drop those ids and do not claim they were added.
- After spotify_add_tracks_to_playlist returns, ALWAYS call spotify_playlist_tracks once for that playlist_id and report to the user only the tracks that are actually present (name + artist from the listing). Never claim specific song titles were added unless they appear in that follow-up listing. If the add response includes `added_count`, quote that number.
- If the user asks you to play a playlist after adding tracks, you must issue spotify_start_resume_playback with context_uri = `spotify:playlist:<id>` and the active device_id from the UI (or let the tool pick it — it auto-transfers when idle). If that call returns an error JSON, surface the error verbatim; do not say "starting playback" without actually calling the tool or after it failed.
- To start a playlist AT a specific track (e.g. "play my playlist starting with the one we just added"), call spotify_start_resume_playback with context_uri = `spotify:playlist:<id>` AND offset = `{"uri": "spotify:track:<id>"}` (or `{"position": N}` for 0-based index).
- When the user asks to repeat / loop / "cycle back"/"keep playing" a playlist or album, CALL spotify_set_repeat with state: "context" AFTER spotify_start_resume_playback has returned ok. Use state: "track" for looping one song, "off" to stop. For shuffle use spotify_set_shuffle with state: true/false. You DO have these tools — never tell the user you cannot set repeat or that they must do it in the Spotify app.
- PLAY NOW vs PLAY NEXT — distinguish strictly by the user's verb. This is the most common mis-routing bug:
  * DEFAULT = PLAY NOW (interrupts). Any of these phrases means the user wants the track/playlist playing RIGHT NOW:
      "play X", "start playing X", "play X now", "play [playlist]", "start playing [playlist]",
      "play [playlist] at [track]", "play [playlist] starting with [track]", "start playing [playlist] beginning with [track]",
      "begin [playlist] with [track]", "cycle back around", "play it".
    For these → USE spotify_play_playlist (for playlist context, the correct tool 95% of the time) or spotify_start_resume_playback. These tools VERIFY the device actually switched; if response has ok:false and playback_verified:false, playback did NOT start and the previous song is still on — do NOT tell the user it started. Surface the error and suggest tapping play on the intended device or spotify_transfer_playback.
  * PLAY NEXT (queue, does not interrupt) — ONLY when the user's phrase explicitly contains the word "next" or "queue" or "after":
      "play X next", "queue X", "add X to the queue", "after this song play X", "put X up next".
    For these → use spotify_play_next (alias of spotify_add_to_queue). Never use these tools for any of the PLAY NOW phrases above. In particular, "start playing the playlist at that track" is PLAY NOW, not PLAY NEXT — it belongs in spotify_play_playlist.
  * If you're ever uncertain, default to spotify_play_playlist / spotify_start_resume_playback. Queueing is only correct when the user literally says "next", "queue", or "after".
- Multi-verb requests (e.g. "add a track, verify it, then play starting at that track with repeat on") must actually run EVERY verb as a tool call. Prefer the COMPOSITE tools when they match: spotify_add_tracks_by_query (search + dedupe + add in one call, with optional min_year) and spotify_play_playlist (start context, optional start_at_uri, repeat, shuffle). Typical flow for "add a <artist> track (ideally from <year>) to <playlist>, verify, and play starting there with repeat on": spotify_add_tracks_by_query → spotify_play_playlist with start_at_uri = added_tracks[0].uri and repeat = "context". The composite tools are the source of truth for what was added; quote their added_tracks list in your summary. Only fall back to spotify_search + spotify_add_tracks_to_playlist + spotify_start_resume_playback + spotify_set_repeat if the user specifies tracks you cannot express as a single query.
- NEVER invent / memorize Spotify track ids. All track ids must come from a tool result in this session (spotify_search, spotify_add_tracks_by_query.added_tracks, spotify_playlist_tracks, spotify_get_track). If a previous spotify_add_tracks_to_playlist call returned rejected_uris or try_instead = "spotify_add_tracks_by_query", switch to spotify_add_tracks_by_query instead of guessing new ids.
- KNOWN SPOTIFY API LIMITATIONS — do not iterate tools looking for data the Web API does not expose. Tell the user plainly instead:
  * "Most listened / most played / favorite playlist" — Spotify's Web API does NOT expose per-playlist listen counts or a top-playlists endpoint. User-top is only exposed at track and artist granularity (/me/top/tracks, /me/top/artists, time_range short_term|medium_term|long_term). Say that Spotify does not publish top-playlist analytics and offer to fetch the user's top tracks or top artists instead. Do NOT call spotify_user_playlists + spotify_playlist_tracks in a loop trying to compute a "most listened" ranking.
  * Per-track / per-album play counts — not exposed on the Web API. Say so; offer top tracks/artists as a proxy.
  * Listening history beyond the most recently played items — the Web API caps recently-played at 50. State that clearly rather than iterating.
- The user selects an active device in the UI; omit device_id unless you must override it.
- After tools return, give a short natural language summary for the user.

High-level phrasing (you resolve intent → concrete tools; do not ask the user for Spotify ids first unless truly impossible):
- "Add [artist or songs] to my playlist [name]" / "put these on [name]" → spotify_user_playlists (match the name to an id), then USE spotify_add_tracks_by_query with {playlist_id, query, count, min_year?} as a single call. Only fall back to spotify_search + spotify_add_tracks_to_playlist when the user picked specific songs by title that need individual resolution. Never use a playlist id from someone else's search result.
- "Play [playlist] starting at [track] with repeat on" / "cycle back around" → spotify_play_playlist with {playlist_id, start_at_uri, repeat: "context"} as a single call. Use the uri from the preceding add response's added_tracks list or from spotify_search.
- "Create [name] and add …" / "new playlist with …" → spotify_create_playlist, then spotify_search, then spotify_add_tracks_to_playlist with playlist_id = `id` or `playlist_id_for_add_tracks` from the create response and tracks = search `tracks.items` (either a flat array of those objects or the whole `{items: [...]}` object under `tracks` — both work server-side). Then play with spotify_start_resume_playback and context_uri `spotify:playlist:` + that same id (do not invent permissions errors if add failed — read the tool JSON).
- "What's in / on [playlist]?" → spotify_user_playlists and/or spotify_get_playlist / spotify_playlist_tracks.
- "Play any of my playlists that has 10+ tracks" → spotify_user_playlists (paginate), try playlist_tracks or get_playlist per id to count, or skip counting and start playback with context_uri spotify:playlist:<id> for each candidate until one works; if one id returns 403 try the next id — do not give up on the whole library.
- "My liked / saved songs" → spotify_user_saved_tracks (paginate with offset if needed).
- "Most popular / best / compare" (albums or tracks) → combine spotify_search, spotify_get_album, spotify_artist_top_tracks, or popularity fields from catalog tools — infer reasonable metrics, say what you used.
- Ambiguous artist, album, or playlist names → disambiguate with spotify_search or spotify_user_playlists before playback or edits.
- If a tool fails, read error JSON (detail, hint), adjust the plan (e.g. different playlist id, smaller batch), and continue when possible instead of giving up after one call."""

_JSON_TOOL_MODE_SUFFIX = """

CRITICAL — Ollama JSON tool mode (this model has no native tool API):
- Until you see tool results in the chat, you MUST NOT reply with an empty message, code-only explanations, or chit-chat.
- The API may set JSON-only mode: then output ONLY a JSON array like [{"name":"spotify_user_playlists","arguments":{}}] — no markdown fences, no prose, no code comments.
- Otherwise your next assistant message must be EXACTLY one markdown JSON code block and nothing else — no preamble, no markdown outside the fence:
```json
[{"name": "spotify_search", "arguments": {"query": "John Mayer", "types": "track,artist", "limit": 5}}]
```
- Each array element: "name" (string) and "arguments" (object). Use valid JSON (double quotes).
- If the user asks anything about Spotify (library, search, play, albums, playlists), your FIRST step is almost always `spotify_search` or `spotify_me` / `spotify_user_playlists` — pick the one that best resolves a vague request (e.g. playlist name → user_playlists; artist → search).
- AFTER tool results are pasted into the conversation as user messages, answer in short plain text, or emit another ```json block if you need more tools.

Tool names: spotify_search, spotify_me, spotify_user_playlists, spotify_get_playlist, spotify_playlist_tracks, spotify_user_saved_tracks, spotify_get_album, spotify_get_track, spotify_get_artist, spotify_artist_albums, spotify_artist_top_tracks, spotify_create_playlist, spotify_update_playlist, spotify_add_tracks_to_playlist, spotify_add_tracks_by_query, spotify_remove_playlist_tracks, spotify_reorder_playlist_tracks, spotify_replace_playlist_tracks, spotify_unfollow_playlist, spotify_devices, spotify_playback_state, spotify_transfer_playback, spotify_start_resume_playback, spotify_play_playlist, spotify_pause, spotify_skip_next, spotify_skip_previous, spotify_add_to_queue, spotify_play_next, spotify_set_repeat, spotify_set_shuffle, spotify_seek, spotify_set_volume.
"""

_JSON_MODE_EMPTY_NUDGE = (
    "You returned nothing usable. This chat requires Spotify tools via JSON. "
    "Reply with ONLY a ```json code block containing a JSON array of tool calls, for example:\n"
    "```json\n"
    '[{"name":"spotify_search","arguments":{"query":"the user topic","types":"track,artist","limit":5}}]\n'
    "```\n"
    "No other text before or after the fence."
)

_JSON_PLAIN_ANSWER_FOLLOWUP = (
    "Spot-AI-fy: Spotify tool results are in the messages above. "
    "Answer the user's question in plain English (counts, names, one short paragraph). "
    "Do not start your reply with `{`, `[`, or a markdown code fence. "
    "Use a single ```json ... ``` tool block only if you still need another Spotify API call."
)


def _tool_calls_from_payload(payload: Any) -> list[dict[str, Any]] | None:
    if isinstance(payload, dict):
        if "tool_calls" in payload and isinstance(payload["tool_calls"], list):
            payload = payload["tool_calls"]
        elif ("name" in payload or "tool" in payload) and ("arguments" in payload or "args" in payload):
            payload = [payload]
        else:
            return None
    if not isinstance(payload, list):
        return None
    out: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("tool")
        args = item.get("arguments") or item.get("args") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and name:
            out.append({"function": {"name": name, "arguments": args if isinstance(args, dict) else {}}})
    return out or None


def _parse_json_tool_calls(text: str) -> list[dict[str, Any]] | None:
    for pattern in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        m = re.search(pattern, text, flags=re.IGNORECASE)
        if not m:
            continue
        inner = m.group(1).strip()
        if inner.lower().startswith("json"):
            inner = inner[4:].lstrip()
        if not inner.startswith("["):
            continue
        try:
            payload = json.loads(inner)
        except json.JSONDecodeError:
            continue
        out = _tool_calls_from_payload(payload)
        if out:
            return out
    return None


def _first_json_array_slice(s: str) -> str | None:
    """Return the first top-level JSON array substring, respecting string literals."""
    start = s.find("[")
    if start < 0:
        return None
    depth = 0
    in_string = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _first_json_object_slice(s: str) -> str | None:
    """Return the first top-level JSON object substring (for a single-tool JSON object)."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_string:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_loose_json_tool_calls(text: str) -> list[dict[str, Any]] | None:
    """Some models emit a raw JSON array of tools without a markdown fence (possibly after prose)."""
    t = text.strip()
    if not t:
        return None
    candidates = [t]
    inner = _first_json_array_slice(t)
    if inner and inner not in candidates:
        candidates.append(inner)
    obj = _first_json_object_slice(t)
    if obj and obj not in candidates:
        candidates.append(obj)
    for cand in candidates:
        if cand.startswith("["):
            try:
                payload = json.loads(cand)
            except json.JSONDecodeError:
                continue
        elif cand.startswith("{"):
            try:
                payload = json.loads(cand)
            except json.JSONDecodeError:
                continue
        else:
            continue
        out = _tool_calls_from_payload(payload)
        if out:
            return out
    return None


def _parse_any_json_tool_calls(text: str) -> list[dict[str, Any]] | None:
    return _parse_json_tool_calls(text) or _parse_loose_json_tool_calls(text)


def _message_content_str(msg: dict[str, Any]) -> str:
    """Ollama `message.content` is usually a string; some stacks use a list of parts."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts: list[str] = []
        for item in c:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "".join(parts)
    return ""


def _text_for_tool_fallback(msg: dict[str, Any]) -> str:
    """Use assistant `content` and, if empty, reasoning-style fields some models use."""
    base = _message_content_str(msg).strip()
    if base:
        return base
    for key in ("thinking", "thought", "reasoning"):
        v = msg.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _reasoning_text(msg: dict[str, Any]) -> str:
    """Concatenate reasoning fields for streaming progress (some models fill these before `content`)."""
    parts: list[str] = []
    for key in ("thinking", "thought", "reasoning"):
        v = msg.get(key)
        if isinstance(v, str) and v:
            parts.append(v)
    return "\n".join(parts)


_TOOL_RESULT_CHAT_MAX = 12_000
_REASONING_STREAM_UI_CAP = 10_000


def _cap_tool_result_for_chat(text: str, max_len: int = _TOOL_RESULT_CHAT_MAX) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 120] + "\n... (truncated for chat context; ask a follow-up for more detail.)"


def _assistant_message_for_history(msg: dict[str, Any]) -> dict[str, Any]:
    """Shape assistant turns for the next Ollama request (string content, drop thinking)."""
    role = msg.get("role") or "assistant"
    out: dict[str, Any] = {"role": role}
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    flat = _message_content_str(msg)
    out["content"] = flat
    return out


def _normalize_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]] | None:
    raw = message.get("tool_calls")
    if not raw:
        return None
    normalized: list[dict[str, Any]] = []
    for tc in raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except json.JSONDecodeError:
                args = {}
        if isinstance(name, str) and name:
            normalized.append({"function": {"name": name, "arguments": args if isinstance(args, dict) else {}}})
    return normalized or None


def _ollama_tools_unsupported_error(body_text: str) -> bool:
    try:
        err = str((json.loads(body_text) or {}).get("error", ""))
    except (json.JSONDecodeError, TypeError):
        err = body_text
    el = err.lower()
    return "does not support tools" in el or ("tool" in el and "not support" in el)


def _json_mode_expecting_first_tool_result(messages: list[dict[str, Any]]) -> bool:
    """True until JSON-mode tool results (``Tool ```` lines) appear in the thread."""
    for m in messages:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str) and c.startswith("Tool `"):
            return False
    return True


def _forced_json_tool_calls_for_question(user_text: str) -> list[dict[str, Any]] | None:
    """Obvious Spotify intents when the model returns nothing (JSON tool mode)."""
    t = user_text.lower()
    if "playlist" not in t:
        return None
    if any(w in t for w in ("track", "song", "album", "artist", "follow")):
        return None
    if any(w in t for w in ("how many", "how many playlists", "number of", "count")):
        return [{"function": {"name": "spotify_user_playlists", "arguments": {}}}]
    return None


def _synthetic_assistant_json_content(tool_calls: list[dict[str, Any]]) -> str:
    arr: list[dict[str, Any]] = []
    for tc in tool_calls:
        fn = tc.get("function") or {}
        name = fn.get("name")
        args = fn.get("arguments") if isinstance(fn.get("arguments"), dict) else {}
        if isinstance(name, str) and name:
            arr.append({"name": name, "arguments": args})
    return "```json\n" + json.dumps(arr, ensure_ascii=False) + "\n```"


def _coerce_chat_history(history: Any) -> list[dict[str, str]]:
    """Normalize optional client history to user/assistant turns with string content."""
    out: list[dict[str, str]] = []
    if not history:
        return out
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = (item.get("content") or item.get("text") or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": str(role), "content": content})
    return out


def iter_ollama_chat_events(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yields Spot-AI-fy progress events for the Ollama agent; ends with ``final`` or ``error``."""
    runner = SpotifyToolRunner(settings=settings)
    try:
        ollama_model = read_effective_ollama_model(settings.data_dir, settings.ollama_model)
        base_system = _SYSTEM + load_optional_agent_context_markdown(settings)
        messages: list[dict[str, Any]] = [{"role": "system", "content": base_system}]
        history_turns = _coerce_chat_history(history)
        hist_cap = int(getattr(settings, "ollama_history_messages", 0) or 0)
        if hist_cap > 0 and len(history_turns) > hist_cap:
            history_turns = history_turns[-hist_cap:]
        for turn in history_turns:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": user_text})
        url = f"{settings.ollama_host.rstrip('/')}/api/chat"
        native_tools = True
        json_mode_patched = False

        # Shared Ollama tuning — applied to every /api/chat body so prompts don't
        # silently truncate at n_ctx=4096 and the model stays resident between turns.
        ollama_options: dict[str, Any] = {}
        if settings.ollama_num_ctx and settings.ollama_num_ctx > 0:
            ollama_options["num_ctx"] = int(settings.ollama_num_ctx)
        ollama_keep_alive = (settings.ollama_keep_alive or "").strip()
        tool_result_cap = int(getattr(settings, "ollama_tool_result_max", 0) or 0)
        if tool_result_cap <= 0:
            tool_result_cap = _TOOL_RESULT_CHAT_MAX
        steps_override = int(getattr(settings, "ollama_max_steps", 0) or 0)
        max_steps = steps_override if steps_override > 0 else int(settings.agent_max_steps)

        def _apply_ollama_tuning(b: dict[str, Any]) -> dict[str, Any]:
            if ollama_options:
                existing = b.get("options")
                if isinstance(existing, dict):
                    merged = {**ollama_options, **existing}
                else:
                    merged = dict(ollama_options)
                b["options"] = merged
            if ollama_keep_alive:
                b["keep_alive"] = ollama_keep_alive
            return b

        connect_bits = [ollama_model]
        if ollama_options.get("num_ctx"):
            connect_bits.append(f"ctx={ollama_options['num_ctx']}")
        if ollama_keep_alive:
            connect_bits.append(f"keep={ollama_keep_alive}")
        if hist_cap > 0:
            connect_bits.append(f"hist={hist_cap}")
        if steps_override > 0:
            connect_bits.append(f"steps={steps_override}")
        yield {
            "type": "status",
            "message": f"Connecting to Ollama ({', '.join(connect_bits)})…",
        }

        with httpx.Client(timeout=httpx.Timeout(600.0, connect=30.0)) as client:
            for step_idx in range(max_steps):
                yield {"type": "round", "step": step_idx + 1, "max": max_steps}
                nudge_attempt = 0
                msg: dict[str, Any] = {}
                while nudge_attempt < 3:
                    inner_guard = 0
                    msg = {}
                    while True:
                        inner_guard += 1
                        if inner_guard > 8:
                            yield {
                                "type": "error",
                                "message": "The assistant hit an internal retry limit talking to Ollama. Restart the API.",
                            }
                            return

                        body: dict[str, Any] = {
                            "model": ollama_model,
                            "messages": messages,
                            "stream": True,
                        }
                        if json_mode_patched and _json_mode_expecting_first_tool_result(messages):
                            body["format"] = "json"
                        if native_tools:
                            body["tools"] = OLLAMA_TOOLS
                        _apply_ollama_tuning(body)

                        with client.stream("POST", url, json=body) as r:
                            if r.status_code == 400 and native_tools:
                                err_body = r.read().decode("utf-8", errors="replace")
                                if _ollama_tools_unsupported_error(err_body):
                                    native_tools = False
                                    if not json_mode_patched:
                                        sys0 = messages[0]
                                        if sys0.get("role") == "system" and isinstance(sys0.get("content"), str):
                                            messages[0] = {
                                                "role": "system",
                                                "content": sys0["content"] + _JSON_TOOL_MODE_SUFFIX,
                                            }
                                        json_mode_patched = True
                                        yield {
                                            "type": "status",
                                            "message": "Using JSON tool mode (this model does not support native Ollama tools).",
                                        }
                                    continue
                            try:
                                r.raise_for_status()
                            except httpx.HTTPStatusError as e:
                                yield {
                                    "type": "error",
                                    "message": f"Ollama HTTP {e.response.status_code}: {(e.response.text or '')[:600]}",
                                }
                                return

                            if step_idx >= 1:
                                yield {
                                    "type": "status",
                                    "message": "Summarizing tool results — local models may show a long ‘thinking’ phase before text appears.",
                                }

                            prev_flat_len = 0
                            prev_reason_len = 0
                            reasoning_streamed = 0
                            stream_msg: dict[str, Any] = {}
                            for line in r.iter_lines():
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                except json.JSONDecodeError:
                                    continue
                                m = chunk.get("message")
                                if isinstance(m, dict):
                                    if m.get("content") is not None:
                                        stream_msg["content"] = m["content"]
                                    if m.get("role"):
                                        stream_msg["role"] = m["role"]
                                    if m.get("tool_calls"):
                                        stream_msg["tool_calls"] = m["tool_calls"]
                                    for key in ("thinking", "thought", "reasoning"):
                                        if m.get(key) is not None:
                                            stream_msg[key] = m[key]
                                root_tc = chunk.get("tool_calls")
                                if isinstance(root_tc, list) and root_tc:
                                    stream_msg["tool_calls"] = root_tc
                                flat = _message_content_str(stream_msg)
                                if len(flat) > prev_flat_len:
                                    delta = flat[prev_flat_len:]
                                    prev_flat_len = len(flat)
                                    if delta:
                                        yield {"type": "llm_delta", "text": delta}
                                reason = _reasoning_text(stream_msg)
                                if len(reason) > prev_reason_len:
                                    rd = reason[prev_reason_len:]
                                    prev_reason_len = len(reason)
                                    if rd and reasoning_streamed < _REASONING_STREAM_UI_CAP:
                                        room = _REASONING_STREAM_UI_CAP - reasoning_streamed
                                        piece = rd[:room] + ("…" if len(rd) > room else "")
                                        if piece:
                                            yield {"type": "llm_delta", "text": piece}
                                        reasoning_streamed += min(len(rd), room)
                                if chunk.get("done"):
                                    break
                            msg = stream_msg

                        break

                    if (
                        json_mode_patched
                        and _json_mode_expecting_first_tool_result(messages)
                        and isinstance(msg, dict)
                        and not _message_content_str(msg).strip()
                        and not _normalize_tool_calls(msg)
                    ):
                        yield {
                            "type": "status",
                            "message": "Stream empty — trying one non-stream JSON completion…",
                        }
                        ns_body: dict[str, Any] = {
                            "model": ollama_model,
                            "messages": messages,
                            "stream": False,
                            "format": "json",
                        }
                        _apply_ollama_tuning(ns_body)
                        try:
                            nr = client.post(
                                url,
                                json=ns_body,
                                timeout=httpx.Timeout(180.0, connect=30.0),
                            )
                            nr.raise_for_status()
                            nd = nr.json()
                            nm = nd.get("message")
                            if isinstance(nm, dict) and nm:
                                msg = nm
                        except httpx.HTTPError:
                            pass

                    if not isinstance(msg, dict) or not msg:
                        yield {"type": "error", "message": "Model returned an unexpected response."}
                        return

                    tool_calls = _normalize_tool_calls(msg)
                    parse_src = _text_for_tool_fallback(msg)
                    if not tool_calls and parse_src:
                        tool_calls = _parse_any_json_tool_calls(parse_src)
                    if not tool_calls:
                        forced = _forced_json_tool_calls_for_question(user_text)
                        if forced and json_mode_patched and _json_mode_expecting_first_tool_result(messages):
                            yield {
                                "type": "status",
                                "message": "Spot-AI-fy: calling spotify_user_playlists (playlist count / list question).",
                            }
                            tool_calls = forced
                            msg = {
                                "role": "assistant",
                                "content": _synthetic_assistant_json_content(forced),
                            }

                    if tool_calls:
                        break

                    if json_mode_patched and nudge_attempt < 2 and not parse_src.strip():
                        messages.append({"role": "user", "content": _JSON_MODE_EMPTY_NUDGE})
                        yield {
                            "type": "status",
                            "message": "No JSON tool block yet — prompting the model again…",
                        }
                        nudge_attempt += 1
                        continue

                    hint = (
                        "The model returned no assistant text and no tool calls (Ollama may stream reasoning "
                        "without a final answer, or the run was cut short). Try: (1) a shorter, one-step question, "
                        "(2) another Ollama tag if this one misbehaves with tools, (3) Gemini in Spot-AI-fy, or "
                        "(4) concrete examples in backend/AGENT_CONTEXT.md."
                    )
                    if parse_src:
                        content_only = _message_content_str(msg).strip()
                        if not content_only and parse_src:
                            th = parse_src
                            final_text = th[:8000] + ("…" if len(th) > 8000 else "")
                        else:
                            final_text = parse_src
                    else:
                        final_text = hint
                    yield {"type": "final", "text": final_text}
                    return

                messages.append(_assistant_message_for_history(msg))
                for tc in tool_calls:
                    fn = tc.get("function") or {}
                    name = fn.get("name")
                    args = fn.get("arguments") or {}
                    if not isinstance(name, str):
                        continue
                    if not isinstance(args, dict):
                        args = {}
                    yield {"type": "tool_start", "name": name}
                    result = runner.run(name, args)
                    preview = result[:240] + ("…" if len(result) > 240 else "")
                    yield {"type": "tool_done", "name": name, "preview": preview}
                    result_chat = _cap_tool_result_for_chat(result, max_len=tool_result_cap)
                    if native_tools:
                        messages.append({"role": "tool", "name": name, "content": result_chat})
                    else:
                        messages.append(
                            {"role": "user", "content": f"Tool `{name}` result:\n{result_chat}"},
                        )

                if json_mode_patched:
                    messages.append({"role": "user", "content": _JSON_PLAIN_ANSWER_FOLLOWUP})

            yield {"type": "final", "text": "Stopped after maximum tool steps. Try a simpler request."}
    finally:
        runner.close()


def iter_chat_events(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yields progress for Spot-AI-fy chat (Ollama streaming or Gemini)."""
    provider = read_effective_llm_provider(settings.data_dir, settings.llm_provider)
    if provider == "gemini":
        yield {"type": "status", "message": "Calling Gemini…"}
        try:
            from spot_backend.gemini_llm import run_chat_turn_gemini

            text = run_chat_turn_gemini(user_text, settings, history=history)
            yield {"type": "final", "text": text}
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return

    yield from iter_ollama_chat_events(user_text, settings, history=history)


def run_chat_turn_ollama(
    user_text: str,
    settings: Settings | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    settings = settings or get_settings()
    for ev in iter_ollama_chat_events(user_text, settings, history=history):
        if ev.get("type") == "final":
            return str(ev.get("text") or "")
        if ev.get("type") == "error":
            return str(ev.get("message") or "Error")
    return "No response from model."


def run_chat_turn(
    user_text: str,
    settings: Settings | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    settings = settings or get_settings()
    provider = read_effective_llm_provider(settings.data_dir, settings.llm_provider)
    if provider == "gemini":
        from spot_backend.gemini_llm import run_chat_turn_gemini

        return run_chat_turn_gemini(user_text, settings, history=history)
    return run_chat_turn_ollama(user_text, settings, history=history)
