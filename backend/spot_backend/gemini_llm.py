"""Spot-AI-fy: Gemini HTTP backend for chat + Spotify tools."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from spot_backend.config import Settings
from spot_backend.context_loader import load_optional_agent_context_markdown
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner

logger = logging.getLogger(__name__)

# Google returns 503 UNAVAILABLE when a specific Gemini model is over-subscribed
# (the message literally says "high demand … usually temporary"). 429 is
# rate-limit / quota. In both cases a short exponential backoff is the
# documented remedy before giving up.
# Source: https://ai.google.dev/gemini-api/docs/troubleshooting
_GEMINI_RETRY_STATUSES: frozenset[int] = frozenset({429, 503})
_GEMINI_RETRY_ATTEMPTS: int = 4
_GEMINI_RETRY_BASE_SLEEP: float = 1.5


def _gemini_post_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict[str, Any],
    json_body: dict[str, Any],
) -> httpx.Response:
    """POST to Gemini with exponential backoff on 503 / 429.

    On the final attempt (or any non-retryable status) we still call
    raise_for_status so the outer except httpx.HTTPStatusError handler can
    surface a useful error message to the user.
    """
    last: httpx.Response | None = None
    for attempt in range(_GEMINI_RETRY_ATTEMPTS):
        resp = client.post(url, params=params, json=json_body)
        last = resp
        if resp.status_code not in _GEMINI_RETRY_STATUSES:
            resp.raise_for_status()
            return resp
        # Honor an explicit Retry-After header when present; otherwise
        # backoff 1.5s, 3s, 6s, 12s.
        retry_after = resp.headers.get("Retry-After")
        sleep_s: float
        try:
            sleep_s = float(retry_after) if retry_after else _GEMINI_RETRY_BASE_SLEEP * (2 ** attempt)
        except ValueError:
            sleep_s = _GEMINI_RETRY_BASE_SLEEP * (2 ** attempt)
        if attempt == _GEMINI_RETRY_ATTEMPTS - 1:
            break
        time.sleep(sleep_s)
    assert last is not None
    last.raise_for_status()
    return last

_GEMINI_REST = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Gemini 2.5-flash silently emits empty content (finishReason=STOP, parts=[])
# in `AUTO` function-calling mode when the systemInstruction is large. We
# measured 15/15 empty replies with our full ~12k-char prompt, and 0/15 with
# a short prompt. So we keep a compressed version here (Ollama keeps the long
# version in agent.py — it doesn't suffer from this bug). The compressed prompt
# preserves intent routing, known-API-limitation guardrails, and the few
# response-style rules. Detailed remediation guidance now lives in tool
# response JSON (hint/assistant_guidance/spotify_api_message) which the model
# reads after each call, so we don't need to pre-load all of it.
# See backend/scripts/diag_gemini_repeat.py for the matrix that proved this.
_SYSTEM_FULL = """You are a Spotify assistant with tools to read the user's library and control playback.
Rules:
- Prefer tools over guessing Spotify IDs. Do not refuse or say you "cannot access" Spotify — call the tools.
- For "how many albums does X have": call spotify_search (types including artist), then spotify_artist_albums with artists.items[0].id — or pass the artist's name as artist_id (the tool resolves names). Report totals from the API (paginate if next is set).
- For requests like "play John Mayer", search (artist/track), pick sensible results, then start playback with spotify:track: URIs or context_uri (spotify:album:…, spotify:playlist:…, spotify:artist:…).
- Playlist edits: spotify_get_playlist / spotify_playlist_tracks to read; spotify_create_playlist + spotify_add_tracks_to_playlist to build; spotify_update_playlist, spotify_remove_playlist_tracks, spotify_reorder_playlist_tracks, spotify_replace_playlist_tracks, spotify_unfollow_playlist to change or remove from library. spotify_add_tracks_to_playlist needs playlist_id from spotify_user_playlists or spotify_create_playlist (not search/catalog). On 403, read hint and spotify_api_message; if explain_playlist_id_before_reconnect: true, lead with wrong playlist_id — not reconnect or ownership stories unless spotify_api_message explicitly says so.
- Never tell the user to "reconnect Spotify" when tool JSON includes reconnect_spotify_unnecessary: true — wrong playlist_id or track payload, not OAuth. On spotify_add_tracks_to_playlist: if `reauth_may_resolve` is true, Spotify's message matched scope/token heuristics — sign-out/reconnect may help; quote spotify_api_message. If `sign_out_not_recommended` is true, do not suggest sign-out (fix id/tracks first). HTTP 401 always means re-authenticate. If spotify_create_playlist already returned `id` in this chat, add failures are usually not fixed by signing out unless reauth_may_resolve. Do not claim collaborative mode or verified ownership without tool proof.
- Prefer scope evidence over guesses. If tool JSON has `stale_scopes_need_reauth` true or `missing_scopes` is non-empty, tell the user their Spotify consent is missing those scopes (name them) and must re-consent via Sign out → Connect — don't blame playlist_id or ownership. If `scopes_appear_sufficient` true on a 403, DO NOT lead with Sign out; say scopes look fine and the failure is unusual, and avoid pivoting to another action. spotify_me returns `granted_scopes`; use it to reason about what consent the user has.
- Follow assistant_guidance in tool JSON when present. If do_not_claim_ownership_issue is true, do not claim lack of permission or wrong ownership from the playlist name — fix id via user_playlists or create response, retry add. Do not say the playlist is inaccessible or pivot to another artist after add-tracks fails unless the user asked.
- After add-tracks fails, do not offer to create a substitute playlist by default; paginate spotify_user_playlists, verify with spotify_get_playlist + spotify_me if needed, retry add — only create a new playlist if the user requested one.
- One failed spotify_playlist_tracks or get_playlist does not mean all playlists are inaccessible — retry with ids from spotify_user_playlists; read reauth_heuristic_ambiguous_403 in tool JSON for generic Forbidden.
- If do_not_generalize_to_all_playlists or assistant_guidance_playlist_read appears in tool JSON, never claim you cannot count tracks in playlists or play any playlist for the user — follow that guidance; use spotify_start_resume_playback + context_uri when listing tracks fails.
- If playlist_not_owned_by_user is true on add-tracks, the playlist is followed or another user's — create a copy with spotify_create_playlist or choose an owned id; do not blame OAuth alone.
- If read_403_ambiguous is true on playlist read tools, do not default to Sign out/collaborative excuses. If playlist_not_owned_by_user on spotify_playlist_tracks, use start_resume_playback with context_uri instead of listing tracks.
- Track ids for adds must come from spotify_search results where type == "track" — tracks.items[i].uri or tracks.items[i].id. Do NOT substitute an artist id or album id as a track id. The add tool now verifies tracks exist; if it returns `skipped_uris` or `rejected_uris`, drop those ids and do not claim they were added.
- After every successful spotify_add_tracks_to_playlist, you MUST immediately call spotify_playlist_tracks for that playlist_id and only list song names that actually appear in that response. If the add response has `added_count`, quote it. Never claim a specific song was added unless it appears in the follow-up listing.
- To play a playlist after adding tracks, issue spotify_start_resume_playback with context_uri `spotify:playlist:<id>` and let the tool pick the active device (device_id is optional — it auto-transfers when idle). If that tool returns an error field, surface it verbatim; do not say "starting playback" without calling the tool or after it failed.
- To play a playlist starting AT a specific track, pass context_uri plus offset `{uri: 'spotify:track:<id>'}` (or `{position: N}`).
- When the user asks for repeat / loop / "cycle back"/"keep playing", CALL spotify_set_repeat with state "context" (whole playlist/album) or "track" (one song) or "off" AFTER spotify_start_resume_playback returns ok. For shuffle call spotify_set_shuffle with state true/false. You have these tools — do not tell the user you cannot set repeat or to adjust it in the Spotify app.
- PLAY NOW vs PLAY NEXT — strict verb-based routing:
  * DEFAULT = PLAY NOW (interrupts current playback). ALL of these phrases map to immediate playback: "play X", "start playing X", "play X now", "play [playlist]", "play [playlist] at [track]", "start playing [playlist] beginning with [track]", "begin [playlist] with [track]", "cycle back around". USE spotify_play_playlist (for playlists) or spotify_start_resume_playback. These tools verify that the device actually switched; if ok:false and playback_verified:false, playback did NOT switch — do not claim it started. Surface the error and suggest the user tap play on the intended device or call spotify_transfer_playback.
  * PLAY NEXT (queue only, does NOT interrupt) — ONLY when the user literally says "next", "queue", or "after": "play X next", "queue X", "add X to the queue", "after this song play X". Use spotify_play_next (alias of spotify_add_to_queue). NEVER use these for "start playing X" / "play [playlist] at [track]" — those are PLAY NOW and belong in spotify_play_playlist / spotify_start_resume_playback. If uncertain, default to PLAY NOW.
- Multi-verb requests (e.g. "add, verify, then play starting at X with repeat on") must execute EVERY verb as a tool call. PREFER the composite tools: spotify_add_tracks_by_query (search + dedupe + add with optional min_year) and spotify_play_playlist (start context + start_at_uri + repeat + shuffle in one call). For "add a <artist> song (ideally from <year>) to <playlist>, verify, then play starting there with repeat on" the minimal chain is: spotify_add_tracks_by_query → spotify_play_playlist with start_at_uri = added_tracks[0].uri and repeat = "context". Treat the added_tracks list in the response as authoritative — quote those names/artists. Only chain spotify_search + spotify_add_tracks_to_playlist + spotify_start_resume_playback + spotify_set_repeat when the user picked specific titles that don't fit a single query.
- NEVER invent or memorize Spotify track ids. Every track id you pass to a tool must come from a tool result in this session (spotify_search tracks.items, spotify_add_tracks_by_query.added_tracks, spotify_playlist_tracks, spotify_get_track). If spotify_add_tracks_to_playlist returned rejected_uris or try_instead = "spotify_add_tracks_by_query", switch tools instead of guessing more ids.
- KNOWN SPOTIFY API LIMITATIONS — do not iterate tools trying to fetch data that does not exist. When the user asks for something not supported, you MUST: (1) state in one sentence what is not exposed and why, and (2) immediately offer 2-3 specific tools the user CAN use that are closest to what they wanted. Never just say "I cannot do that" — always pair it with what you can do.
  * "Most listened / most played / favorite playlist" — Spotify's Web API does NOT expose per-playlist listen counts or top-playlists. It only exposes user-top at track and artist granularity. For those, CALL the dedicated tools spotify_top_tracks and spotify_top_artists (time_range=short_term|medium_term|long_term). Reply that Spotify does not publish top-playlist analytics, and offer to fetch the user's top tracks or top artists instead — do NOT call spotify_user_playlists + spotify_playlist_tracks in a loop looking for a count.
  * "My top / favorite / most played artists or tracks" — CALL spotify_top_artists or spotify_top_tracks (do NOT say you lack a tool, do NOT iterate spotify_search). Default time_range = medium_term; "this month / lately" → short_term; "all time / over the years" → long_term. The API caps results at 50; rank is the only ordering signal (no per-item play counts).
  * Per-track / per-album play counts — not exposed on the Web API. Say so, and offer top tracks/artists as a proxy.
  * Listening history beyond the most recently played 50 items — not exposed. spotify_playback_state + /me/player/recently-played (up to 50, not currently wrapped) is the ceiling. Say so rather than iterating.
  * "Who follows me" / "my followers" — Spotify's Web API does NOT expose your follower list, only a count via spotify_me.followers.total. Say so plainly and offer (a) the count, (b) spotify_followed_artists (artists you follow), (c) spotify_top_artists.
  * "People I follow" / "users I follow" — Spotify's Web API does NOT expose users you follow, only ARTISTS (via spotify_followed_artists). Say so and call spotify_followed_artists. Do NOT pretend you fetched users.
  * "What playlists does <user> have" — only PUBLIC playlists are visible. CALL spotify_user_public_playlists with their user_id (the part after spotify:user: or open.spotify.com/user/<id>). The Web API has NO endpoint to look up a user by display name; if only a name is given, ask for the user_id or profile URL. Private/unlisted playlists are never visible to anyone but the owner.
  * Editing someone else's playlist — NOT supported by the Web API. Offer spotify_duplicate_playlist to copy it into a new playlist the user owns; the new playlist is fully writable for spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks / etc.
- The user selects an active device in the UI; omit device_id unless you must override it.
- After tools return, give a short natural language summary for the user.

High-level natural language: infer the user's goal and run the right tool sequence yourself (no need to ask for technical ids first). Examples: "add John Mayer to my Workout playlist" → spotify_user_playlists to find Workout's id, spotify_search for tracks, spotify_add_tracks_to_playlist. "Create a chill mix with …" → spotify_create_playlist then search then add. "What's on my running list?" → user_playlists / get_playlist / playlist_tracks. "My liked songs" → spotify_user_saved_tracks. "My top artists / favorite artists / who do I listen to most" → spotify_top_artists. "My top songs / most played tracks" → spotify_top_tracks. "Artists I follow" → spotify_followed_artists. "Show me <user_id>'s playlists" → spotify_user_public_playlists. "Find me a playlist about <description>" → spotify_search_playlists, then optionally spotify_follow_playlist or spotify_play_playlist. "Copy <someone else's playlist> so I can edit it" → spotify_duplicate_playlist, then edit with spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks on the new id. "Most popular album" → search + get_album / artist_top_tracks and explain the metric. On tool errors, read detail/hint and retry with a corrected plan when possible.
For create-then-add-then-play: playlist_id = create response `id` or `playlist_id_for_add_tracks`. Pass search results as `tracks` (array of tracks.items objects), or the whole search `tracks` object `{items: [...]}` — the server unwraps `items`. Start playback with context_uri `spotify:playlist:<id>`. On add failure: obey suggest_sign_out_of_spotify; if false, retry tools — never sign-out advice. Do not say "usually permissions." """


# Compressed system prompt actually sent to Gemini. Keep this under ~3000 chars.
# DO NOT pad this back up: every operational rule you add here measurably raises
# the empty-content rate in AUTO mode. Put detailed remediation guidance in the
# tool response JSON (hint, assistant_guidance, spotify_api_message) instead —
# the model reads those after each tool call.
_SYSTEM = """You are a Spotify assistant with tools to read the user's library and control playback.

Behavior rules:
- Prefer tools over guessing. Never say "I cannot access Spotify" — call a tool.
- Don't invent Spotify ids. Every track id you pass must come from a tool result this session (search / playlist_tracks / get_track).
- After a tool returns, READ its `hint`, `assistant_guidance`, `spotify_api_message`, and any flag fields like `playlist_not_owned_by_user`, `reauth_may_resolve`, `stale_scopes_need_reauth`, `missing_scopes`, `read_403_ambiguous`, `do_not_claim_ownership_issue`, `do_not_generalize_to_all_playlists`. Follow that guidance — it is more authoritative than your priors.
- HTTP 401 = re-authenticate. On 403, do not default to "Sign out / collaborative / usually permissions" unless the tool JSON explicitly says so via the flags above. Wrong playlist_id is the most common cause.
- Verify before claiming. After spotify_add_tracks_to_playlist, immediately call spotify_playlist_tracks for the same playlist and only quote songs that appear in the response. If `added_count` is present, quote it.
- After a play tool returns, if `playback_verified: false` or `ok: false`, do NOT claim playback started. Surface the error and offer spotify_transfer_playback or have the user tap play in the Spotify app.
- The user selects an active device in the UI. Omit device_id unless overriding.

Tool routing (high-level intent → tool):
- "Who am I" → spotify_me. "My playlists" → spotify_user_playlists. "What's playing / current device" → spotify_playback_state, spotify_devices.
- "Play X / start playing X / play X at track Y" → spotify_play_playlist (preferred composite) or spotify_start_resume_playback with context_uri (and offset for at-track). "Play X next / queue X" → spotify_play_next.
- "Add <artist> to <playlist>" → spotify_add_tracks_by_query (composite: search + dedupe + add). For specific titles → spotify_search then spotify_add_tracks_to_playlist (playlist_id from spotify_user_playlists or spotify_create_playlist; never from search).
- "Remove / replace / reorder / rename / unfollow playlist" → spotify_remove_playlist_tracks / spotify_replace_playlist_tracks / spotify_reorder_playlist_tracks / spotify_update_playlist / spotify_unfollow_playlist.
- "My top artists / favorite artists / most listened" → spotify_top_artists. "My top tracks / most played songs" → spotify_top_tracks. Default time_range=medium_term; "this month/lately"=short_term; "all time"=long_term.
- "Artists I follow" → spotify_followed_artists.
- "<user_id>'s playlists" → spotify_user_public_playlists. "Find a playlist about <topic>" → spotify_search_playlists → optional spotify_follow_playlist / spotify_play_playlist / spotify_duplicate_playlist.
- "Copy someone's playlist so I can edit it" → spotify_duplicate_playlist (creates a new playlist you own); then edit with spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks on the new id.
- "Repeat / loop" → spotify_set_repeat (context|track|off) AFTER playback starts. "Shuffle" → spotify_set_shuffle.

KNOWN SPOTIFY API LIMITATIONS — the Web API does NOT expose: per-playlist or per-track play counts, "most listened playlist", listening history beyond ~50 recent items, your follower list (only spotify_me.followers.total count), users you follow (only artists, via spotify_followed_artists), another user's PRIVATE playlists, lookup of a user by display name (need user_id), or editing someone else's playlist (offer spotify_duplicate_playlist instead). When asked for any of these, respond in two parts: (1) one short sentence saying what is not exposed and why, (2) 2-3 specific tools you CAN call that are closest to the intent. Never just say "I can't" — always pair it with what you can do.

DEV-MODE ENDPOINT GATES (Feb-2026 Spotify migration) — the app is in dev/non-Extended-Quota mode, so some endpoints ALWAYS return an error regardless of input. When a tool response includes `endpoint_gated_in_dev_mode: true` or `extended_quota_mode_required: true`, DO NOT retry with different arguments and DO NOT suggest sign-out. Explain the gate in one sentence and pivot to the alternatives the tool listed under `try_instead`. Specifically: `spotify_user_public_playlists` (GET /users/{id}/playlists) is fully gated for every user_id — always pivot to `spotify_search_playlists` (search playlists by topic) or `spotify_user_playlists` (the signed-in user's own playlists). Also: `spotify_artist_albums` has a hard Spotify `limit` cap of 10 per call in dev mode — if you need a total album count, read `response.total` rather than paginating.

After tools return, give a short natural-language answer for the user."""


def _schema_for_gemini(obj: Any) -> Any:
    """Gemini expects JSON Schema type enums in UPPERCASE (OBJECT, STRING, …)."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.upper()
            else:
                out[k] = _schema_for_gemini(v)
        return out
    if isinstance(obj, list):
        return [_schema_for_gemini(x) for x in obj]
    return obj


# Gemini quietly emits empty content (finishReason=STOP, parts=[]) when the
# cumulative tool schema gets large — even when the model wants to call a tool.
# Empirically, the same 40 tools that break Gemini work fine once descriptions
# are aggressively trimmed. Keep tool descriptions short here; put operational
# rules in the system prompt and in tool response JSON (`hint`, etc.) instead.
# See backend/scripts/diag_gemini.py for the bisect that established this.
_GEMINI_TOOL_DESC_MAX = 120
_GEMINI_PARAM_DESC_MAX = 40


def _trim_for_gemini(text: str, limit: int) -> str:
    if not isinstance(text, str):
        return ""
    s = text.strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    # Prefer to end on a sentence or clause boundary so descriptions still read well.
    for sep in (". ", " — ", "; ", ", "):
        idx = cut.rfind(sep)
        if idx >= int(limit * 0.5):
            return cut[: idx + 1].rstrip()
    return cut.rstrip() + "…"


def _trim_param_descriptions(params: dict[str, Any]) -> dict[str, Any]:
    """Recursively trim `description` fields inside a parameters schema."""
    if not isinstance(params, dict):
        return params
    out: dict[str, Any] = {}
    for k, v in params.items():
        if k == "description" and isinstance(v, str):
            out[k] = _trim_for_gemini(v, _GEMINI_PARAM_DESC_MAX)
        elif isinstance(v, dict):
            out[k] = _trim_param_descriptions(v)
        elif isinstance(v, list):
            out[k] = [_trim_param_descriptions(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


def _openai_tools_to_gemini_declarations(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    decls: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function")
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        desc = fn.get("description", "")
        params = fn.get("parameters")
        if not isinstance(name, str) or not name.strip():
            continue
        entry: dict[str, Any] = {
            "name": name,
            "description": _trim_for_gemini(str(desc or ""), _GEMINI_TOOL_DESC_MAX),
        }
        if isinstance(params, dict) and params:
            trimmed = _trim_param_descriptions(params)
            entry["parameters"] = _schema_for_gemini(trimmed)
        else:
            entry["parameters"] = {"type": "OBJECT", "properties": {}}
        decls.append(entry)
    return decls


def _function_response_struct(result: str) -> dict[str, Any]:
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        return {"result": result}


def _user_message_wants_spotify_data(text: str) -> bool:
    t = text.lower()
    return any(
        k in t
        for k in (
            "spotify",
            "playlist",
            "album",
            "artist",
            "track",
            "song",
            "my library",
            "playback",
            "device",
            "queue",
            "listen",
            "play ",
        )
    )


_GEMINI_TOOL_NUDGE = (
    "Spot-AI-fy: You did not call any Spotify tools yet. The user question requires live Spotify data. "
    "Decompose high-level requests: playlist by name → spotify_user_playlists; artist/tracks → spotify_search; "
    "add tracks → owned playlist id + spotify_add_tracks_by_query (preferred) or spotify_add_tracks_to_playlist; play → spotify_play_playlist (preferred) or spotify_start_resume_playback. "
    "Call the tools that fit the intent, then answer."
)


def run_chat_turn_gemini(
    user_text: str,
    settings: Settings,
    history: list[dict[str, str]] | None = None,
) -> str:
    from spot_backend.agent import _coerce_chat_history

    key = (settings.gemini_api_key or "").strip()
    if not key:
        return "GEMINI_API_KEY is not set. Add it to backend/.env (from Google AI Studio)."

    from spot_backend.llm_prefs import read_effective_gemini_model

    model = read_effective_gemini_model(settings.data_dir, settings.gemini_model) or _DEFAULT_GEMINI_MODEL
    declarations = _openai_tools_to_gemini_declarations(OLLAMA_TOOLS)
    runner = SpotifyToolRunner(settings=settings)
    full_system = _SYSTEM + load_optional_agent_context_markdown(settings)

    hist = _coerce_chat_history(history)
    contents: list[dict[str, Any]] = []
    for turn in hist:
        gem_role = "user" if turn["role"] == "user" else "model"
        contents.append({"role": gem_role, "parts": [{"text": turn["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_text}]})

    spotify_intent_blob = " ".join([user_text] + [t["content"] for t in hist[-10:]])

    url = f"{_GEMINI_REST}/models/{model}:generateContent"
    params = {"key": key}

    had_tool_results = False
    # Gemini 2.5-flash with our 40-tool catalog is *unreliable* in AUTO function-
    # calling mode — measured empty-content rate is 12/15 (80%) even with a
    # short system prompt. ANY mode forces the model to emit a tool call, which
    # is 0/15 empty. So we use:
    #   - ANY  on the FIRST round of an obvious-Spotify question (forces a call)
    #   - AUTO on subsequent rounds (so the model can write a text answer
    #          instead of being forced into yet another tool call after results)
    #   - AUTO for chitchat that doesn't look like a Spotify request
    # See backend/scripts/diag_gemini_repeat.py for the matrix that proved this.
    wants_spotify = _user_message_wants_spotify_data(spotify_intent_blob)
    # Gemini 2.5 also occasionally returns empty content stochastically; the
    # same body retried almost always succeeds. Keep a small retry budget.
    _empty_content_retries = 2
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            for _ in range(settings.agent_max_steps):
                if wants_spotify and not had_tool_results:
                    fc_mode = "ANY"
                else:
                    fc_mode = "AUTO"
                body: dict[str, Any] = {
                    "systemInstruction": {"parts": [{"text": full_system}]},
                    "contents": contents,
                    "tools": [{"functionDeclarations": declarations}],
                    "toolConfig": {"functionCallingConfig": {"mode": fc_mode}},
                }

                resp = _gemini_post_with_retry(client, url, params=params, json_body=body)
                data = resp.json()

                cands = data.get("candidates")
                if not isinstance(cands, list) or not cands:
                    pf = data.get("promptFeedback") or {}
                    return f"No response from Gemini. promptFeedback={json.dumps(pf)[:800]}"

                cand = cands[0]
                if not isinstance(cand, dict):
                    return "Unexpected Gemini response shape."

                fr = cand.get("finishReason")
                if fr in ("SAFETY", "RECITATION"):
                    return (
                        "Gemini's safety filter blocked that turn. Please rephrase, or switch to "
                        "Ollama in Settings if you need to keep going."
                    )
                if fr == "MAX_TOKENS":
                    # Gemini 2.5 thinking can quietly exhaust the response budget on its
                    # internal reasoning, leaving content.parts empty. Tell the user plainly
                    # and steer them at the lighter model that doesn't have that failure mode.
                    logger.warning(
                        "gemini_finish_reason_max_tokens model=%s usage=%s",
                        model,
                        data.get("usageMetadata"),
                    )
                    return (
                        f"Gemini ({model}) ran out of response budget while reasoning, before it "
                        "produced a visible reply. This is a known quirk of the 2.5 'thinking' "
                        "models on long prompts. Please try again, switch to gemini-2.5-flash-lite "
                        "or gemini-1.5-flash in Settings, or switch to Ollama in Settings."
                    )

                c_content = cand.get("content")
                if not isinstance(c_content, dict):
                    logger.warning(
                        "gemini_no_content model=%s finish_reason=%s candidate=%s",
                        model,
                        fr,
                        json.dumps(cand)[:1200],
                    )
                    return (
                        "Gemini returned an empty response. Please try again, or switch to a "
                        "different model in Settings if it keeps happening."
                    )

                parts = c_content.get("parts")
                if not isinstance(parts, list):
                    parts = []

                model_parts_out: list[dict[str, Any]] = []
                fr_parts: list[dict[str, Any]] = []
                visible_text_chunks: list[str] = []

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    is_thought = bool(part.get("thought"))
                    text_val = part.get("text") if "text" in part else None
                    if isinstance(text_val, str):
                        # Gemini 2.5 thinking models echo their reasoning as parts with
                        # thought=true. Keep them in the model turn so Gemini can chain
                        # reasoning across rounds, but DON'T treat them as the user-visible
                        # final answer (else we'd surface raw chain-of-thought to the user).
                        model_parts_out.append({"text": text_val})
                        if not is_thought and text_val.strip():
                            visible_text_chunks.append(text_val)
                    fc = part.get("functionCall")
                    if isinstance(fc, dict) and fc.get("name"):
                        model_parts_out.append({"functionCall": fc})
                        name = str(fc["name"])
                        raw_args = fc.get("args")
                        args: dict[str, Any] = {}
                        if isinstance(raw_args, dict):
                            args = raw_args
                        result = runner.run(name, args)
                        fr_parts.append(
                            {
                                "functionResponse": {
                                    "name": name,
                                    "response": _function_response_struct(result),
                                }
                            }
                        )

                if model_parts_out:
                    contents.append({"role": "model", "parts": model_parts_out})

                if fr_parts:
                    contents.append({"role": "user", "parts": fr_parts})
                    had_tool_results = True
                    continue

                joined = "\n".join(t for t in visible_text_chunks if isinstance(t, str) and t.strip()).strip()
                if joined:
                    if not had_tool_results and _user_message_wants_spotify_data(spotify_intent_blob):
                        contents.append({"role": "user", "parts": [{"text": _GEMINI_TOOL_NUDGE}]})
                        continue
                    return joined

                # No visible text and no tool calls. Log everything we have so we can
                # diagnose schema rejections, thought-only responses, etc.
                logger.warning(
                    "gemini_empty_content model=%s finish_reason=%s n_parts=%s n_thought_parts=%s "
                    "had_tool_results=%s candidate=%s usage=%s",
                    model,
                    fr,
                    len(parts),
                    sum(1 for p in parts if isinstance(p, dict) and p.get("thought")),
                    had_tool_results,
                    json.dumps(cand)[:1500],
                    data.get("usageMetadata"),
                )
                # Gemini 2.5 with a large tool catalog sometimes returns empty
                # content on a borderline tool-selection. A plain retry of the
                # same body almost always succeeds (we've measured this).
                if _empty_content_retries > 0:
                    _empty_content_retries -= 1
                    logger.info("gemini_empty_content_retry remaining=%s", _empty_content_retries)
                    continue
                # If the model returned ONLY thought parts on the first turn for an
                # obvious Spotify question, give it one chance to actually act.
                if not had_tool_results and _user_message_wants_spotify_data(spotify_intent_blob):
                    contents.append({"role": "user", "parts": [{"text": _GEMINI_TOOL_NUDGE}]})
                    continue
                return (
                    "Gemini didn't return any text on that turn. Please rephrase or try again, "
                    "or switch to Ollama in Settings if it keeps happening."
                )

        return (
            "I made several tool calls trying to answer that, but couldn't finish in time. "
            "Try a simpler or more specific request, or break it into smaller steps."
        )
    except httpx.HTTPStatusError as e:
        return _gemini_friendly_error_message(e, model)
    except httpx.TimeoutException:
        return (
            f"Gemini took too long to respond just now. This is usually a temporary network or "
            f"capacity hiccup. Please try again in a moment, pick a lighter model from the "
            f"Settings dropdown (e.g. gemini-2.5-flash-lite or gemini-1.5-flash instead of "
            f"{model}), or switch to Ollama in Settings."
        )
    except httpx.HTTPError:
        return (
            "I couldn't reach Gemini just now (network error talking to Google's API). "
            "Please try again in a minute, or switch to Ollama in Settings if it keeps happening."
        )
    finally:
        runner.close()


def _gemini_friendly_error_message(exc: httpx.HTTPStatusError, model: str) -> str:
    """Translate a Gemini HTTP error into a single human-readable sentence + a recommendation.

    We deliberately avoid leaking raw HTTP status codes, JSON snippets, or Google URLs into
    the chat — the user wanted plain language and a clear next step, not stack traces.
    """
    code = exc.response.status_code
    if code == 503:
        return (
            f"Gemini is overloaded right now — Google has been returning 'service unavailable' "
            f"for {model} even after a few automatic retries. Please try again in a minute, "
            f"pick a lighter model from the Settings dropdown (e.g. gemini-2.5-flash-lite or "
            f"gemini-1.5-flash), or switch to Ollama in Settings to keep going."
        )
    if code == 429:
        return (
            f"Gemini hit its rate limit / quota for {model}. On the free tier, this typically "
            f"resets daily. Try a lighter model from the Settings dropdown (e.g. gemini-2.5-flash-lite), "
            f"switch to Ollama in Settings, or enable billing on your Google AI key if you need more headroom."
        )
    if code == 404:
        return (
            f"The Gemini model {model} isn't available for your API key right now (it may have "
            f"been retired, or your key isn't enabled for it). Pick a different model from the "
            f"Settings dropdown — {_DEFAULT_GEMINI_MODEL} is a safe default."
        )
    if code in (401, 403):
        return (
            "Google rejected the Gemini API key (it's missing, expired, or doesn't have access "
            "to this model). Please double-check GEMINI_API_KEY in backend/.env and restart the "
            "backend, or switch to Ollama in Settings if you don't have a working key handy."
        )
    if 500 <= code < 600:
        return (
            f"Gemini is having a server-side issue right now. Please try again in a minute, "
            f"pick a different model from the Settings dropdown, or switch to Ollama in Settings."
        )
    return (
        "Gemini ran into an unexpected problem on that request. Please try again in a moment, "
        "pick a different model from the Settings dropdown, or switch to Ollama in Settings."
    )
