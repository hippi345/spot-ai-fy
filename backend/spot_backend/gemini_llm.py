"""Spot-AI-fy: Gemini HTTP backend for chat + Spotify tools."""

from __future__ import annotations

import json
from typing import Any

import httpx

from spot_backend.config import Settings
from spot_backend.context_loader import load_optional_agent_context_markdown
from spot_backend.spotify_tools import OLLAMA_TOOLS, SpotifyToolRunner

_GEMINI_REST = "https://generativelanguage.googleapis.com/v1beta"
_DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

_SYSTEM = """You are a Spotify assistant with tools to read the user's library and control playback.
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
- The user selects an active device in the UI; omit device_id unless you must override it.
- After tools return, give a short natural language summary for the user.

High-level natural language: infer the user's goal and run the right tool sequence yourself (no need to ask for technical ids first). Examples: "add John Mayer to my Workout playlist" → spotify_user_playlists to find Workout's id, spotify_search for tracks, spotify_add_tracks_to_playlist. "Create a chill mix with …" → spotify_create_playlist then search then add. "What's on my running list?" → user_playlists / get_playlist / playlist_tracks. "My liked songs" → spotify_user_saved_tracks. "Most popular album" → search + get_album / artist_top_tracks and explain the metric. On tool errors, read detail/hint and retry with a corrected plan when possible.
For create-then-add-then-play: playlist_id = create response `id` or `playlist_id_for_add_tracks`. Pass search results as `tracks` (array of tracks.items objects), or the whole search `tracks` object `{items: [...]}` — the server unwraps `items`. Start playback with context_uri `spotify:playlist:<id>`. On add failure: obey suggest_sign_out_of_spotify; if false, retry tools — never sign-out advice. Do not say "usually permissions." """


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
        entry: dict[str, Any] = {"name": name, "description": str(desc or "")}
        if isinstance(params, dict) and params:
            entry["parameters"] = _schema_for_gemini(params)
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
    try:
        with httpx.Client(timeout=httpx.Timeout(120.0, connect=30.0)) as client:
            for _ in range(settings.agent_max_steps):
                body: dict[str, Any] = {
                    "systemInstruction": {"parts": [{"text": full_system}]},
                    "contents": contents,
                    "tools": [{"functionDeclarations": declarations}],
                    "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
                }

                resp = client.post(url, params=params, json=body)
                resp.raise_for_status()
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
                    return f"Gemini stopped ({fr}). Try rephrasing your request."

                c_content = cand.get("content")
                if not isinstance(c_content, dict):
                    return "Gemini returned no content."

                parts = c_content.get("parts")
                if not isinstance(parts, list):
                    parts = []

                model_parts_out: list[dict[str, Any]] = []
                fr_parts: list[dict[str, Any]] = []

                for part in parts:
                    if not isinstance(part, dict):
                        continue
                    if "text" in part:
                        model_parts_out.append({"text": part.get("text", "")})
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

                texts = [p.get("text", "") for p in model_parts_out if isinstance(p, dict) and "text" in p]
                joined = "\n".join(t for t in texts if isinstance(t, str) and t.strip()).strip()
                if joined:
                    if not had_tool_results and _user_message_wants_spotify_data(spotify_intent_blob):
                        contents.append({"role": "user", "parts": [{"text": _GEMINI_TOOL_NUDGE}]})
                        continue
                    return joined

                return "Gemini returned an empty reply."

        return "Stopped after maximum tool steps. Try a simpler request."
    except httpx.HTTPStatusError as e:
        code = e.response.status_code
        snippet = (e.response.text or "")[:500]
        if code == 404:
            return (
                f"Gemini HTTP 404: model {model!r} is not available for this API (name retired or wrong for your key). "
                f"Set GEMINI_MODEL in backend/.env to a current id (e.g. {_DEFAULT_GEMINI_MODEL!r} or gemini-1.5-flash). "
                f"List models: GET {_GEMINI_REST}/models?key=YOUR_KEY — "
                "https://ai.google.dev/gemini-api/docs/models"
            )
        return f"Gemini HTTP {code}: {snippet or str(e)}"
    finally:
        runner.close()
