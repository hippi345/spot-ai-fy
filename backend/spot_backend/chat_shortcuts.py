"""Deterministic chat shortcuts (like/save this, undo, play artist) without relying on the LLM."""

from __future__ import annotations

import json
import logging
import re

from spot_backend.chat_shortcut_policy import prompt_blocks_deterministic_play_shortcuts
from spot_backend.deterministic_chat_types import DeterministicChatResult
from spot_backend.play_bare import format_play_bare_chat_reply, play_bare_tool_step
from spot_backend.play_bare_intent import extract_bare_play_target, extract_play_music_by_artist
from spot_backend.play_artist import format_play_artist_reply, play_artist_tool_step
from spot_backend.play_track import format_play_track_chat_reply, play_track_tool_step
from spot_backend.play_track_intent import extract_play_track_request
from spot_backend.play_mood_intent import extract_play_something_mood
from spot_backend.playback_reply import (
    format_now_playing_chat_reply,
    format_playlist_play_chat_reply,
    format_skip_reply,
    prompt_asks_previous,
    prompt_asks_skip,
    prompt_asks_whats_playing,
)
from spot_backend.library_mutation_store import (
    load_played_playlist_ids,
    record_played_playlist_id,
)
from spot_backend.playlist_pick import fetch_owned_playlist_candidates_paginated, playlist_id_is_spotify_curated
from spot_backend.prompt_intent import prompt_is_surprise_me_request
from spot_backend.spotify_dev_limits import SPOTIFY_DEV_MAX_PAGE
from spot_backend.prompt_intent import (
    prompt_is_alternate_owned_playlist_play_request,
    prompt_is_vague_playlist_play_request,
    prompt_requests_current_track_release,
    prompt_requests_recent_listening_history,
)
from spot_backend.queue_track_intent import extract_queue_track_request
from spot_backend.spotify_tools import SpotifyToolRunner

_LIKE_THIS_RE = re.compile(
    r"^\s*(?:please\s+)?(?:(?:like|heart)\s+this|save\s+this(?:\s+(?:song|track))?)\s*[.!?]*\s*$",
    re.I,
)
_SAVE_SONG_RE = re.compile(
    r"^\s*(?:please\s+)?save\s+this\s+(?:song|track)\s*[.!?]*\s*$",
    re.I,
)
_UNDO_RE = re.compile(r"^\s*(?:please\s+)?undo\s+that\s*[.!?]*\s*$", re.I)

_NOTHING_TO_UNDO = "There's nothing to undo yet."
_LOGGER = logging.getLogger(__name__)

_WHAT_PLAYLISTS_RE = re.compile(
    r"^\s*what playlists do i have\s*\??\s*$",
    re.I,
)


def _parse_tool_json(raw: str) -> dict:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        data = {}
    if isinstance(data, dict) and data.get("items"):
        return data
    # Truncated tool payloads: recover a partial items array if present.
    marker = '"items"'
    idx = (raw or "").find(marker)
    if idx >= 0:
        fragment = (raw or "")[idx:]
        if not fragment.strip().endswith("}"):
            fragment = fragment.rsplit("}", 1)[0] + "]}"
        try:
            wrapped = json.loads("{" + fragment)
        except (json.JSONDecodeError, TypeError, ValueError):
            wrapped = {}
        if isinstance(wrapped, dict) and isinstance(wrapped.get("items"), list):
            return wrapped
    return data if isinstance(data, dict) else {}


def _summarize_recent_plays(items: list) -> str:
    labels: list[str] = []
    for row in items[:12]:
        if not isinstance(row, dict):
            continue
        track = row.get("track") if isinstance(row.get("track"), dict) else row
        if not isinstance(track, dict):
            continue
        name = track.get("name") if isinstance(track.get("name"), str) else "Unknown track"
        artists = track.get("artists") if isinstance(track.get("artists"), list) else []
        artist_names = [
            a.get("name")
            for a in artists
            if isinstance(a, dict) and isinstance(a.get("name"), str)
        ]
        if artist_names:
            labels.append(f"{name} — {', '.join(artist_names)}")
        else:
            labels.append(name)
    if not labels:
        return ""
    if len(labels) == 1:
        return f"Your most recent play was {labels[0]}."
    head = ", ".join(labels[:8])
    extra = len(items) - len(labels[:8])
    if extra > 0:
        return f"Recently you played: {head}, and {extra} more."
    return f"Recently you played: {head}."


def _run_tool(
    runner: SpotifyToolRunner,
    name: str,
    args: dict,
    steps: list[tuple[str, dict, str]],
) -> str:
    raw = runner.run(name, args)
    steps.append((name, args, raw))
    return raw


def try_deterministic_recently_played_reply(
    user_text: str,
    runner: SpotifyToolRunner,
) -> DeterministicChatResult | None:
    if not prompt_requests_recent_listening_history(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    args = {"limit": SPOTIFY_DEV_MAX_PAGE}
    raw = _run_tool(runner, "spotify_recently_played", args, steps)
    data = _parse_tool_json(raw)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if items:
        summary = _summarize_recent_plays(items)
        reply = summary or (
            f"I pulled your {len(items)} most recent plays from Spotify — "
            "check the tool results for track details."
        )
    else:
        reply = "I couldn't find any recent listening history in your Spotify account just now."
    return DeterministicChatResult(reply, steps)


def try_deterministic_current_track_release_reply(
    user_text: str,
    runner: SpotifyToolRunner,
) -> DeterministicChatResult | None:
    if not prompt_requests_current_track_release(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    state_raw = _run_tool(runner, "spotify_playback_state", {}, steps)
    state = _parse_tool_json(state_raw)
    item = state.get("item") if isinstance(state.get("item"), dict) else {}
    if not item:
        return DeterministicChatResult(
            "Nothing is playing right now, so I can't tell when the current song came out.",
            steps,
        )
    album = item.get("album") if isinstance(item.get("album"), dict) else {}
    release = album.get("release_date") if isinstance(album.get("release_date"), str) else ""
    track_name = item.get("name") if isinstance(item.get("name"), str) else "This track"
    if not release and isinstance(album.get("id"), str):
        album_raw = _run_tool(
            runner,
            "spotify_get_album",
            {"album_id": album["id"]},
            steps,
        )
        album_data = _parse_tool_json(album_raw)
        rd = album_data.get("release_date")
        if isinstance(rd, str):
            release = rd
    if release:
        return DeterministicChatResult(f"{track_name} came out on {release}.", steps)
    return DeterministicChatResult(
        f"I couldn't find a release date for {track_name} just now.",
        steps,
    )


def _try_owned_playlist_play_reply(
    runner: SpotifyToolRunner,
    steps: list[tuple[str, dict, str]],
    *,
    conversation_id: str | None,
    exclude_ids: set[str] | None = None,
    exclude_current_context: bool = False,
    offer_alternate: bool,
    empty_message: str,
    failure_message: str,
) -> DeterministicChatResult:
    me_raw = _run_tool(runner, "spotify_me", {}, steps)
    me = _parse_tool_json(me_raw)
    me_id = me.get("id") if isinstance(me.get("id"), str) else ""
    skip = set(exclude_ids or set())
    skip.update(load_played_playlist_ids(conversation_id))
    if exclude_current_context:
        state_raw = _run_tool(runner, "spotify_playback_state", {}, steps)
        state = _parse_tool_json(state_raw)
        ctx = state.get("context") if isinstance(state.get("context"), dict) else {}
        uri = ctx.get("uri") if isinstance(ctx.get("uri"), str) else ""
        if uri.lower().startswith("spotify:playlist:"):
            skip.add(uri.rsplit(":", 1)[-1].strip())
    probe_cache: dict[str, bool] = {}

    def _fetch_pl(args: dict[str, int]) -> str:
        return _run_tool(runner, "spotify_user_playlists", args, steps)

    def _probe_items(playlist_id: str) -> str:
        return _run_tool(
            runner,
            "spotify_playlist_tracks",
            {"playlist_id": playlist_id, "limit": 1},
            steps,
        )

    candidates, _pl_steps = fetch_owned_playlist_candidates_paginated(
        _fetch_pl,
        me_id,
        exclude_ids=skip,
        probe_cache=probe_cache,
        run_probe=_probe_items,
    )
    if not candidates:
        return DeterministicChatResult(empty_message, steps)
    attempts = min(3, len(candidates))
    for row in candidates[:attempts]:
        pid = row.get("id")
        pname = row.get("name") if isinstance(row.get("name"), str) else "your playlist"
        if not isinstance(pid, str):
            continue
        play_raw = _run_tool(
            runner,
            "spotify_play_playlist",
            {"playlist_id": pid, "playback_request_label": pname},
            steps,
        )
        play = _parse_tool_json(play_raw)
        verified = bool(play.get("playback_verified"))
        if play.get("ok") and verified:
            record_played_playlist_id(conversation_id, pid)
            reply = format_playlist_play_chat_reply(
                pname,
                play_raw,
                offer_alternate=offer_alternate,
            )
            return DeterministicChatResult(reply, steps)
        err_body = play.get("spotify_error_body_redacted")
        if isinstance(err_body, str) and err_body.strip():
            _LOGGER.warning(
                "owned_playlist_play_failed playlist_id=%s body=%s",
                pid,
                err_body[:400],
            )
    return DeterministicChatResult(failure_message, steps)


def try_deterministic_surprise_me_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None = None,
) -> DeterministicChatResult | None:
    if not prompt_is_surprise_me_request(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    owned = _try_owned_playlist_play_reply(
        runner,
        steps,
        conversation_id=conversation_id,
        exclude_current_context=False,
        offer_alternate=False,
        empty_message="",
        failure_message="",
    )
    if owned.reply:
        return owned
    top_raw = _run_tool(
        runner,
        "spotify_top_tracks",
        {"limit": SPOTIFY_DEV_MAX_PAGE, "time_range": "short_term"},
        steps,
    )
    top = _parse_tool_json(top_raw)
    items = top.get("items") if isinstance(top.get("items"), list) else []
    if items and isinstance(items[0], dict):
        tr = items[0]
        uri = tr.get("uri")
        if isinstance(uri, str):
            play_raw = _run_tool(
                runner,
                "spotify_start_resume_playback",
                {"uris": [uri]},
                steps,
            )
            play = _parse_tool_json(play_raw)
            if play.get("ok"):
                name = str(tr.get("name") or "a top track")
                return DeterministicChatResult(f"Playing {name} from your recent favorites.", steps)
    saved_raw = _run_tool(runner, "spotify_user_saved_tracks", {"limit": 1}, steps)
    saved = _parse_tool_json(saved_raw)
    saved_items = saved.get("items") if isinstance(saved.get("items"), list) else []
    if saved_items and isinstance(saved_items[0], dict):
        row = saved_items[0]
        track = row.get("track") if isinstance(row.get("track"), dict) else row
        uri = track.get("uri") if isinstance(track, dict) else None
        if isinstance(uri, str):
            play_raw = _run_tool(
                runner,
                "spotify_start_resume_playback",
                {"uris": [uri]},
                steps,
            )
            play = _parse_tool_json(play_raw)
            if play.get("ok"):
                title = str(track.get("name") or "a liked song")
                return DeterministicChatResult(f"Playing {title} from your liked songs.", steps)
    recent_raw = _run_tool(runner, "spotify_recently_played", {"limit": 1}, steps)
    recent = _parse_tool_json(recent_raw)
    recent_items = recent.get("items") if isinstance(recent.get("items"), list) else []
    if recent_items and isinstance(recent_items[0], dict):
        track = recent_items[0].get("track")
        if isinstance(track, dict) and isinstance(track.get("uri"), str):
            play_raw = _run_tool(
                runner,
                "spotify_start_resume_playback",
                {"uris": [track["uri"]]},
                steps,
            )
            play = _parse_tool_json(play_raw)
            if play.get("ok"):
                title = str(track.get("name") or "something recent")
                return DeterministicChatResult(f"Playing {title} from your recent listens.", steps)
    return DeterministicChatResult(
        "I couldn't find anything in your library to surprise you with just now.",
        steps,
    )


def try_deterministic_vague_playlist_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None = None,
) -> DeterministicChatResult | None:
    if not prompt_is_vague_playlist_play_request(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    return _try_owned_playlist_play_reply(
        runner,
        steps,
        conversation_id=conversation_id,
        exclude_current_context=False,
        offer_alternate=True,
        empty_message="I couldn't find a playlist in your library that I can play from here.",
        failure_message="I couldn't start a playlist just now.",
    )


def try_deterministic_alternate_playlist_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None = None,
) -> DeterministicChatResult | None:
    if not prompt_is_alternate_owned_playlist_play_request(user_text):
        return None
    steps: list[tuple[str, dict, str]] = []
    return _try_owned_playlist_play_reply(
        runner,
        steps,
        conversation_id=conversation_id,
        exclude_current_context=True,
        offer_alternate=True,
        empty_message="I couldn't find another playlist in your library to play.",
        failure_message="I couldn't switch to another playlist just now.",
    )


def try_deterministic_list_playlists_reply(
    user_text: str,
    runner: SpotifyToolRunner,
) -> DeterministicChatResult | None:
    if not _WHAT_PLAYLISTS_RE.match((user_text or "").strip()):
        return None
    steps: list[tuple[str, dict, str]] = []
    raw = _run_tool(runner, "spotify_user_playlists", {"limit": SPOTIFY_DEV_MAX_PAGE}, steps)
    data = _parse_tool_json(raw)
    total = data.get("total")
    items = data.get("items") if isinstance(data.get("items"), list) else []
    names: list[str] = []
    for row in items[:6]:
        if isinstance(row, dict) and isinstance(row.get("name"), str):
            name = row["name"].strip()
            if name:
                names.append(name)
    if isinstance(total, int):
        head = f"You have {total} playlists"
    else:
        head = "Here are your playlists"
    if names:
        sample = ", ".join(names[:5])
        reply = f"{head}, including {sample}."
    else:
        reply = f"{head}."
    return DeterministicChatResult(reply, steps)


def try_deterministic_mood_play_reply(
    user_text: str,
    runner: SpotifyToolRunner,
) -> DeterministicChatResult | None:
    mood = extract_play_something_mood(user_text)
    if not mood:
        return None
    steps: list[tuple[str, dict, str]] = []
    search_raw = _run_tool(
        runner,
        "spotify_search_playlists",
        {"query": mood, "limit": 5},
        steps,
    )
    search = _parse_tool_json(search_raw)
    items = search.get("items") if isinstance(search.get("items"), list) else None
    if not items:
        playlists = search.get("playlists")
        items = playlists.get("items") if isinstance(playlists, dict) else None
    if not isinstance(items, list) or not items:
        search_raw = _run_tool(
            runner,
            "spotify_search",
            {"query": mood, "types": "track", "limit": 5},
            steps,
        )
        search = _parse_tool_json(search_raw)
        tracks = search.get("tracks")
        track_items = tracks.get("items") if isinstance(tracks, dict) else None
        if not isinstance(track_items, list) or not track_items:
            return DeterministicChatResult(
                f"I couldn't find anything {mood} to play just now.",
                steps,
            )
        first = track_items[0]
        if not isinstance(first, dict):
            return DeterministicChatResult(
                f"I couldn't find anything {mood} to play just now.",
                steps,
            )
        title = str(first.get("name") or mood).strip()
        artists = first.get("artists") if isinstance(first.get("artists"), list) else []
        artist = ""
        if artists and isinstance(artists[0], dict):
            artist = str(artists[0].get("name") or "").strip()
        name, args, raw = play_track_tool_step(runner, title, artist)
        steps.append((name, args, raw))
        reply = format_play_track_chat_reply(title, artist, raw)
        if reply.lower().startswith("playing"):
            return DeterministicChatResult(
                f"{reply.rstrip('.')} — want something different?",
                steps,
            )
        return DeterministicChatResult(reply, steps)
    first_pl = None
    for candidate in items:
        if not isinstance(candidate, dict):
            continue
        pid_c = candidate.get("id")
        if isinstance(pid_c, str) and playlist_id_is_spotify_curated(pid_c):
            continue
        owner = candidate.get("owner") if isinstance(candidate.get("owner"), dict) else {}
        if str(owner.get("id") or "").strip().lower() == "spotify":
            continue
        first_pl = candidate
        break
    if first_pl is None:
        first_pl = items[0] if items else None
    if not isinstance(first_pl, dict):
        return DeterministicChatResult(
            f"I couldn't find a {mood} playlist to play just now.",
            steps,
        )
    pid = first_pl.get("id")
    if isinstance(pid, str) and playlist_id_is_spotify_curated(pid):
        return DeterministicChatResult(
            f"I couldn't find a {mood} playlist to play just now.",
            steps,
        )
    pname = str(first_pl.get("name") or mood).strip()
    if not isinstance(pid, str):
        return DeterministicChatResult(
            f"I couldn't find a {mood} playlist to play just now.",
            steps,
        )
    play_raw = _run_tool(
        runner,
        "spotify_play_playlist",
        {"playlist_id": pid, "playback_request_label": pname},
        steps,
    )
    play = _parse_tool_json(play_raw)
    if play.get("ok") and play.get("playback_verified"):
        reply = format_playlist_play_chat_reply(pname, play_raw, offer_alternate=True)
        return DeterministicChatResult(reply, steps)
    return DeterministicChatResult(
        format_playlist_play_chat_reply(pname, play_raw, offer_alternate=False),
        steps,
    )


def try_deterministic_chat_reply(
    user_text: str,
    runner: SpotifyToolRunner,
    *,
    conversation_id: str | None = None,
) -> DeterministicChatResult | None:
    """Run a fixed tool chain for obvious control phrases; return structured outcome or None."""
    t = (user_text or "").strip()
    if not t:
        return None

    steps: list[tuple[str, dict, str]] = []

    release = try_deterministic_current_track_release_reply(user_text, runner)
    if release is not None:
        return release

    if _UNDO_RE.match(t):
        track_ids = runner.last_saved_track_ids_for_undo(
            conversation_id or runner.conversation_id
        )
        if not track_ids:
            return DeterministicChatResult(_NOTHING_TO_UNDO, [])
        tid = track_ids[-1]
        raw = _run_tool(runner, "spotify_unsave_tracks", {"track_id": tid}, steps)
        data = _parse_tool_json(raw)
        if data.get("ok"):
            removed = data.get("removed_track_ids") or data.get("track_ids") or []
            if isinstance(removed, list) and removed:
                return DeterministicChatResult("Removed that track from your liked songs.", steps)
            return DeterministicChatResult("Undid the last save to your library.", steps)
        return DeterministicChatResult(
            "I could not undo the last action — nothing was changed.",
            steps,
        )

    if _WHAT_PLAYLISTS_RE.match(t):
        listed = try_deterministic_list_playlists_reply(user_text, runner)
        if listed is not None:
            return listed

    surprise = try_deterministic_surprise_me_reply(
        user_text, runner, conversation_id=conversation_id
    )
    if surprise is not None:
        return surprise

    mood = try_deterministic_mood_play_reply(user_text, runner)
    if mood is not None:
        return mood

    if prompt_asks_whats_playing(t):
        state_raw = _run_tool(runner, "spotify_playback_state", {}, steps)
        try:
            state = json.loads(state_raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            state = {}
        player = state if isinstance(state, dict) else {}
        return DeterministicChatResult(format_now_playing_chat_reply(player), steps)

    if prompt_asks_previous(t):
        raw = _run_tool(runner, "spotify_skip_previous", {}, steps)
        data = _parse_tool_json(raw)
        player = data.get("player_after")
        skipped = bool(data.get("skipped"))
        return DeterministicChatResult(
            format_skip_reply(
                player if isinstance(player, dict) else None,
                skipped=skipped,
                direction="previous",
            ),
            steps,
        )

    if prompt_asks_skip(t):
        raw = _run_tool(runner, "spotify_skip_next", {}, steps)
        data = _parse_tool_json(raw)
        player = data.get("player_after")
        skipped = bool(data.get("skipped"))
        return DeterministicChatResult(
            format_skip_reply(player if isinstance(player, dict) else None, skipped=skipped),
            steps,
        )

    if prompt_is_alternate_owned_playlist_play_request(t):
        alt = try_deterministic_alternate_playlist_reply(
            t,
            runner,
            conversation_id=conversation_id,
        )
        if alt is not None:
            return alt

    if prompt_is_vague_playlist_play_request(t):
        vague = try_deterministic_vague_playlist_reply(
            t,
            runner,
            conversation_id=conversation_id,
        )
        if vague is not None:
            return vague

    track_req = extract_play_track_request(t) if not prompt_blocks_deterministic_play_shortcuts(t) else None
    if track_req:
        track_title, track_artist = track_req
        name, args, raw = play_track_tool_step(runner, track_title, track_artist)
        steps.append((name, args, raw))
        return DeterministicChatResult(
            format_play_track_chat_reply(track_title, track_artist, raw),
            steps,
        )

    if not prompt_blocks_deterministic_play_shortcuts(t):
        by_artist = extract_play_music_by_artist(t)
        if by_artist:
            name, args, raw = play_artist_tool_step(runner, by_artist)
            steps.append((name, args, raw))
            return DeterministicChatResult(
                format_play_artist_reply(by_artist, raw),
                steps,
            )
        bare = extract_bare_play_target(t)
        if bare:
            name, args, raw = play_bare_tool_step(runner, bare)
            steps.append((name, args, raw))
            return DeterministicChatResult(format_play_bare_chat_reply(bare, raw), steps)

    queue_req = extract_queue_track_request(t) if not prompt_blocks_deterministic_play_shortcuts(t) else None
    if queue_req:
        track_title, artist = queue_req
        qargs: dict[str, str] = {"track_name": track_title}
        if artist:
            qargs["artist_name"] = artist
        raw = _run_tool(runner, "spotify_add_to_queue", qargs, steps)
        data = _parse_tool_json(raw)
        if data.get("ok"):
            label = f"{track_title} by {artist}" if artist else track_title
            return DeterministicChatResult(f"Queued {label} on Spotify.", steps)
        err = str(data.get("error") or "I could not queue that track.")
        return DeterministicChatResult(err, steps)

    if _LIKE_THIS_RE.match(t) or _SAVE_SONG_RE.match(t):
        state_raw = _run_tool(runner, "spotify_playback_state", {}, steps)
        state = _parse_tool_json(state_raw)
        item = state.get("item") if isinstance(state.get("item"), dict) else {}
        track_name = item.get("name") if isinstance(item.get("name"), str) else "that track"
        save_raw = _run_tool(runner, "spotify_save_tracks", {"track_id": "this"}, steps)
        save = _parse_tool_json(save_raw)
        if save.get("ok"):
            return DeterministicChatResult(f"Saved “{track_name}” to your liked songs.", steps)
        err = str(save.get("error") or save.get("hint") or "")
        if "playback" in err.lower() or "nothing" in err.lower():
            return DeterministicChatResult(
                "Nothing is playing right now, so I cannot save a track. "
                "Start playback first, then ask again.",
                steps,
            )
        return DeterministicChatResult(
            "I could not save that track to your library. Try again when something is playing.",
            steps,
        )

    return None
