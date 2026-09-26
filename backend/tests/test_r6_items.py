"""Round-6 PR items — dedicated test_r6_itemN_* per requirement."""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from spot_backend.config import Settings
from spot_backend.chat_messages import scrub_internal_tool_references
from spot_backend.gemini_llm import run_chat_turn_gemini
from spot_backend.prompt_intent import SPOTIFY_MUTATING_TOOL_NAMES, prompt_is_informational
from tests.recheck_helpers import (
    gemini_candidates_payload,
    gemini_stop_candidate,
    make_gemini_post_recorder,
    run_unknown_device_playback_fallback,
)

HOW_TO_INFORMATIONAL_PHRASES = (
    "how do I make a playlist private?",
    "how can I play an artist?",
    "what does shuffle do?",
    "what's the way to like a song",
    "how would I add a song to a playlist?",
    "is there a way to download my liked songs?",
    "can you explain how repeat modes work?",
    "can you explain how to follow an artist?",
    "how does queueing work on Spotify?",
    "what do I say to play an album?",
    "what should I type to pause playback?",
    "tell me how to make a playlist secret",
    "explain how shuffle works",
    "what is the way to save a track to my library?",
    "how do I turn on shuffle in the app?",
)

ACTION_NOT_INFORMATIONAL_PHRASES = (
    "make my playlist Workout private",
    "delete playlist Old Mix",
    "like this",
    "save this album",
    "play Radiohead",
    "shuffle on",
    "skip",
    "turn it up",
    "add this to Workout",
    "follow this artist",
)


@pytest.mark.parametrize("phrase", HOW_TO_INFORMATIONAL_PHRASES)
def test_r6_item1_how_to_phrasings_are_informational(phrase: str) -> None:
    assert prompt_is_informational(phrase) is True


@pytest.mark.parametrize("phrase", ACTION_NOT_INFORMATIONAL_PHRASES)
def test_r6_item1_action_phrasings_are_not_informational(phrase: str) -> None:
    assert prompt_is_informational(phrase) is False


def test_r6_item1_gemini_like_song_how_to_not_any_and_no_mutating_tools(
    data_dir, signed_in_tokens,
) -> None:
    settings = Settings(gemini_api_key="k")
    phrase = "what's the way to like a song"
    assert prompt_is_informational(phrase)

    bodies, fake_post = make_gemini_post_recorder(
        lambda body, _n, req: httpx.Response(
            200,
            json=gemini_candidates_payload(
                gemini_stop_candidate(
                    {"text": "Tap the heart icon next to the track, or type “like this” here."}
                )
            ),
            request=req,
        )
    )
    with patch("httpx.Client.post", fake_post):
        run_chat_turn_gemini(phrase, settings)

    assert bodies
    mode = (bodies[0].get("toolConfig") or {}).get("functionCallingConfig", {}).get("mode")
    assert mode != "ANY"
    decls = (bodies[0].get("tools") or [{}])[0].get("functionDeclarations") or []
    names = {d.get("name") for d in decls if isinstance(d, dict)}
    assert not names & SPOTIFY_MUTATING_TOOL_NAMES


def test_r6_item2_scrubber_keeps_song_album_playlist_titles_in_backticks() -> None:
    song = scrub_internal_tool_references("Try `Bohemian Rhapsody` by Queen.")
    assert "Bohemian Rhapsody" in song
    assert "Queen" in song
    assert "`" not in song

    album = scrub_internal_tool_references("From the album `OK Computer` you can start playback.")
    assert "OK Computer" in album

    playlist = scrub_internal_tool_references("Open `My Chill Mix` in Your Library.")
    assert "My Chill Mix" in playlist


def test_r6_item2_scrubber_still_strips_tool_parameter_and_json_spans() -> None:
    raw = (
        "Call `spotify_save_tracks` with `track_id` and `{\"uris\":[\"spotify:track:abc\"]}` "
        "or set the `public` parameter to `False`."
    )
    cleaned = scrub_internal_tool_references(raw)
    assert "spotify_save_tracks" not in cleaned
    assert "track_id" not in cleaned
    assert "uris" not in cleaned
    assert "public" not in cleaned.lower()
    assert "False" not in cleaned
    assert "`" not in cleaned


def test_r6_item3_scrubber_preserves_parameter_in_normal_prose() -> None:
    raw = "The volume parameter in audio engineering is unrelated to Spotify tools."
    cleaned = scrub_internal_tool_references(raw)
    assert "parameter" in cleaned.lower()


def test_r6_item3_scrubber_still_removes_set_parameter_doc_phrasing() -> None:
    raw = "use the Spotify tool and set the `public` parameter to `False`"
    cleaned = scrub_internal_tool_references(raw)
    assert "set the" not in cleaned.lower() or "public" not in cleaned.lower()
    assert "`" not in cleaned


@respx.mock
def test_r6_item4_unknown_device_helper_covers_fallback(data_dir, signed_in_tokens) -> None:
    settings = Settings()
    data, play_requests = run_unknown_device_playback_fallback(
        saved_device_id="helper_device_xyz",
        unknown_device_id="bogus_device",
        settings=settings,
    )
    assert data.get("ok") is True
    assert "bogus_device" in (data.get("device_fallback_note") or "")
    assert play_requests
    assert "helper_device_xyz" in str(play_requests[-1].url)
