"""Round-15 — PR #9: surprise block, podcasts, library contains, removed routes, chat bank."""

from __future__ import annotations

import json

import httpx
import respx

from spot_backend.config import Settings
from spot_backend.playlist_builder_store import clear_playlist_preview, load_playlist_preview
from spot_backend.playlist_pick import playlist_id_is_spotify_curated
from spot_backend.simulated_chat_bank import run_simulated_chat_bank
from spot_backend.spotify_client import SpotifyClient, SpotifyQuotaExceededError
from spot_backend.spotify_removed_routes import SpotifyRemovedRouteError, assert_spotify_route_allowed
from spot_backend.spotify_tools import SpotifyToolRunner
from spot_backend.chat_shortcuts import try_deterministic_surprise_me_reply


def test_r15_editorial_playlist_blocked_without_http_call(data_dir, signed_in_tokens) -> None:
    editorial = "37i9dQZF1Ept8KpxrQJ0R9"
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_play_playlist", {"playlist_id": editorial})
    runner.close()
    data = json.loads(raw)
    assert data.get("failure_reason") == "editorial_playlist_blocked"
    assert data.get("ok") is False


def test_r15_curated_prefix_is_37i9() -> None:
    assert playlist_id_is_spotify_curated("37i9dQZF1Ept8KpxrQJ0R9")
    assert not playlist_id_is_spotify_curated("6zCID88oNjNv9zx6puDHKj")


@respx.mock
def test_r15_user_public_playlists_never_calls_removed_route(
    data_dir, signed_in_tokens,
) -> None:
    route = respx.get(url__regex=r"https://api\.spotify\.com/v1/users/.*/playlists.*").mock(
        return_value=httpx.Response(200, json={"items": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run("spotify_user_public_playlists", {"user_id": "someone"})
    runner.close()
    assert not route.called
    data = json.loads(raw)
    assert data.get("endpoint_removed") is True


@respx.mock
def test_r15_follow_playlist_uses_me_library(data_dir, signed_in_tokens) -> None:
    pid = "plaaaaaaaaaaaaaaaaaaaa"
    respx.get(f"https://api.spotify.com/v1/tracks/{'1' * 22}").mock(
        return_value=httpx.Response(404, json={})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{pid}").mock(
        return_value=httpx.Response(200, json={"id": pid, "type": "playlist"})
    )
    lib = respx.put(url__regex=r"https://api\.spotify\.com/v1/me/library.*").mock(
        return_value=httpx.Response(200)
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/playlists/.*/followers.*").mock(
        return_value=httpx.Response(200)
    )
    runner = SpotifyToolRunner(settings=Settings())
    runner._session_known_ids.add(pid)
    raw = runner.run("spotify_follow_playlist", {"playlist_id": pid})
    runner.close()
    assert lib.called
    assert json.loads(raw).get("ok") is True


def test_r15_removed_route_guard_blocks_playlist_tracks_path() -> None:
    try:
        assert_spotify_route_allowed("GET", "/playlists/abc123/tracks")
        raise AssertionError("expected block")
    except SpotifyRemovedRouteError as e:
        assert e.failure_reason == "removed_route"


@respx.mock
def test_r15_library_contains(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[True])
    )
    runner = SpotifyToolRunner(settings=Settings())
    raw = runner.run(
        "spotify_library_contains",
        {"uris": ["spotify:track:1111111111111111111111"]},
    )
    runner.close()
    data = json.loads(raw)
    assert data.get("saved") == [True]


@respx.mock
def test_r15_get_show_and_episode(data_dir, signed_in_tokens) -> None:
    sid = "ssssssssssssssssssssss"
    eid = "eeeeeeeeeeeeeeeeeeeeee"
    respx.get(f"https://api.spotify.com/v1/shows/{sid}").mock(
        return_value=httpx.Response(200, json={"id": sid, "name": "Astro"})
    )
    respx.get(f"https://api.spotify.com/v1/episodes/{eid}").mock(
        return_value=httpx.Response(200, json={"id": eid, "name": "Ep 1"})
    )
    runner = SpotifyToolRunner(settings=Settings())
    show = json.loads(runner.run("spotify_get_show", {"show_id": sid}))
    ep = json.loads(runner.run("spotify_get_episode", {"episode_id": eid}))
    runner.close()
    assert show["name"] == "Astro"
    assert ep["name"] == "Ep 1"


@respx.mock
def test_r15_playlist_builder_preview_and_commit(data_dir, signed_in_tokens) -> None:
    clear_playlist_preview("pb-test")
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "tracks": {
                    "items": [
                        {
                            "id": "1111111111111111111111",
                            "uri": "spotify:track:1111111111111111111111",
                            "name": "One",
                            "artists": [{"name": "A"}],
                        }
                    ]
                }
            },
        )
    )
    respx.post("https://api.spotify.com/v1/me/playlists").mock(
        return_value=httpx.Response(
            200,
            json={"id": "newpl0000000000000001", "name": "spot-ai-fy test", "public": False},
        )
    )
    respx.post("https://api.spotify.com/v1/playlists/newpl0000000000000001/items").mock(
        return_value=httpx.Response(200, json={})
    )
    runner = SpotifyToolRunner(settings=Settings(), conversation_id="pb-test")
    preview_raw = runner.run(
        "spotify_playlist_builder_preview",
        {"name": "spot-ai-fy test", "track_queries": ["one song"]},
    )
    preview = json.loads(preview_raw)
    assert preview.get("awaiting_approval") is True
    assert load_playlist_preview("pb-test")
    commit_raw = runner.run(
        "spotify_playlist_builder_commit",
        {"approve": True, "name": "spot-ai-fy test"},
    )
    commit = json.loads(commit_raw)
    runner.close()
    assert commit.get("ok") is True
    assert commit.get("public") is False


@respx.mock
def test_r15_quota_exceeded_no_retry(data_dir, signed_in_tokens) -> None:
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"status": 429, "message": "quota", "reason": "QUOTA_EXCEEDED"}},
        )
    )
    client = SpotifyClient(settings=Settings())
    try:
        try:
            client.api_get("/search", params={"q": "x", "type": "track", "limit": 1})
            raise AssertionError("expected quota error")
        except SpotifyQuotaExceededError as e:
            assert "usage limit" in str(e).lower()
    finally:
        client.close()


@respx.mock
def test_r15_surprise_me_blocks_editorial_playlist(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    owned = "ownedpl000000000000099"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(
            200,
            json={
                "items": [
                    {
                        "id": owned,
                        "name": "Mine",
                        "owner": {"id": me_id},
                        "tracks": {"total": 1},
                    }
                ]
            },
        )
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/playlists/owned.*/items.*").mock(
        return_value=httpx.Response(200, json={"total": 1, "items": [{"track": {"id": "t1"}}]})
    )
    respx.get(f"https://api.spotify.com/v1/playlists/{owned}").mock(
        return_value=httpx.Response(200, json={"id": owned, "owner": {"id": me_id}})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/play.*").mock(
        return_value=httpx.Response(204)
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(
            200,
            json={"is_playing": True, "context": {"uri": f"spotify:playlist:{owned}"}},
        )
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": []})
    )
    runner = SpotifyToolRunner(settings=Settings())
    outcome = try_deterministic_surprise_me_reply("Surprise me", runner)
    runner.close()
    assert outcome is not None
    assert "37i9" not in outcome.reply
    assert "spotify_play_playlist" in outcome.tool_names()


@respx.mock
def test_r15_playback_tools_shuffle_repeat_seek_transfer(data_dir, signed_in_tokens) -> None:
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(200, text="ok")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/repeat.*").mock(
        return_value=httpx.Response(200, text="ok")
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/seek.*").mock(
        return_value=httpx.Response(200, text="ok")
    )
    respx.get("https://api.spotify.com/v1/me/player/devices").mock(
        return_value=httpx.Response(200, json={"devices": [{"id": "d1", "name": "Phone"}]})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player.*").mock(
        return_value=httpx.Response(204)
    )
    runner = SpotifyToolRunner(settings=Settings())
    assert json.loads(runner.run("spotify_set_shuffle", {"state": True})).get("ok")
    assert json.loads(runner.run("spotify_set_repeat", {"state": "context"})).get("ok")
    assert json.loads(runner.run("spotify_seek", {"position_ms": 30000})).get("ok")
    assert json.loads(runner.run("spotify_transfer_playback", {"device_id": "d1"})).get("ok")
    runner.close()


@respx.mock
def test_r15_simulated_chat_bank_runs(data_dir, signed_in_tokens) -> None:
    me_id = "user123456789012345678901"
    respx.get("https://api.spotify.com/v1/me").mock(
        return_value=httpx.Response(200, json={"id": me_id})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/playlists.*").mock(
        return_value=httpx.Response(200, json={"items": [], "total": 0})
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/me/library/contains.*").mock(
        return_value=httpx.Response(200, json=[False])
    )
    respx.get(url__regex=r"https://api\.spotify\.com/v1/search.*").mock(
        return_value=httpx.Response(200, json={"shows": {"items": []}})
    )
    respx.get("https://api.spotify.com/v1/me/player").mock(
        return_value=httpx.Response(200, json={})
    )
    respx.put(url__regex=r"https://api\.spotify\.com/v1/me/player/shuffle.*").mock(
        return_value=httpx.Response(200, text="ok")
    )
    runner = SpotifyToolRunner(settings=Settings())
    rows = run_simulated_chat_bank(runner)
    runner.close()
    assert rows
    assert all(isinstance(r.prompt, str) for r in rows)
