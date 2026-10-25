"""chat_tool_events — play tool args on tool_done (live eval traces)."""

from spot_backend.chat_tool_events import play_tool_arguments_for_event_trace


def test_play_tool_trace_includes_title_artist_uri_only() -> None:
    args = {
        "track_name": "Down",
        "artist_name": "Jay Sean",
        "uri": "spotify:track:abc",
        "access_token": "must-not-appear",
        "refresh_token": "must-not-appear",
    }
    out = play_tool_arguments_for_event_trace("spotify_play_track", args)
    assert out == {
        "track_name": "Down",
        "artist_name": "Jay Sean",
        "uri": "spotify:track:abc",
    }
    blob = str(out)
    assert "must-not-appear" not in blob
    assert "AIzaSy" not in blob


def test_play_tool_trace_empty_when_no_play_identity_keys() -> None:
    assert play_tool_arguments_for_event_trace("spotify_search", {"query": "x"}) == {}
