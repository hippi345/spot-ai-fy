"""Spot-AI-fy: MCP stdio server exposing Spotify tools (same token/device files as the web API)."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from spot_backend.spotify_tools import SpotifyToolRunner

mcp = FastMCP("Spotify")
_runner = SpotifyToolRunner()


@mcp.tool()
def spotify_search(
    query: str, types: str = "track,artist,album", market: str = "", limit: int = 10, offset: int = 0
) -> str:
    """Search Spotify for tracks, artists, or albums."""
    args: dict = {"query": query, "types": types, "limit": limit, "offset": offset}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_search", args)


@mcp.tool()
def spotify_me() -> str:
    """Get the current Spotify user profile."""
    return _runner.run("spotify_me", {})


@mcp.tool()
def spotify_user_playlists(limit: int = 10, offset: int = 0) -> str:
    """List the current user's playlists."""
    return _runner.run("spotify_user_playlists", {"limit": limit, "offset": offset})


@mcp.tool()
def spotify_get_playlist(playlist_id: str, market: str = "") -> str:
    """Get playlist metadata and a page of tracks."""
    args: dict = {"playlist_id": playlist_id}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_get_playlist", args)


@mcp.tool()
def spotify_playlist_tracks(playlist_id: str, limit: int = 10, offset: int = 0) -> str:
    """List tracks in a playlist."""
    return _runner.run(
        "spotify_playlist_tracks",
        {"playlist_id": playlist_id, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_user_saved_tracks(limit: int = 10, offset: int = 0) -> str:
    """List the user's saved (liked) tracks."""
    return _runner.run("spotify_user_saved_tracks", {"limit": limit, "offset": offset})


@mcp.tool()
def spotify_top_artists(
    time_range: str = "medium_term", limit: int = 10, offset: int = 0
) -> str:
    """Return the signed-in user's top artists for a time window.

    time_range is one of: short_term (~last 4 weeks), medium_term (~last 6 months,
    default), long_term (~all-time). Requires Spotify scope user-top-read.
    """
    return _runner.run(
        "spotify_top_artists",
        {"time_range": time_range, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_top_tracks(
    time_range: str = "medium_term", limit: int = 10, offset: int = 0
) -> str:
    """Return the signed-in user's top tracks for a time window.

    time_range is one of: short_term (~last 4 weeks), medium_term (~last 6 months,
    default), long_term (~all-time). Requires Spotify scope user-top-read.
    """
    return _runner.run(
        "spotify_top_tracks",
        {"time_range": time_range, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_followed_artists(limit: int = 10, after: str = "") -> str:
    """List artists the signed-in user follows (cursor-paginated).

    The Web API does not expose users the user follows, nor the user's own
    follower list — only artists. Requires scope user-follow-read.
    """
    args: dict[str, Any] = {"limit": limit}
    if after:
        args["after"] = after
    return _runner.run("spotify_followed_artists", args)


@mcp.tool()
def spotify_user_public_playlists(user_id: str, limit: int = 10, offset: int = 0) -> str:
    """List a Spotify user's PUBLIC playlists by their user_id (no scope required)."""
    return _runner.run(
        "spotify_user_public_playlists",
        {"user_id": user_id, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_search_playlists(
    query: str, limit: int = 5, offset: int = 0, market: str = ""
) -> str:
    """Search Spotify's catalog for public playlists matching a free-text description."""
    args: dict[str, Any] = {"query": query, "limit": limit, "offset": offset}
    if market:
        args["market"] = market
    return _runner.run("spotify_search_playlists", args)


@mcp.tool()
def spotify_follow_playlist(playlist_id: str, public: bool = True) -> str:
    """Follow (save to library) a playlist by id. Does NOT make you the owner."""
    return _runner.run(
        "spotify_follow_playlist",
        {"playlist_id": playlist_id, "public": public},
    )


@mcp.tool()
def spotify_duplicate_playlist(
    source_playlist_id: str,
    name: str = "",
    description: str = "",
    public: bool = False,
    max_tracks: int = 5000,
) -> str:
    """Copy any accessible playlist into a new playlist OWNED by the signed-in user.

    Returns the new playlist id, which is fully writable for subsequent
    spotify_add_tracks_to_playlist / spotify_remove_playlist_tracks calls.
    """
    args: dict[str, Any] = {
        "source_playlist_id": source_playlist_id,
        "public": public,
        "max_tracks": max_tracks,
    }
    if name:
        args["name"] = name
    if description:
        args["description"] = description
    return _runner.run("spotify_duplicate_playlist", args)


@mcp.tool()
def spotify_get_album(album_id: str, market: str = "US") -> str:
    """Get album metadata including release date and tracks."""
    return _runner.run("spotify_get_album", {"album_id": album_id, "market": market})


@mcp.tool()
def spotify_get_track(track_id: str, market: str = "US") -> str:
    """Get track metadata."""
    return _runner.run("spotify_get_track", {"track_id": track_id, "market": market})


@mcp.tool()
def spotify_artist_albums(
    artist_id: str, include_groups: str = "album", limit: int = 10, offset: int = 0
) -> str:
    """List albums and singles for an artist."""
    return _runner.run(
        "spotify_artist_albums",
        {
            "artist_id": artist_id,
            "include_groups": include_groups,
            "limit": limit,
            "offset": offset,
        },
    )


@mcp.tool()
def spotify_artist_latest_album(artist_id: str, include_groups: str = "album,single") -> str:
    """Newest album or single for an artist (by release_date)."""
    return _runner.run(
        "spotify_artist_latest_album",
        {"artist_id": artist_id, "include_groups": include_groups},
    )


@mcp.tool()
def spotify_play_artist_latest_release(
    artist_id: str = "",
    artist_name: str = "",
    kind: str = "single",
    market: str = "",
) -> str:
    """Find newest released single/album for an artist and start playback."""
    args: dict[str, str] = {}
    if artist_id.strip():
        args["artist_id"] = artist_id.strip()
    if artist_name.strip():
        args["artist_name"] = artist_name.strip()
    if kind.strip():
        args["kind"] = kind.strip()
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_play_artist_latest_release", args)


@mcp.tool()
def spotify_get_artist(artist_id: str, market: str = "") -> str:
    """Get artist profile (genres, popularity)."""
    args: dict = {"artist_id": artist_id}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_get_artist", args)


@mcp.tool()
def spotify_play_artist_popular_track(
    artist: str = "",
    artist_name: str = "",
    artist_id: str = "",
    market: str = "",
) -> str:
    """Play an artist's most popular track (search-based; dev-mode safe)."""
    args: dict[str, str] = {}
    if artist.strip():
        args["artist"] = artist.strip()
    if artist_name.strip():
        args["artist_name"] = artist_name.strip()
    if artist_id.strip():
        args["artist_id"] = artist_id.strip()
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_play_artist_popular_track", args)


@mcp.tool()
def spotify_create_playlist(name: str, public: bool = True, description: str = "") -> str:
    """Create a playlist for the current user."""
    return _runner.run(
        "spotify_create_playlist",
        {"name": name, "public": public, "description": description},
    )


@mcp.tool()
def spotify_add_tracks_to_playlist(playlist_id: str, track_uris: list[str]) -> str:
    """Add spotify track URIs to a playlist."""
    return _runner.run("spotify_add_tracks_to_playlist", {"playlist_id": playlist_id, "track_uris": track_uris})


@mcp.tool()
def spotify_update_playlist(
    playlist_id: str,
    name: str | None = None,
    description: str | None = None,
    public: bool | None = None,
    collaborative: bool | None = None,
) -> str:
    """Update playlist name, description, public, or collaborative (pass only fields to change)."""
    args: dict = {"playlist_id": playlist_id}
    if name is not None:
        args["name"] = name
    if description is not None:
        args["description"] = description
    if public is not None:
        args["public"] = public
    if collaborative is not None:
        args["collaborative"] = collaborative
    return _runner.run("spotify_update_playlist", args)


@mcp.tool()
def spotify_remove_playlist_tracks(
    playlist_id: str, track_uris: list[str], snapshot_id: str = ""
) -> str:
    """Remove tracks from a playlist (max 100 per call)."""
    args: dict = {"playlist_id": playlist_id, "track_uris": track_uris}
    if snapshot_id.strip():
        args["snapshot_id"] = snapshot_id.strip()
    return _runner.run("spotify_remove_playlist_tracks", args)


@mcp.tool()
def spotify_reorder_playlist_tracks(
    playlist_id: str,
    insert_before: int,
    range_start: int = 0,
    range_length: int = 1,
    snapshot_id: str = "",
) -> str:
    """Reorder tracks in a playlist."""
    args: dict = {
        "playlist_id": playlist_id,
        "insert_before": insert_before,
        "range_start": range_start,
        "range_length": range_length,
    }
    if snapshot_id.strip():
        args["snapshot_id"] = snapshot_id.strip()
    return _runner.run("spotify_reorder_playlist_tracks", args)


@mcp.tool()
def spotify_replace_playlist_tracks(playlist_id: str, track_uris: list[str]) -> str:
    """Replace all tracks in a playlist (max 100 URIs per call)."""
    return _runner.run("spotify_replace_playlist_tracks", {"playlist_id": playlist_id, "track_uris": track_uris})


@mcp.tool()
def spotify_unfollow_playlist(playlist_id: str) -> str:
    """Remove playlist from the user's library."""
    return _runner.run("spotify_unfollow_playlist", {"playlist_id": playlist_id})


@mcp.tool()
def spotify_devices() -> str:
    """List Spotify Connect devices."""
    return _runner.run("spotify_devices", {})


@mcp.tool()
def spotify_playback_state() -> str:
    """Get current playback state."""
    return _runner.run("spotify_playback_state", {})


@mcp.tool()
def spotify_transfer_playback(device_id: str) -> str:
    """Transfer playback to a device."""
    return _runner.run("spotify_transfer_playback", {"device_id": device_id})


@mcp.tool()
def spotify_start_resume_playback(
    device_id: str = "",
    uris: list[str] | None = None,
    context_uri: str = "",
) -> str:
    """Start or resume playback. Uses the UI-selected device when device_id is empty."""
    args: dict = {}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    if uris:
        args["uris"] = uris
    if context_uri.strip():
        args["context_uri"] = context_uri.strip()
    return _runner.run("spotify_start_resume_playback", args)


@mcp.tool()
def spotify_pause() -> str:
    """Pause playback."""
    return _runner.run("spotify_pause", {})


@mcp.tool()
def spotify_skip_next() -> str:
    """Skip to the next track."""
    return _runner.run("spotify_skip_next", {})


@mcp.tool()
def spotify_skip_previous() -> str:
    """Skip to the previous track."""
    return _runner.run("spotify_skip_previous", {})


@mcp.tool()
def spotify_add_to_queue(uri: str, device_id: str = "") -> str:
    """Add a track URI to the playback queue."""
    args: dict = {"uri": uri}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_add_to_queue", args)


@mcp.tool()
def spotify_play_next(uri: str, device_id: str = "") -> str:
    """Queue a track to play next (does not interrupt current playback)."""
    args: dict = {"uri": uri}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_play_next", args)


@mcp.tool()
def spotify_add_tracks_by_query(
    playlist_id: str,
    query: str,
    count: int = 1,
    min_year: int = 0,
) -> str:
    """Search for tracks matching a query and add them to a playlist."""
    args: dict[str, Any] = {"playlist_id": playlist_id, "query": query, "count": count}
    if min_year > 0:
        args["min_year"] = min_year
    return _runner.run("spotify_add_tracks_by_query", args)


@mcp.tool()
def spotify_play_artist(artist_name: str = "", artist_id: str = "", device_id: str = "") -> str:
    """Play an artist by starting their top tracks (uris list)."""
    args: dict[str, Any] = {}
    if artist_name.strip():
        args["artist_name"] = artist_name.strip()
    if artist_id.strip():
        args["artist_id"] = artist_id.strip()
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_play_artist", args)


@mcp.tool()
def spotify_play_track(
    track_name: str,
    artist_name: str = "",
    device_id: str = "",
) -> str:
    """Play a specific track by title and artist (PUT /me/player/play, not queue)."""
    args: dict[str, Any] = {"track_name": track_name.strip()}
    if artist_name.strip():
        args["artist_name"] = artist_name.strip()
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_play_track", args)


@mcp.tool()
def spotify_play_playlist(
    playlist_id: str,
    device_id: str = "",
    start_at_uri: str = "",
    repeat: str = "",
    shuffle: bool | None = None,
) -> str:
    """Start playing a playlist, optionally at a track, with repeat/shuffle."""
    args: dict[str, Any] = {"playlist_id": playlist_id}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    if start_at_uri.strip():
        args["start_at_uri"] = start_at_uri.strip()
    if repeat.strip():
        args["repeat"] = repeat.strip()
    if shuffle is not None:
        args["shuffle"] = shuffle
    return _runner.run("spotify_play_playlist", args)


@mcp.tool()
def spotify_set_repeat(state: str, device_id: str = "") -> str:
    """Set repeat mode: track, context, or off."""
    args: dict[str, Any] = {"state": state}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_set_repeat", args)


@mcp.tool()
def spotify_set_shuffle(state: bool, device_id: str = "") -> str:
    """Toggle shuffle on the active device."""
    args: dict[str, Any] = {"state": state}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_set_shuffle", args)


@mcp.tool()
def spotify_seek(position_ms: int, device_id: str = "") -> str:
    """Seek to position_ms in the current track."""
    args: dict[str, Any] = {"position_ms": position_ms}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_seek", args)


@mcp.tool()
def spotify_set_volume(volume_percent: int, device_id: str = "") -> str:
    """Set playback volume 0-100."""
    args: dict[str, Any] = {"volume_percent": volume_percent}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_set_volume", args)


@mcp.tool()
def spotify_recently_played(limit: int = 10, after: str = "", before: str = "") -> str:
    """Recently played tracks for the signed-in user."""
    args: dict[str, Any] = {"limit": limit}
    if after.strip():
        args["after"] = after.strip()
    if before.strip():
        args["before"] = before.strip()
    return _runner.run("spotify_recently_played", args)


@mcp.tool()
def spotify_save_tracks(track_id: str = "", track_ids: list[str] | None = None) -> str:
    """Save (like) tracks to the user's library."""
    args: dict[str, Any] = {}
    if track_id.strip():
        args["track_id"] = track_id.strip()
    if track_ids:
        args["track_ids"] = track_ids
    return _runner.run("spotify_save_tracks", args)


@mcp.tool()
def spotify_unsave_tracks(track_id: str = "", track_ids: list[str] | None = None) -> str:
    """Remove saved tracks from the user's library."""
    args: dict[str, Any] = {}
    if track_id.strip():
        args["track_id"] = track_id.strip()
    if track_ids:
        args["track_ids"] = track_ids
    return _runner.run("spotify_unsave_tracks", args)


@mcp.tool()
def spotify_save_albums(album_id: str = "", album_ids: list[str] | None = None) -> str:
    """Save albums to the user's library."""
    args: dict[str, Any] = {}
    if album_id.strip():
        args["album_id"] = album_id.strip()
    if album_ids:
        args["album_ids"] = album_ids
    return _runner.run("spotify_save_albums", args)


@mcp.tool()
def spotify_unsave_albums(album_id: str = "", album_ids: list[str] | None = None) -> str:
    """Remove saved albums from the user's library."""
    args: dict[str, Any] = {}
    if album_id.strip():
        args["album_id"] = album_id.strip()
    if album_ids:
        args["album_ids"] = album_ids
    return _runner.run("spotify_unsave_albums", args)


@mcp.tool()
def spotify_saved_albums(limit: int = 10, offset: int = 0, market: str = "") -> str:
    """List saved albums."""
    args: dict[str, Any] = {"limit": limit, "offset": offset}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_saved_albums", args)


@mcp.tool()
def spotify_follow_artist(artist_id: str = "", artist_ids: list[str] | None = None) -> str:
    """Follow artists."""
    args: dict[str, Any] = {}
    if artist_id.strip():
        args["artist_id"] = artist_id.strip()
    if artist_ids:
        args["artist_ids"] = artist_ids
    return _runner.run("spotify_follow_artist", args)


@mcp.tool()
def spotify_unfollow_artist(artist_id: str = "", artist_ids: list[str] | None = None) -> str:
    """Unfollow artists."""
    args: dict[str, Any] = {}
    if artist_id.strip():
        args["artist_id"] = artist_id.strip()
    if artist_ids:
        args["artist_ids"] = artist_ids
    return _runner.run("spotify_unfollow_artist", args)


@mcp.tool()
def spotify_get_queue() -> str:
    """Return the current playback queue."""
    return _runner.run("spotify_get_queue", {})


@mcp.tool()
def spotify_remove_from_queue() -> str:
    """Explain that Spotify cannot remove queue items (offer skip instead)."""
    return _runner.run("spotify_remove_from_queue", {})


@mcp.tool()
def spotify_playlists_containing_track(
    track_id: str,
    max_playlists: int = 30,
    max_pages_per_playlist: int = 3,
) -> str:
    """Find playlists in the user's library that contain a track."""
    return _runner.run(
        "spotify_playlists_containing_track",
        {
            "track_id": track_id,
            "max_playlists": max_playlists,
            "max_pages_per_playlist": max_pages_per_playlist,
        },
    )


@mcp.tool()
def spotify_library_contains(uris: list[str] | None = None, uri: str = "") -> str:
    """Check whether URIs are saved in the user's library."""
    args: dict[str, Any] = {}
    if uri.strip():
        args["uri"] = uri.strip()
    if uris:
        args["uris"] = uris
    return _runner.run("spotify_library_contains", args)


@mcp.tool()
def spotify_library_save(uris: list[str]) -> str:
    """Save Spotify URIs to the user's library."""
    return _runner.run("spotify_library_save", {"uris": uris})


@mcp.tool()
def spotify_library_remove(uris: list[str]) -> str:
    """Remove Spotify URIs from the user's library."""
    return _runner.run("spotify_library_remove", {"uris": uris})


@mcp.tool()
def spotify_get_show(show_id: str) -> str:
    return _runner.run("spotify_get_show", {"show_id": show_id})


@mcp.tool()
def spotify_get_show_episodes(show_id: str, limit: int = 10, offset: int = 0) -> str:
    return _runner.run(
        "spotify_get_show_episodes",
        {"show_id": show_id, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_get_episode(episode_id: str) -> str:
    return _runner.run("spotify_get_episode", {"episode_id": episode_id})


@mcp.tool()
def spotify_get_audiobook(audiobook_id: str) -> str:
    return _runner.run("spotify_get_audiobook", {"audiobook_id": audiobook_id})


@mcp.tool()
def spotify_get_audiobook_chapters(audiobook_id: str, limit: int = 10, offset: int = 0) -> str:
    return _runner.run(
        "spotify_get_audiobook_chapters",
        {"audiobook_id": audiobook_id, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_get_chapter(chapter_id: str) -> str:
    return _runner.run("spotify_get_chapter", {"chapter_id": chapter_id})


@mcp.tool()
def spotify_user_saved_shows(limit: int = 10, offset: int = 0) -> str:
    return _runner.run("spotify_user_saved_shows", {"limit": limit, "offset": offset})


@mcp.tool()
def spotify_user_saved_episodes(limit: int = 10, offset: int = 0) -> str:
    return _runner.run("spotify_user_saved_episodes", {"limit": limit, "offset": offset})


@mcp.tool()
def spotify_user_saved_audiobooks(limit: int = 10, offset: int = 0) -> str:
    return _runner.run("spotify_user_saved_audiobooks", {"limit": limit, "offset": offset})


@mcp.tool()
def spotify_playlist_builder_preview(name: str, track_queries: list[str]) -> str:
    return _runner.run(
        "spotify_playlist_builder_preview",
        {"name": name, "track_queries": track_queries},
    )


@mcp.tool()
def spotify_playlist_builder_commit(approve: bool = True, name: str = "") -> str:
    args: dict[str, Any] = {"approve": approve}
    if name.strip():
        args["name"] = name.strip()
    return _runner.run("spotify_playlist_builder_commit", args)


@mcp.tool()
def spotify_play_show_latest_episode(show_id: str, device_id: str = "") -> str:
    args: dict[str, Any] = {"show_id": show_id}
    if device_id.strip():
        args["device_id"] = device_id.strip()
    return _runner.run("spotify_play_show_latest_episode", args)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
