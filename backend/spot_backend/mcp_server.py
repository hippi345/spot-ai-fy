"""Spot-AI-fy: MCP stdio server exposing Spotify tools (same token/device files as the web API)."""

from __future__ import annotations

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
def spotify_user_playlists(limit: int = 20, offset: int = 0) -> str:
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
def spotify_playlist_tracks(playlist_id: str, limit: int = 50, offset: int = 0) -> str:
    """List tracks in a playlist."""
    return _runner.run(
        "spotify_playlist_tracks",
        {"playlist_id": playlist_id, "limit": limit, "offset": offset},
    )


@mcp.tool()
def spotify_user_saved_tracks(limit: int = 50, offset: int = 0) -> str:
    """List the user's saved (liked) tracks."""
    return _runner.run("spotify_user_saved_tracks", {"limit": limit, "offset": offset})


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
    artist_id: str, include_groups: str = "album,single", limit: int = 50, offset: int = 0
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
def spotify_get_artist(artist_id: str, market: str = "") -> str:
    """Get artist profile (genres, popularity)."""
    args: dict = {"artist_id": artist_id}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_get_artist", args)


@mcp.tool()
def spotify_artist_top_tracks(artist_id: str, market: str = "") -> str:
    """Get an artist's top tracks."""
    args: dict = {"artist_id": artist_id}
    if market.strip():
        args["market"] = market.strip()
    return _runner.run("spotify_artist_top_tracks", args)


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


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
