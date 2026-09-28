# Spotify playlist privacy (PR #9 research notes)

## Observed behavior (Web API, dev-mode apps, 2025–2026)

- **Create**: `POST /me/playlists` with `"public": false` in the JSON body is the supported way to create a user-owned playlist. The legacy `POST /users/{id}/playlists` route is discouraged but behaves similarly for owned playlists.
- **Update**: `PUT /playlists/{id}` with `"public": false` (and usually `"collaborative": false`) is required when create-time privacy is ignored or flipped.
- **Read-back**: `GET /playlists/{id}?fields=id,public` is the source of truth for whether the API considers the playlist public. The app **must not** tell the user a playlist is private unless this GET returns `public: false`.
- **Eventual consistency**: Some tokens/apps report `public: true` immediately after create even when the create body set `public: false`. A PUT plus short backoff polling (≈0.35s, 0.75s, 1.5s) often converges; when it does not, the playlist remains public in API read-back.
- **Collaborative flag**: Collaborative playlists have different sharing semantics; we force `collaborative: false` on the privacy PUT to avoid ambiguous visibility.
- **User fallback**: When read-back stays `public: true`, instruct the user to open the playlist in the Spotify app → ⋯ → **Make private**.

## Implementation in spot-ai-fy

`spotify_playlist_builder_commit` creates with `public: false`, adds tracks, then calls `_ensure_playlist_private` (PUT + polled GET). The tool JSON includes `verified_private` and, when needed, `privacy_warning` / `user_message` with app instructions.
