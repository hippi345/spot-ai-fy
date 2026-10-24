# Live eval summary (Gemini)

- **When:** 2026-10-24 19:27 UTC
- **Provider:** `gemini` (agent) + `google:gemini-2.5-flash` (judges)
- **Spotify:** dummy client id + refresh token; HTTP mocked per case via `respx` (no real Spotify API)
- **Cases:** 6 (`metadata.live`)

## Rubric means (0–1)
- **accurate_summary:** 0.500
- **concise_friendly:** 1.000
- **no_false_claims:** 0.833

## Judge errors
- **Count:** 0

## Failed cases (assertion or judge)
- `library_index_on_semantic_hit`: assertions=['accurate_summary_pass'], evaluator_failures=0
- `library_index_off_uses_catalog_search`: assertions=['accurate_summary_pass', 'no_false_claims_pass'], evaluator_failures=0
- `mood_sad_acoustic`: assertions=['accurate_summary_pass'], evaluator_failures=0

## Library index on vs off (live)
### `library_index_on_semantic_hit` (SPOT_LIBRARY_INDEX=on)
- **First tools:** search_my_library, spotify_search, spotify_user_saved_tracks, spotify_search, spotify_search_playlists
- **accurate_summary_score:** 0.0
- **no_false_claims_score:** 1.0
- **Assertion fails:** accurate_summary_pass

### `library_index_off_uses_catalog_search` (SPOT_LIBRARY_INDEX=off)
- **First tools:** spotify_search, spotify_search_playlists, spotify_search, spotify_user_playlists, spotify_search_playlists
- **accurate_summary_score:** 0.0
- **no_false_claims_score:** 0.0
- **Assertion fails:** accurate_summary_pass, no_false_claims_pass

## Notes
- Regression gate exits non-zero because mock baseline expects 13 CI cases; live suite intentionally runs 6 live-tagged cases.
- Fixes on this branch: LLM HTTP passthrough under respx, sync LLM judges, library sync mocks, pre-build library index for live arms, Gemini judge deps.
