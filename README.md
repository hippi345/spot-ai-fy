# Spot-AI-fy

**Ask Spotify in plain language — search, playlists, playback.**

Spot-AI-fy is a local-first natural-language front end for the Spotify Web API. You type things like *"add a SZA song from 2024 to RNB2025 and play the playlist starting at that track with repeat on"* and an LLM translates that into a sequence of Spotify API calls — search, dedupe against the playlist, add, verify, play, set repeat — returning one short summary.

![Spot-AI-fy UI](docs/spot-ai-fy-screenshot.png)

---

## What it is

- **A Python backend** (FastAPI) that handles Spotify PKCE OAuth, wraps the Spotify Web API as a set of strongly-typed tools, and routes chat turns through either [Ollama](https://ollama.com/) (local) or [Google Gemini](https://ai.google.dev/) (cloud).
- **A React/Vite frontend** — minimal UI with a chat panel, LLM provider switcher, Spotify sign-in, and device picker.
- **An MCP server** (`run_mcp.py`) exposing the same Spotify tools over the [Model Context Protocol](https://modelcontextprotocol.io) so any MCP-aware client (Claude Desktop, Cursor, etc.) can drive your Spotify account directly.
- **Tokens live on your machine only** — refresh tokens go to `%USERPROFILE%\.spot_ai_fy\tokens.json` (outside the repo). Nothing runs in the cloud except the LLM call itself (and only if you choose Gemini; with Ollama the whole stack is local).

## Features

- **Natural-language Spotify control** across search, library, playlists, and playback.
- **Composite tools** that do multi-step workflows in one LLM call:
  - `spotify_add_tracks_by_query` — search + year filter + dedupe against the target playlist + add, all guaranteeing real Spotify track IDs (no fabricated / ghost rows).
  - `spotify_play_playlist` — start a playlist at a specific track **and** apply repeat/shuffle in one call, with post-call verification that the device actually switched.
- **Play-now vs. play-next clarity** — explicit separate tools (`spotify_start_resume_playback` / `spotify_play_playlist` for immediate interruption, `spotify_add_to_queue` / `spotify_play_next` for queueing).
- **Resilient playback** — verifies Spotify actually switched to the requested track; force-skips past queue reorderings when needed; falls back to pause-then-replay when a Spotify Connect session refuses to switch context; recovers from Spotify's transient `5xx` edge errors by polling `/me/player`.
- **Resilient LLM calls** — exponential backoff + `Retry-After` handling for Gemini `429` and `503` responses; user-friendly surfacing of quota / high-demand errors instead of raw HTTP text.
- **Known-limitation guardrails** — the system prompt tells the agent which Spotify endpoints don't exist (per-playlist listen counts, per-track play counts, long listening history) so it answers plainly instead of looping through tools.
- **Good OAuth diagnostics** — distinguishes stale scopes (requires re-consent, since Spotify refresh tokens don't upgrade scopes), not-owned playlists, and the Spotify Web API [Feb 2026 dev-mode migration](https://developer.spotify.com/blog/2026-02-06-update-on-developer-access-and-platform-security) (`/tracks` → `/items`, removed `/artists/{id}/top-tracks`, capped `/search` limit).
- **Two LLM backends, swappable at runtime from the UI** — no `.env` edit needed to switch between local Ollama and Gemini. Model tags for both providers are populated dynamically from what the provider reports (`ollama list` for Ollama, the Google Generative Language models API for Gemini).
- **Live agent-progress panel (Ollama)** — shows rounds, tool calls, and elapsed time per step with a live ticker; toggle "Show details" to expand the raw tool-result previews. Gemini calls skip the panel to stay quiet.
- **CPU-friendly Ollama tuning knobs** — per-provider settings for context window, keep-alive, history replay, tool-result caps, and agent-step caps so a local model on a laptop stays responsive without silently truncating prompts. See [Bring your own LLM](#bring-your-own-llm).
- **Optional agent-context file** — drop a markdown file in `backend/AGENT_CONTEXT.md` (or point `AGENT_CONTEXT_FILE` at a path) and it's appended to the system prompt for both backends, letting you tune tone and rules without editing Python.

## Architecture

```
┌─────────────────┐      HTTP / SSE       ┌──────────────────────────┐      HTTPS       ┌─────────────────┐
│ React + Vite UI │ ───────────────────▶  │ FastAPI backend          │ ───────────────▶ │ Spotify Web API │
│  localhost:5173 │                       │  /login /callback        │                  └─────────────────┘
└─────────────────┘                       │  /chat /chat/stream      │
                                          │  /me /devices /playback  │                  ┌─────────────────┐
                                          │  /llm/provider           │  Ollama HTTP ──▶ │ Ollama (local)  │
                                          │  SpotifyToolRunner       │  or Gemini API   │ or Gemini cloud │
                                          └────────────┬─────────────┘                  └─────────────────┘
                                                       │
                                                       │ stdio
                                                       ▼
                                          ┌──────────────────────────┐
                                          │ MCP server (optional)    │
                                          │  run_mcp.py              │
                                          └──────────────────────────┘
```

Refresh tokens, device choice, and LLM preferences persist in `%USERPROFILE%\.spot_ai_fy\` (Windows) or `~/.spot_ai_fy/` (macOS/Linux) — **never** inside the repo.

## Quickstart

### Prerequisites

- Python 3.12+ and Node 20+.
- A Spotify Developer app — create one at [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard). Add `http://127.0.0.1:8765/callback` as a Redirect URI. Copy the **Client ID** (you do **not** need a client secret for the default PKCE flow).
- Either [Ollama](https://ollama.com/download) running locally **or** a [Google AI Studio API key](https://aistudio.google.com/apikey) for Gemini.

### 1. Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
notepad .env   # paste your SPOTIFY_CLIENT_ID (and GEMINI_API_KEY if using Gemini)

uvicorn spot_backend.app:app --host 127.0.0.1 --port 8765 --reload
```

macOS/Linux equivalent:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
${EDITOR:-nano} .env

uvicorn spot_backend.app:app --host 127.0.0.1 --port 8765 --reload
```

### 2. Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open [http://localhost:5173](http://localhost:5173), click **Connect Spotify**, pick a playback device, and start chatting.

### 3. (Optional) MCP server

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python run_mcp.py
```

Point any MCP client at this stdio server to use the same Spotify tools from inside Claude Desktop, Cursor, etc.

## Bring your own LLM

Spot-AI-fy routes every chat turn through a pluggable LLM provider. Two are supported out of the box; picking one is a two-step process (drop creds/endpoint in `.env`, optionally pick a specific model in the UI).

### Ollama (local, default)

Best for privacy, offline use, and "I already have a GPU / spare laptop running Ollama".

1. Install Ollama from [ollama.com/download](https://ollama.com/download) and pull a model that supports tool calling:

   ```powershell
   # Good defaults on CPU-only machines (8 GB+ RAM)
   ollama pull qwen2.5:3b-instruct        # recommended: native tool calls, small, fast
   ollama pull llama3.2:3b-instruct       # similar tier, Meta format

   # Higher quality if you have a GPU or plenty of CPU headroom
   ollama pull qwen2.5:7b-instruct
   ollama pull llama3.1:8b-instruct
   ```

2. Set these in `backend/.env`:

   ```ini
   LLM_PROVIDER=ollama
   OLLAMA_HOST=http://127.0.0.1:11434
   OLLAMA_MODEL=qwen2.5:3b-instruct
   ```

3. Restart the backend, then either pick the model from the **Model** dropdown in the UI (it's populated from `ollama list`) or click **Reset to .env** to use the `.env` default.

**Tuning for CPU-only machines.** Ollama's default context is 4096 tokens, which routinely gets silently truncated by Spot-AI-fy's system prompt + history + tool results. The following knobs (all Ollama-only — they do not affect Gemini) are safe defaults on a 16 GB CPU laptop:

```ini
OLLAMA_NUM_CTX=8192            # stop silent "truncating input prompt" warnings
OLLAMA_KEEP_ALIVE=30m          # skip the cold-load penalty between prompts (can be 1–2 min on CPU)
OLLAMA_HISTORY_MESSAGES=10     # only replay the last N UI messages each round
OLLAMA_TOOL_RESULT_MAX=5000    # cap per-tool result bytes fed back into the prompt
OLLAMA_MAX_STEPS=8             # bail out of runaway tool loops faster than AGENT_MAX_STEPS
```

The first line of the progress panel ("Connecting to Ollama (…)") echoes whichever of these are in effect, so you can confirm at a glance. Watch `%LOCALAPPDATA%\Ollama\server.log` on Windows (or `~/.ollama/logs/` on macOS/Linux) for `truncating input prompt` warnings — if you still see them, raise `OLLAMA_NUM_CTX`.

### Gemini (cloud)

Best when you want fast answers and don't mind sending each chat turn (plus relevant Spotify tool-result JSON) to Google per their [Gemini API terms](https://ai.google.dev/gemini-api/terms).

1. Grab a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
2. Set in `backend/.env`:

   ```ini
   LLM_PROVIDER=gemini
   GEMINI_API_KEY=AIza...
   GEMINI_MODEL=gemini-2.5-flash
   ```

3. Restart the backend. Like the Ollama path, the UI dropdown is populated from the provider's own list-models endpoint (filtered to ones that support `generateContent`), so you can try `gemini-2.5-pro`, `gemini-2.0-flash`, etc. at runtime without editing `.env`.

Spot-AI-fy retries `429` / `503` Gemini responses with exponential backoff (1.5 s → 3 s → 6 s → 12 s, honoring `Retry-After`) and surfaces quota / high-demand errors in plain English rather than raw HTTP.

### What a new LLM backend would need to support

The agent loop depends on tool calling — it needs a model that will either emit native tool/function-call messages (Ollama's `tools`, OpenAI's `tools`, Anthropic's `tools`) or reliably produce well-formed JSON blocks when asked. For Ollama, Spot-AI-fy has a fallback that coaxes tool calls out of non-native models via fenced JSON, but quality drops quickly on small non-instruction-tuned tags. If you substitute a provider, pick a model that's tool-use-capable.

> **Heads up** — OpenAI (GPT-*), Anthropic (Claude), and OpenAI-compatible proxies like OpenRouter / Groq / Together are not wired up yet. See [Roadmap](#roadmap) below.

## Environment variables

All variables live in `backend/.env` (see [`backend/.env.example`](backend/.env.example) for the annotated template). Only `SPOTIFY_CLIENT_ID` is strictly required.

| Variable | Purpose |
| --- | --- |
| `SPOTIFY_CLIENT_ID` | Required. Public client id from the Spotify developer dashboard (safe to keep in `.env`). |
| `SPOTIFY_CLIENT_SECRET` | Optional — only used if you want classic confidential OAuth instead of PKCE. Leave empty otherwise. |
| `SPOTIFY_REDIRECT_URI` | Defaults to `http://127.0.0.1:8765/callback`. Must match the one registered on the Spotify dashboard. |
| `API_HOST` / `API_PORT` | Where the FastAPI backend binds (defaults `127.0.0.1:8765`). |
| `FRONTEND_ORIGIN` | CORS origin for the Vite dev server (default `http://localhost:5173`). |
| `LLM_PROVIDER` | `ollama` (default, local) or `gemini` (cloud). Runtime overridable from the UI. |
| `OLLAMA_HOST` / `OLLAMA_MODEL` | Ollama endpoint and default model tag. Model tag is overridable from the UI (dropdown is populated from `ollama list`). |
| `OLLAMA_NUM_CTX` | Ollama context window in tokens. Default `8192` to avoid silent truncation of long prompts. Set `0` to use the model's built-in default. |
| `OLLAMA_KEEP_ALIVE` | How long Ollama keeps the model resident after the last request (e.g. `30m`, `2h`, `-1` = forever). Avoids the ~100 s cold-load penalty on CPU. |
| `OLLAMA_HISTORY_MESSAGES` | Number of previous chat messages replayed to Ollama each round. `0` = send everything the UI passed (currently up to 40). Recommended `10` for CPU. |
| `OLLAMA_TOOL_RESULT_MAX` | Character cap on each tool result fed back into the Ollama prompt. `0` = use the built-in 12 000-char default. Recommended `5000` for CPU. |
| `OLLAMA_MAX_STEPS` | Ollama-specific agent step cap. `0` = use `AGENT_MAX_STEPS`. Recommended `6`–`8` for CPU so runaway tool loops bail out sooner. |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Required only when using Gemini. Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). Default model `gemini-2.5-flash`. Model is overridable from the UI (dropdown is populated from Google's list-models endpoint, filtered to `generateContent` support). |
| `AGENT_MAX_STEPS` | Overall cap on tool-call rounds per chat turn (default `16`). |
| `AGENT_CONTEXT_FILE` | Optional path to a markdown file appended to the system prompt for both LLMs. |
| `DATA_DIR` | Optional override for where tokens / device / LLM prefs are stored (defaults to `%USERPROFILE%\.spot_ai_fy`). |
| `SPOTIFY_SHOW_DIALOG` | Set to `false` to skip forcing the Spotify consent screen on every `/login` (default `true`). |

## Spotify tool surface

The backend exposes ~35 tools to the LLM (and via MCP). A few highlights:

- **Search / catalog**: `spotify_search`, `spotify_search_playlists` (find playlists by free-text description), `spotify_get_track`, `spotify_get_album`, `spotify_get_artist`, `spotify_artist_albums`, `spotify_artist_top_tracks`.
- **Library**: `spotify_me`, `spotify_user_playlists`, `spotify_user_saved_tracks`, `spotify_get_playlist`, `spotify_playlist_tracks`.
- **User stats**: `spotify_top_artists`, `spotify_top_tracks` (`time_range` = `short_term` ~last 4 weeks, `medium_term` ~last 6 months, `long_term` ~all-time; capped at 50). Requires the `user-top-read` scope.
- **Following**: `spotify_followed_artists` (artists you follow — Spotify's API does **not** expose followed users), `spotify_user_public_playlists` (any user's *public* playlists, by their `user_id`), `spotify_follow_playlist`, `spotify_unfollow_playlist`. Returning users may need to **Sign out → Connect** once to re-consent for the new `user-follow-read` scope.
- **Playlist edits (yours)**: `spotify_create_playlist`, `spotify_update_playlist`, `spotify_add_tracks_to_playlist`, `spotify_add_tracks_by_query` (composite), `spotify_remove_playlist_tracks`, `spotify_reorder_playlist_tracks`, `spotify_replace_playlist_tracks`.
- **Playlist edits (someone else's)**: `spotify_duplicate_playlist` — Spotify's API forbids editing other users' playlists, so this composite copies a source playlist into a brand-new one **owned by you** (paginated source read + new playlist + 100-uri batched copy). The returned `new_playlist_id` is fully writable for `spotify_add_tracks_to_playlist` / `spotify_remove_playlist_tracks` / etc.
- **Playback (play now)**: `spotify_start_resume_playback`, `spotify_play_playlist` (composite — start at track + repeat/shuffle), `spotify_pause`, `spotify_skip_next`, `spotify_skip_previous`, `spotify_seek`.
- **Playback (queue / next)**: `spotify_add_to_queue`, `spotify_play_next`.
- **Modes & devices**: `spotify_set_repeat`, `spotify_set_shuffle`, `spotify_set_volume`, `spotify_devices`, `spotify_transfer_playback`, `spotify_playback_state`.

All tools return structured JSON with explicit error flags (`stale_scopes_need_reauth`, `playlist_not_owned_by_user`, `playback_verified`, `rejected_uris`, `spotify_feb_2026_migration_possible`, ...) so the LLM stops guessing when something goes wrong.

### What Spotify's Web API does *not* expose

The agent's system prompt is wired to tell you plainly when something isn't possible **and** offer the closest available alternative. Known limits the agent will surface this way:

- **Your follower list** — only the *count* is available (via `spotify_me.followers.total`). Closest alternatives: `spotify_followed_artists`, `spotify_top_artists`.
- **Users you follow** — the Web API only exposes followed *artists*, not users (`spotify_followed_artists`).
- **Another user's private playlists** — only public playlists are visible (`spotify_user_public_playlists`).
- **Looking a user up by display name** — there's no endpoint for it; you must provide a Spotify `user_id` (the part after `spotify:user:` or `open.spotify.com/user/<id>`).
- **Editing another user's playlist** — not possible. The agent will offer `spotify_duplicate_playlist` to copy it into a writable playlist you own.
- **Per-playlist / per-track / per-album play counts** — not in the API. The agent will offer `spotify_top_artists` / `spotify_top_tracks` as the closest proxy.
- **Listening history beyond the most recent ~50 items** — not in the API.

## Security & privacy

- `backend/.env` is in `.gitignore` — keep your real `SPOTIFY_CLIENT_ID` and `GEMINI_API_KEY` there, not in commits.
- Spotify refresh tokens live in `%USERPROFILE%\.spot_ai_fy\tokens.json`, **outside the repo**.
- The PKCE flow never asks for a Spotify client secret, so the client id is a public identifier — safe to share.
- When `LLM_PROVIDER=ollama`, no user data leaves your machine.
- When `LLM_PROVIDER=gemini`, each chat turn (and relevant tool-result JSON) is sent to Google's Gemini API per their [terms](https://ai.google.dev/gemini-api/terms).

## Roadmap

Likely next additions, in descending priority:

- **OpenAI-compatible provider** (`LLM_PROVIDER=openai_compat` + `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `OPENAI_MODEL`). One driver, one tool-call shape, zero provider-specific code — unlocks OpenAI itself, Azure OpenAI, OpenRouter (which in turn fronts Anthropic Claude, Meta Llama 4, Mistral, Cohere, and most other hosted models), Groq, Together, DeepInfra, Fireworks, vLLM, LM Studio, and any self-hosted inference server that speaks the OpenAI `chat/completions` shape.
- **Native Anthropic provider** (`LLM_PROVIDER=anthropic`) for direct Claude access when users want Anthropic-specific features (prompt caching, extended thinking) beyond what OpenRouter exposes.
- **Token/cost accounting** surfaced in the progress panel for cloud providers.

If you'd like to contribute a new provider, the shape to match is the existing `backend/spot_backend/gemini_llm.py` (non-streaming final-text return) plus an entry in `iter_chat_events` in `backend/spot_backend/agent.py` so the UI can hit `/api/chat/stream`.

## Tech stack

- **Backend**: Python 3.12, FastAPI, Uvicorn, httpx, pydantic / pydantic-settings, [mcp](https://pypi.org/project/mcp/) for the MCP server.
- **Frontend**: React 19, Vite 6, TypeScript.
- **LLMs**: Pluggable. Ollama (local, default) or Gemini (`gemini-2.5-flash` by default) via Google Generative Language API. See [Bring your own LLM](#bring-your-own-llm) for model and tuning guidance; see [Roadmap](#roadmap) for planned provider support.
- **Spotify**: Web API, PKCE OAuth, scopes include `playlist-modify-public`/`-private`, `playlist-read-private`/`-collaborative`, `user-read-playback-state`, `user-modify-playback-state`, `user-library-read`, `user-top-read`, `user-follow-read`, `user-read-private`.

## License

TBD — add a `LICENSE` file before publishing broadly. Until then, the repo is "all rights reserved" by default.
