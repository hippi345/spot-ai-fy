# Ollama smoke test (opt-in)

This document describes the **automated** counterpart to the manual steps in [SMOKE_TEST.md](SMOKE_TEST.md) (prompts 1–7). It exercises the same `/api/chat/stream` agent loop as production, but with Spotify HTTP mocked (see `backend/tests/smoke_spotify_state.py`) and a **real** local Ollama model performing tool calls.

Deterministic chat shortcuts are disabled (`SPOT_AI_FY_DISABLE_DETERMINISTIC_CHAT=1`) so every prompt goes through the LLM tool loop—not the shortcut path used by the default mocked Gemini smoke test.

## Prerequisites

- Python dev dependencies: `pip install -r backend/requirements-dev.txt`
- [Ollama](https://ollama.com/download) installed and running (`ollama serve` or the desktop app)
- A **tool-calling** model pulled locally

### Recommended CPU models

| Model tag | Approx size | RAM hint | Notes |
|-----------|-------------|----------|--------|
| `qwen2.5:3b` or `qwen2.5:3b-instruct` | ~2 GB | 8 GB+ system RAM | Best first try; native tool calls |
| `llama3.2:3b` | ~2 GB | 8 GB+ | Fallback if Qwen tag unavailable |
| `qwen2.5:7b-instruct` | ~4.5 GB | 16 GB+ | Slower on CPU but fewer tool mistakes |

On CPU-only hosts, expect **30–120 seconds per prompt** for 3B models with the default agent prompt and tool list. The first prompt after a cold start is often much slower until the model is loaded (`OLLAMA_KEEP_ALIVE=30m` helps).

Point the backend at Ollama with:

```ini
LLM_PROVIDER=ollama
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5:3b
```

The smoke test sets these via pytest `monkeypatch`; you do not need a configured `backend/.env` for the test itself.

## Run locally

```bash
# Terminal 1 — Ollama
ollama serve
ollama pull qwen2.5:3b   # or qwen2.5:3b-instruct / llama3.2:3b

# Terminal 2 — smoke test (from repo root)
export RUN_OLLAMA_SMOKE=1
export OLLAMA_MODEL=qwen2.5:3b
export SPOT_AI_FY_DISABLE_DETERMINISTIC_CHAT=1
export OLLAMA_SMOKE_REPORT_PATH=/tmp/ollama-smoke-report.json   # optional

pytest backend/tests/test_ollama_smoke.py -m ollama_smoke -v -s
```

Without `RUN_OLLAMA_SMOKE=1`, the test is **skipped** so normal `pytest backend/tests` and PR CI stay unchanged.

Set `OLLAMA_SMOKE_STRICT=1` to fail pytest when any prompt misses expected tools, reply quality, or Spotify side effects (default is report-only).

## CI

Workflow [`.github/workflows/ollama-smoke.yml`](../.github/workflows/ollama-smoke.yml) runs on `workflow_dispatch` and nightly schedule. It is **not** a required PR check. Results are written to a job summary and uploaded as `ollama-smoke-report.json`.

## Known weak spots (small models)

Small local models often:

- Call `spotify_search` instead of `spotify_play_artist` for “Play …” (still acceptable if playback starts)
- Miss exact tool names for one-word commands (`Skip`, `Pause`, `Resume`)—tighter tool descriptions or a one-line system hint helps
- Hallucinate playlist names instead of calling `spotify_user_playlists`

The smoke report marks **expected vs actual** honestly; do not weaken assertions in the mocked Gemini smoke test to compensate.
