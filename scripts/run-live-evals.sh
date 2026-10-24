#!/usr/bin/env bash
# Run the full live agent eval suite (metadata.live cases + optional LLM judges).
# Mirrors .github/workflows/agent-eval-live.yml — use on a machine with Spotify + LLM secrets.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
DATA_DIR="${DATA_DIR:-/tmp/spot-live-eval}"
REPORT_DIR="${REPORT_DIR:-/tmp/spot-live-eval-reports}"
if [[ -n "${EVAL_JUDGE_MODEL:-}" ]]; then
  JUDGE="$EVAL_JUDGE_MODEL"
elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
  JUDGE="google:gemini-2.5-flash"
else
  JUDGE="ollama:qwen2.5:7b"
fi
PROVIDER="${LLM_EVAL_PROVIDER:-gemini}"

has_llm=false
if [[ -n "${GEMINI_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" || -n "${ANTHROPIC_API_KEY:-}" || -n "${XAI_API_KEY:-}" ]]; then
  has_llm=true
fi
if [[ -n "${OLLAMA_HOST:-}" ]]; then
  has_llm=true
fi

has_spotify=false
if [[ -n "${SPOTIFY_CLIENT_ID:-}" && -n "${SPOTIFY_REFRESH_TOKEN:-}" ]]; then
  has_spotify=true
fi

if [[ "$has_llm" != true || "$has_spotify" != true ]]; then
  echo "Missing prerequisites for live evals." >&2
  echo "  LLM: set GEMINI_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, XAI_API_KEY, or OLLAMA_HOST" >&2
  echo "  Spotify: set SPOTIFY_CLIENT_ID and SPOTIFY_REFRESH_TOKEN" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$REPORT_DIR"
export DATA_DIR

if [[ ! -f "$DATA_DIR/tokens.json" ]]; then
  python3 <<'PY'
import json, os, time
from pathlib import Path
data_dir = Path(os.environ["DATA_DIR"])
p = data_dir / "tokens.json"
p.write_text(
    json.dumps(
        {
            "access_token": "live-eval-placeholder",
            "refresh_token": os.environ["SPOTIFY_REFRESH_TOKEN"],
            "expires_at": time.time() + 3600,
            "scope": (
                "user-read-playback-state user-read-email playlist-read-private user-library-read"
            ),
        }
    ),
    encoding="utf-8",
)
print(f"Wrote {p}")
PY
fi

export LLM_PROVIDER="${LLM_PROVIDER:-gemini}"

cd "$BACKEND"
python3 -m evals.run \
  --mode live \
  --provider "$PROVIDER" \
  --repeat 1 \
  --judge "$JUDGE" \
  --regression-report \
  --report-json "$REPORT_DIR/agent-eval-live-report.json" \
  --report-md "$REPORT_DIR/agent-eval-live-report.md" \
  --save-live-baseline

echo "Reports: $REPORT_DIR/agent-eval-live-report.{json,md}"
