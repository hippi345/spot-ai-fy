#!/usr/bin/env bash
# Run the full live agent eval suite (metadata.live cases + optional LLM judges).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
DATA_DIR="${DATA_DIR:-/tmp/spot-live-eval}"
REPORT_DIR="${REPORT_DIR:-/tmp/spot-live-eval-reports}"
JUDGE="${EVAL_JUDGE_MODEL:-ollama:qwen2.5:7b}"
PROVIDER="${LLM_EVAL_PROVIDER:-gemini}"

if [[ -z "${GEMINI_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" && -z "${XAI_API_KEY:-}" && -z "${OLLAMA_HOST:-}" ]]; then
  echo "Missing LLM credential: set GEMINI_API_KEY (CI) or another provider." >&2
  exit 1
fi

export SPOTIFY_CLIENT_ID="${SPOTIFY_CLIENT_ID:-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}"
mkdir -p "$DATA_DIR" "$REPORT_DIR"
export DATA_DIR
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
