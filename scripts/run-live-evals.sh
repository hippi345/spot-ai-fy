#!/usr/bin/env bash
# Run the full live agent eval suite (metadata.live cases + LLM judges + thresholds.yaml).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND="$ROOT/backend"
DATA_DIR="${DATA_DIR:-/tmp/spot-live-eval}"
REPORT_DIR="${REPORT_DIR:-/tmp/spot-live-eval-reports}"
JUDGE_REPEAT="${EVAL_JUDGE_REPEAT:-2}"
if [[ -n "${EVAL_JUDGE_MODEL:-}" ]]; then
  JUDGE="$EVAL_JUDGE_MODEL"
elif [[ -n "${GEMINI_API_KEY:-}" ]]; then
  JUDGE="google:gemini-2.5-flash"
else
  JUDGE="ollama:qwen2.5:7b"
fi
PROVIDER="${LLM_EVAL_PROVIDER:-gemini}"

if [[ -z "${GEMINI_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" && -z "${ANTHROPIC_API_KEY:-}" && -z "${XAI_API_KEY:-}" && -z "${OLLAMA_HOST:-}" ]]; then
  echo "Missing LLM credential: set GEMINI_API_KEY (required in CI) or another provider." >&2
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
  --judge-repeat "$JUDGE_REPEAT" \
  --thresholds \
  --report-json "$REPORT_DIR/agent-eval-live-report.json" \
  --report-md "$REPORT_DIR/agent-eval-live-report.md" \
  --save-live-baseline

echo "Reports:"
echo "  $REPORT_DIR/agent-eval-live-report_lib_off.json (threshold-gated OFF arm)"
echo "  $REPORT_DIR/agent-eval-live-report_lib_on.json (ON arm + grounding/schema vs OFF)"
echo "  $REPORT_DIR/agent-eval-live-report_library_comparison.{json,md}"
echo "  $REPORT_DIR/agent-eval-live-report.md (combined summary)"
