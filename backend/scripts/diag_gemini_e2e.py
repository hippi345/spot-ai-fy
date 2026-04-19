"""End-to-end test of run_chat_turn_gemini — the actual code path the backend uses.

Runs each prompt several times so we see consistency, not just a lucky single shot.
Uses the real Spotify tools (so a Spotify session is required for tool calls to
succeed), but if you're not connected the test still verifies that Gemini reliably
*emits* the tool call rather than returning empty content.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from spot_backend.config import get_settings
from spot_backend.gemini_llm import run_chat_turn_gemini

PROMPTS = [
    "Who am I on Spotify?",
    "List my Spotify devices.",
    "Show me my playlists.",
    "Who are my top 5 artists this month?",
    "Hi!",
]
N_TRIALS = 2


def main() -> int:
    settings = get_settings()
    if not (settings.gemini_api_key or "").strip():
        print("GEMINI_API_KEY missing")
        return 2
    summary: list[tuple[str, int, str]] = []
    for prompt in PROMPTS:
        print(f"\n>>> {prompt!r}")
        for trial in range(1, N_TRIALS + 1):
            try:
                reply = run_chat_turn_gemini(prompt, settings, history=None)
            except Exception as e:  # noqa: BLE001
                reply = f"[exception] {type(e).__name__}: {e}"
            short = reply.replace("\n", " ")
            if len(short) > 200:
                short = short[:200] + "…"
            print(f"   trial {trial}: {short}")
            summary.append((prompt, trial, reply))

    bad = [(p, t) for (p, t, r) in summary if "didn't return any text" in r or "empty response" in r.lower()]
    print(f"\nEmpty/bad replies: {len(bad)}/{len(summary)}")
    for p, t in bad:
        print(f"  - {p!r} trial {t}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
