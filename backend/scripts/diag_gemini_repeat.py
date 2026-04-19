"""Repeat the Gemini WITH_TOOLS calls many times to see the empty-content rate."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from spot_backend.config import get_settings
from spot_backend.gemini_llm import (
    _DEFAULT_GEMINI_MODEL,
    _GEMINI_REST,
    _GEMINI_TOOL_DESC_MAX,
    _GEMINI_PARAM_DESC_MAX,
    _SYSTEM,
    _openai_tools_to_gemini_declarations,
)
from spot_backend.llm_prefs import read_effective_gemini_model
from spot_backend.spotify_tools import OLLAMA_TOOLS

PROMPTS = [
    "Who am I on Spotify?",
    "List my Spotify devices.",
    "What's playing right now?",
    "Show me my playlists.",
    "Who are my top 5 artists this month?",
]
N_TRIALS = 3


def main() -> int:
    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()
    if not key:
        print("GEMINI_API_KEY missing")
        return 2
    model = read_effective_gemini_model(settings.data_dir, settings.gemini_model) or _DEFAULT_GEMINI_MODEL
    declarations = _openai_tools_to_gemini_declarations(OLLAMA_TOOLS)
    print(f"Model: {model}")
    print(f"Tools: {len(declarations)}  desc_cap={_GEMINI_TOOL_DESC_MAX}  param_desc_cap={_GEMINI_PARAM_DESC_MAX}")
    url = f"{_GEMINI_REST}/models/{model}:generateContent"

    SHORT_SYSTEM = (
        "You are a Spotify assistant. Use the provided tools to answer the user's "
        "question. Pick the most appropriate tool and call it; then summarize the result."
    )

    matrix = [
        ("FULL_PROMPT_AUTO", _SYSTEM, "AUTO"),
        ("SHORT_PROMPT_AUTO", SHORT_SYSTEM, "AUTO"),
        ("FULL_PROMPT_ANY", _SYSTEM, "ANY"),
        ("SHORT_PROMPT_ANY", SHORT_SYSTEM, "ANY"),
    ]

    grand = {}
    with httpx.Client(timeout=httpx.Timeout(60.0, connect=30.0)) as client:
        for label, sys_prompt, mode in matrix:
            print(f"\n{'=' * 78}\n{label}  system_chars={len(sys_prompt)}  mode={mode}\n{'=' * 78}")
            tot = {"ok": 0, "empty": 0, "other": 0}
            for prompt in PROMPTS:
                print(f"\n>>> {prompt!r}")
                for trial in range(1, N_TRIALS + 1):
                    body = {
                        "systemInstruction": {"parts": [{"text": sys_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                        "tools": [{"functionDeclarations": declarations}],
                        "toolConfig": {"functionCallingConfig": {"mode": mode}},
                    }
                    resp = client.post(url, params={"key": key}, json=body)
                    if resp.status_code != 200:
                        print(f"   trial {trial}: HTTP {resp.status_code} {resp.text[:200]}")
                        tot["other"] += 1
                        continue
                    cand = (resp.json().get("candidates") or [{}])[0]
                    parts = (cand.get("content") or {}).get("parts") or []
                    n_text = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("text").strip())
                    n_fc = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict))
                    if n_text + n_fc > 0:
                        fcs = [p["functionCall"].get("name") for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)]
                        print(f"   trial {trial}: OK   text={n_text} fc={n_fc} {fcs}")
                        tot["ok"] += 1
                    else:
                        print(f"   trial {trial}: EMPTY finishReason={cand.get('finishReason')}")
                        tot["empty"] += 1
            grand[label] = tot
            print(f"\nSubtotal {label}: {tot}")

    print("\n\nGRAND TOTALS")
    for k, v in grand.items():
        denom = v["ok"] + v["empty"] + v["other"]
        print(f"  {k:24s} ok={v['ok']:2d}  empty={v['empty']:2d}  other={v['other']:2d}  ({v['empty']}/{denom} empty)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
