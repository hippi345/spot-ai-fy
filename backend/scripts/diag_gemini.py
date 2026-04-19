"""Standalone diagnostic: probe the Gemini API with our exact tool-declaration list.

Run from the backend folder so Settings can find .env:

    cd backend
    .\.venv\Scripts\python.exe scripts\diag_gemini.py

Prints, for each test prompt:
  - The HTTP status returned
  - finishReason, usageMetadata
  - How many parts came back, how many were thought-only
  - The first 2000 chars of the raw candidate JSON

This bypasses the chat history loop so you can see what the very first call
to generateContent actually returns.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make `from spot_backend...` work whether you launch from backend/ or repo root.
HERE = Path(__file__).resolve()
BACKEND_DIR = HERE.parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx

from spot_backend.config import get_settings
from spot_backend.gemini_llm import (
    _DEFAULT_GEMINI_MODEL,
    _GEMINI_REST,
    _SYSTEM,
    _openai_tools_to_gemini_declarations,
)
from spot_backend.llm_prefs import read_effective_gemini_model
from spot_backend.spotify_tools import OLLAMA_TOOLS

PROMPTS = [
    "Say hi in one word.",
    "Who am I on Spotify?",
    "List my Spotify devices.",
]


def main() -> int:
    settings = get_settings()
    key = (settings.gemini_api_key or "").strip()
    if not key:
        print("GEMINI_API_KEY is not set in backend/.env")
        return 2

    model = read_effective_gemini_model(settings.data_dir, settings.gemini_model) or _DEFAULT_GEMINI_MODEL
    print(f"Model: {model}")
    print(f"Tools wired: {len(OLLAMA_TOOLS)}")
    declarations = _openai_tools_to_gemini_declarations(OLLAMA_TOOLS)
    print(f"Declarations after Gemini transform: {len(declarations)}")
    print(f"System prompt chars: {len(_SYSTEM)}")
    print()

    url = f"{_GEMINI_REST}/models/{model}:generateContent"

    # Tools added recently, in the order they appear in OLLAMA_TOOLS:
    new_tool_names = [
        "spotify_top_artists",
        "spotify_top_tracks",
        "spotify_followed_artists",
        "spotify_user_public_playlists",
        "spotify_search_playlists",
        "spotify_follow_playlist",
        "spotify_duplicate_playlist",
    ]
    decls_by_name = {d["name"]: d for d in declarations}

    def _probe(label: str, decls_subset: list[dict], prompt: str, client: httpx.Client) -> bool:
        body: dict = {
            "systemInstruction": {"parts": [{"text": _SYSTEM}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "tools": [{"functionDeclarations": decls_subset}],
            "toolConfig": {"functionCallingConfig": {"mode": "AUTO"}},
        }
        resp = client.post(url, params={"key": key}, json=body)
        if resp.status_code != 200:
            print(f"  [{label}] HTTP {resp.status_code}: {resp.text[:600]}")
            return False
        data = resp.json()
        cand = (data.get("candidates") or [{}])[0]
        parts = (cand.get("content") or {}).get("parts") or []
        n_text = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("text").strip())
        n_fc = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict))
        ok = (n_text + n_fc) > 0
        marker = "OK " if ok else "BAD"
        fcs = [p["functionCall"].get("name") for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)]
        print(f"  [{label}] {marker} parts(text={n_text}, fc={n_fc}) finish={cand.get('finishReason')!r} fcs={fcs}")
        return ok

    with httpx.Client(timeout=httpx.Timeout(60.0, connect=30.0)) as client:
        for variant_name, tools_block in (
            ("WITH_TOOLS", [{"functionDeclarations": declarations}]),
            ("NO_TOOLS", None),
        ):
            print("=" * 78)
            print(f"VARIANT: {variant_name}")
            print("=" * 78)
            for prompt in PROMPTS:
                body: dict = {
                    "systemInstruction": {"parts": [{"text": _SYSTEM}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                }
                if tools_block is not None:
                    body["tools"] = tools_block
                    body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}
                resp = client.post(url, params={"key": key}, json=body)
                print(f"\n>>> prompt: {prompt!r}")
                print(f"    status: {resp.status_code}")
                if resp.status_code != 200:
                    print(f"    body[:1000]: {resp.text[:1000]}")
                    continue
                data = resp.json()
                cand = (data.get("candidates") or [{}])[0]
                parts = (cand.get("content") or {}).get("parts") or []
                n_thought = sum(1 for p in parts if isinstance(p, dict) and p.get("thought"))
                n_text = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("text").strip())
                n_fc = sum(1 for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict))
                print(f"    finishReason: {cand.get('finishReason')!r}")
                print(f"    parts: total={len(parts)} thought={n_thought} text(non-empty)={n_text} functionCall={n_fc}")
                print(f"    usage: {data.get('usageMetadata')}")
                if n_text == 0 and n_fc == 0:
                    print(f"    candidate[:1500]: {json.dumps(cand)[:1500]}")
                else:
                    if n_text:
                        snippets = [p["text"][:200] for p in parts if isinstance(p, dict) and isinstance(p.get("text"), str) and p["text"].strip()]
                        print(f"    text_snippets: {snippets[:2]}")
                    if n_fc:
                        fcs = [(p["functionCall"].get("name"), p["functionCall"].get("args")) for p in parts if isinstance(p, dict) and isinstance(p.get("functionCall"), dict)]
                        print(f"    function_calls: {fcs}")

        print("\n" + "=" * 78)
        print("BISECT: which tool(s) break Gemini for 'Who am I on Spotify?'")
        print("=" * 78)
        bisect_prompt = "Who am I on Spotify?"
        old_decls = [d for d in declarations if d["name"] not in new_tool_names]
        print(f"\n-- OLD tools only ({len(old_decls)} decls) --")
        _probe("OLD", old_decls, bisect_prompt, client)
        print(f"\n-- OLD + each NEW tool, one at a time --")
        for n in new_tool_names:
            subset = old_decls + [decls_by_name[n]]
            _probe(n, subset, bisect_prompt, client)
        print(f"\n-- ALL OLD + each PAIR of new tools (OLD + n1 + n2) is too noisy; instead: ALL NEW only --")
        new_only = [decls_by_name[n] for n in new_tool_names if n in decls_by_name]
        _probe("NEW_ONLY", new_only, bisect_prompt, client)

        # Bisect OLD tools by halves. We binary-narrow until we find the smallest
        # subset that still fails. Then we know exactly which schema(s) poison Gemini.
        print("\n" + "=" * 78)
        print("BISECT: narrowing OLD tools to find the breaker")
        print("=" * 78)
        # Sanity: empty tools should be OK.
        _probe("EMPTY", [], bisect_prompt, client)

        old_names = [d["name"] for d in old_decls]
        print(f"\n-- OLD names ({len(old_names)}): {old_names}")

        # Halve repeatedly, keeping the failing half.
        current = list(old_decls)
        attempt = 0
        while len(current) > 1 and attempt < 12:
            attempt += 1
            mid = len(current) // 2
            left = current[:mid]
            right = current[mid:]
            l_names = [d["name"] for d in left]
            r_names = [d["name"] for d in right]
            print(f"\n[attempt {attempt}] split into LEFT({len(left)}) and RIGHT({len(right)})")
            l_ok = _probe(f"LEFT[{','.join(l_names[:3])}...({len(l_names)})]", left, bisect_prompt, client)
            r_ok = _probe(f"RIGHT[{','.join(r_names[:3])}...({len(r_names)})]", right, bisect_prompt, client)
            if not l_ok and r_ok:
                current = left
            elif not r_ok and l_ok:
                current = right
            elif not l_ok and not r_ok:
                # Both halves break. Could be a count/size threshold.
                print("  Both halves BAD — likely a tool-count or total-schema-size threshold,")
                print("  not a single bad schema. Stopping bisect.")
                break
            else:
                print("  Both halves OK — issue requires both halves together. Stopping bisect.")
                break

        if len(current) <= 1:
            print(f"\nSmallest failing OLD subset: {[d['name'] for d in current]}")
        else:
            final_names = [d["name"] for d in current]
            print(f"\nFinal narrowed OLD subset ({len(final_names)}): {final_names}")

        # Print description sizes so we can see which tools have monster descriptions.
        print("\n" + "=" * 78)
        print("Tool description sizes (chars). Long ones are the prime suspects.")
        print("=" * 78)
        sized = []
        for d in declarations:
            desc_chars = len(d.get("description", ""))
            params_chars = len(json.dumps(d.get("parameters", {})))
            sized.append((d["name"], desc_chars, params_chars, desc_chars + params_chars))
        sized.sort(key=lambda x: -x[3])
        for name, dc, pc, total in sized:
            print(f"  {name:42s}  desc={dc:5d}  params_json={pc:5d}  total={total:5d}")
        print(f"  {'(SUM)':42s}  desc={sum(s[1] for s in sized):5d}  "
              f"params_json={sum(s[2] for s in sized):5d}  "
              f"total={sum(s[3] for s in sized):5d}")

        # Final test: ALL 40 tools with descriptions truncated to 100 chars each.
        # If this works, we know the issue is cumulative description size.
        print("\n" + "=" * 78)
        print("EXPERIMENT: ALL 40 tools, descriptions truncated to 100 chars each")
        print("=" * 78)
        slim = []
        for d in declarations:
            d2 = dict(d)
            desc = d.get("description", "")
            d2["description"] = desc[:100]
            # Also strip property descriptions inside parameters.
            params = d.get("parameters", {})
            if isinstance(params, dict):
                p2 = json.loads(json.dumps(params))
                props = p2.get("properties") if isinstance(p2.get("properties"), dict) else {}
                for k, v in list(props.items()):
                    if isinstance(v, dict) and "description" in v:
                        v["description"] = v["description"][:60] if isinstance(v["description"], str) else v["description"]
                d2["parameters"] = p2
            slim.append(d2)
        _probe("ALL_SLIM", slim, bisect_prompt, client)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
