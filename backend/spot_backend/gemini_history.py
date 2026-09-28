"""Validate and normalize Gemini generateContent history (tool turns, thought signatures)."""

from __future__ import annotations

from typing import Any


def gemini_part_for_history(part: dict[str, Any]) -> dict[str, Any]:
    """Preserve fields Gemini 3.x needs when replaying a model turn (incl. thoughtSignature)."""
    if not isinstance(part, dict):
        return {}
    out: dict[str, Any] = {}
    if "text" in part and isinstance(part.get("text"), str):
        out["text"] = part["text"]
    if part.get("thought") is True:
        out["thought"] = True
    sig = part.get("thoughtSignature")
    if isinstance(sig, str) and sig.strip():
        out["thoughtSignature"] = sig
    fc = part.get("functionCall")
    if isinstance(fc, dict) and fc.get("name"):
        out["functionCall"] = fc
    fr = part.get("functionResponse")
    if isinstance(fr, dict) and fr.get("name"):
        out["functionResponse"] = fr
    return out


_REPAIRED_THOUGHT_SIGNATURE = "repaired_replay_sig"


def _part_is_nonempty(part: Any) -> bool:
    if not isinstance(part, dict):
        return False
    if part.get("functionCall") or part.get("functionResponse"):
        return True
    text = part.get("text")
    if isinstance(text, str) and text.strip():
        return True
    return False


def _ensure_model_function_call_signatures(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        p = dict(part)
        if p.get("functionCall") and not p.get("thoughtSignature") and not p.get("thought"):
            p["thoughtSignature"] = _REPAIRED_THOUGHT_SIGNATURE
        out.append(p)
    return out


def repair_gemini_contents(contents: list[Any]) -> list[dict[str, Any]]:
    """Best-effort fix for Gemini multi-turn history before generateContent."""
    if not isinstance(contents, list):
        return []
    repaired: list[dict[str, Any]] = []
    for entry in contents:
        if not isinstance(entry, dict):
            continue
        role = entry.get("role")
        if role not in ("user", "model"):
            continue
        raw_parts = entry.get("parts")
        if not isinstance(raw_parts, list):
            continue
        parts: list[dict[str, Any]] = []
        for part in raw_parts:
            if not _part_is_nonempty(part):
                continue
            cleaned = gemini_part_for_history(part) if isinstance(part, dict) else {}
            if cleaned:
                parts.append(cleaned)
        if not parts:
            continue
        if role == "model":
            parts = _ensure_model_function_call_signatures(parts)

        if (
            role == "user"
            and repaired
            and repaired[-1].get("role") == "user"
            and any(
                isinstance(p, dict) and p.get("functionResponse")
                for p in (repaired[-1].get("parts") or [])
            )
            and all(
                not isinstance(p, dict) or (not p.get("functionResponse") and not p.get("functionCall"))
                for p in parts
            )
        ):
            merged_parts = list(repaired[-1].get("parts") or []) + parts
            repaired[-1] = {"role": "user", "parts": merged_parts}
            continue

        if (
            role == "user"
            and repaired
            and repaired[-1].get("role") == "user"
            and any(p.get("functionResponse") for p in parts)
        ):
            prev_parts = repaired[-1].get("parts") or []
            prev_is_plain = all(
                not isinstance(p, dict) or (not p.get("functionResponse") and not p.get("functionCall"))
                for p in prev_parts
            )
            if prev_is_plain:
                synthetic: list[dict[str, Any]] = []
                for p in parts:
                    fr = p.get("functionResponse") if isinstance(p, dict) else None
                    if isinstance(fr, dict) and fr.get("name"):
                        synthetic.append(
                            {
                                "thoughtSignature": _REPAIRED_THOUGHT_SIGNATURE,
                                "functionCall": {"name": str(fr["name"]), "args": {}},
                            }
                        )
                if synthetic:
                    repaired.append({"role": "model", "parts": synthetic})

        if (
            repaired
            and repaired[-1].get("role") == role
            and role == "model"
            and not any(p.get("functionResponse") for p in parts)
            and not any(p.get("functionCall") for p in parts)
        ):
            merged = list(repaired[-1].get("parts") or [])
            for p in parts:
                if isinstance(p, dict) and isinstance(p.get("text"), str):
                    if merged and isinstance(merged[-1], dict) and isinstance(merged[-1].get("text"), str):
                        merged[-1] = {
                            **merged[-1],
                            "text": (merged[-1]["text"].rstrip() + "\n" + p["text"]).strip(),
                        }
                    else:
                        merged.append(p)
                else:
                    merged.append(p)
            repaired[-1] = {"role": role, "parts": merged}
            continue

        repaired.append({"role": role, "parts": parts})

    if repaired and repaired[0].get("role") != "user":
        repaired.insert(0, {"role": "user", "parts": [{"text": "(conversation context)"}]})
    return repaired


def validate_gemini_contents(contents: list[Any]) -> list[str]:
    """Return human-readable violations of Gemini multi-turn rules (empty list = ok)."""
    errors: list[str] = []
    if not contents:
        errors.append("contents is empty")
        return errors
    if not isinstance(contents, list):
        errors.append("contents is not a list")
        return errors

    prev_role: str | None = None
    for i, entry in enumerate(contents):
        if not isinstance(entry, dict):
            errors.append(f"contents[{i}] is not an object")
            continue
        role = entry.get("role")
        if role not in ("user", "model"):
            errors.append(f"contents[{i}].role must be user|model, got {role!r}")
            continue
        parts = entry.get("parts")
        if not isinstance(parts, list) or not parts:
            errors.append(f"contents[{i}] has empty or missing parts")
            continue
        has_fc = False
        has_fr = False
        has_text = False
        for j, part in enumerate(parts):
            if not isinstance(part, dict):
                errors.append(f"contents[{i}].parts[{j}] is not an object")
                continue
            if part.get("functionCall"):
                has_fc = True
                if role != "model":
                    errors.append(f"contents[{i}]: functionCall on non-model role")
                if not isinstance(part.get("functionCall"), dict) or not part["functionCall"].get("name"):
                    errors.append(f"contents[{i}].parts[{j}].functionCall missing name")
            if part.get("functionResponse"):
                has_fr = True
                if role != "user":
                    errors.append(f"contents[{i}]: functionResponse on non-user role")
            if isinstance(part.get("text"), str) and part["text"].strip():
                has_text = True
        if role == "model" and has_fr:
            errors.append(f"contents[{i}]: model turn must not contain functionResponse")
        if role == "user" and has_fc:
            errors.append(f"contents[{i}]: user turn must not contain functionCall")
        if role == "user" and has_fr and has_text and not (
            len(parts) > 1 and all(isinstance(p, dict) for p in parts)
        ):
            errors.append(f"contents[{i}]: user turn mixes functionResponse with plain text")
        if role == "model" and has_fc:
            for j, part in enumerate(parts):
                if not isinstance(part, dict) or not part.get("functionCall"):
                    continue
                if not part.get("thoughtSignature") and not part.get("thought"):
                    errors.append(
                        f"contents[{i}].parts[{j}]: functionCall missing thoughtSignature "
                        "(required for Gemini 3.x replay)"
                    )
        if prev_role == role:
            if role == "user" and has_fr:
                pass
            elif role == "model" and has_fc:
                pass
            else:
                errors.append(f"contents[{i}]: consecutive {role} turns without tool response bridge")
        if (
            role == "user"
            and has_fr
            and prev_role == "user"
        ):
            errors.append(
                f"contents[{i}]: functionResponse user turn immediately after plain user turn "
                "(missing model functionCall)"
            )
        prev_role = role

    if contents[0].get("role") != "user":
        errors.append("contents must start with role=user")
    return errors
