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
        if prev_role == role and not (role == "user" and has_fr):
            errors.append(f"contents[{i}]: consecutive {role} turns without tool response bridge")
        prev_role = role

    if contents[0].get("role") != "user":
        errors.append("contents must start with role=user")
    return errors
