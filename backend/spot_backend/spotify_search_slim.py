"""Slim Spotify search JSON for LLM consumption (null-safe)."""

from __future__ import annotations

from typing import Any


def strip_null_search_items(data: dict[str, Any]) -> None:
    """Remove literal null placeholders from search `items` arrays (dev-mode sparsification)."""
    for bucket_key in (
        "tracks",
        "artists",
        "albums",
        "playlists",
        "shows",
        "episodes",
        "audiobooks",
    ):
        bucket = data.get(bucket_key)
        if not isinstance(bucket, dict):
            continue
        raw_items = bucket.get("items") if isinstance(bucket.get("items"), list) else []
        clean = [it for it in raw_items if isinstance(it, dict)]
        bucket["items"] = clean
        bucket["returned_count"] = len(clean)
        if "total" in bucket:
            bucket["total_in_catalog"] = bucket.get("total")


def slim_show_search_items(shows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in shows:
        if not isinstance(row, dict):
            continue
        out.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "publisher": row.get("publisher"),
                "uri": row.get("uri"),
                "description": (row.get("description") or "")[:240],
            }
        )
    return out


def attach_show_search_summary(data: dict[str, Any]) -> None:
    bucket = data.get("shows")
    if not isinstance(bucket, dict):
        return
    items = bucket.get("items")
    if not isinstance(items, list):
        return
    slim = slim_show_search_items(items)
    bucket["items"] = slim
    from spot_backend.list_format import format_indexed_name_detail

    lines = [
        format_indexed_name_detail(
            i + 1,
            str(r.get("name") or ""),
            str(r.get("publisher") or "") if r.get("publisher") else None,
            default_name="Show",
        )
        for i, r in enumerate(slim)
    ]
    if lines:
        data["show_search_summary_lines"] = lines
        data["user_message"] = "\n".join(lines)
