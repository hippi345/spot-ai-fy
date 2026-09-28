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


def slim_audiobook_search_items(books: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in books:
        if not isinstance(row, dict):
            continue
        authors = row.get("authors")
        author_names: list[str] = []
        if isinstance(authors, list):
            for a in authors:
                if isinstance(a, dict) and isinstance(a.get("name"), str):
                    author_names.append(a["name"])
        narrators = row.get("narrators")
        narrator_names: list[str] = []
        if isinstance(narrators, list):
            for n in narrators:
                if isinstance(n, dict) and isinstance(n.get("name"), str):
                    narrator_names.append(n["name"])
        out.append(
            {
                "id": row.get("id"),
                "name": row.get("name"),
                "uri": row.get("uri"),
                "authors": author_names,
                "narrators": narrator_names,
            }
        )
    return out


def attach_audiobook_search_summary(data: dict[str, Any]) -> None:
    bucket = data.get("audiobooks")
    if not isinstance(bucket, dict):
        return
    items = bucket.get("items")
    if not isinstance(items, list):
        return
    slim = slim_audiobook_search_items(items)
    bucket["items"] = slim
    from spot_backend.list_format import format_indexed_name_detail

    lines = []
    for i, r in enumerate(slim):
        name = str(r.get("name") or "")
        authors = r.get("authors") if isinstance(r.get("authors"), list) else []
        detail = ", ".join(str(a) for a in authors[:2]) if authors else None
        narrators = r.get("narrators") if isinstance(r.get("narrators"), list) else []
        if narrators:
            nar = ", ".join(str(n) for n in narrators[:2])
            detail = f"{detail} — narrated by {nar}" if detail else f"narrated by {nar}"
        lines.append(format_indexed_name_detail(i + 1, name, detail, default_name="Audiobook"))
    if lines:
        data["audiobook_search_summary_lines"] = lines
        data["user_message"] = "\n".join(lines)


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
