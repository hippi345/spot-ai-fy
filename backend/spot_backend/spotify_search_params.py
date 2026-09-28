"""Build validated query params for Spotify GET /v1/search (dev-mode 2025–2026)."""

from __future__ import annotations

from typing import Any

from spot_backend.spotify_dev_limits import SPOTIFY_SEARCH_DEFAULT_LIMIT

# Spotify rejects plural type tokens (e.g. type=shows); models often emit them.
_TYPE_ALIASES: dict[str, str] = {
    "tracks": "track",
    "track": "track",
    "artists": "artist",
    "artist": "artist",
    "albums": "album",
    "album": "album",
    "playlists": "playlist",
    "playlist": "playlist",
    "shows": "show",
    "show": "show",
    "podcast": "show",
    "podcasts": "show",
    "episodes": "episode",
    "episode": "episode",
    "audiobooks": "audiobook",
    "audiobook": "audiobook",
}

_SHOW_OR_EPISODE = frozenset({"show", "episode"})


def normalize_search_type_tokens(raw: str) -> str:
    """Comma-separated Spotify search `type` values, each a valid API token."""
    text = (raw or "").strip().replace(" ", "")
    if not text:
        return "track,artist,album"
    out: list[str] = []
    seen: set[str] = set()
    for part in text.split(","):
        if not part:
            continue
        token = _TYPE_ALIASES.get(part.lower())
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return ",".join(out) if out else "track"


def search_types_need_market(types_csv: str) -> bool:
    tokens = {t.strip() for t in types_csv.split(",") if t.strip()}
    return bool(tokens & _SHOW_OR_EPISODE)


def build_spotify_search_params(
    *,
    query: str,
    types_raw: str,
    market: str,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    """Params dict for GET /search; omits invalid keys and adds podcast-friendly flags."""
    types_norm = normalize_search_type_tokens(types_raw)
    lim = max(1, min(int(limit), 10))
    off = max(0, int(offset))
    market_norm = (market or "").strip() or "from_token"
    if market_norm.lower() != "from_token" and (
        len(market_norm) != 2 or not market_norm.isalpha()
    ):
        market_norm = "from_token"
    if search_types_need_market(types_norm) and market_norm.lower() != "from_token":
        # Show/episode catalog is tied to the user's market; avoid bogus ISO codes from the model.
        market_norm = "from_token"
    params: dict[str, Any] = {
        "q": query,
        "type": types_norm,
        "market": market_norm,
        "limit": lim,
        "offset": off,
    }
    if search_types_need_market(types_norm):
        params["include_external"] = "audio"
    return params


def pick_search_types_argument(arguments: dict[str, Any]) -> str:
    """Read model tool args (`types` or `type`) with a safe default."""
    for key in ("types", "type"):
        val = arguments.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "track,artist,album"


def default_search_limit(arguments: dict[str, Any]) -> int:
    raw = arguments.get("limit")
    try:
        return int(raw) if raw is not None else SPOTIFY_SEARCH_DEFAULT_LIMIT
    except (TypeError, ValueError):
        return SPOTIFY_SEARCH_DEFAULT_LIMIT
