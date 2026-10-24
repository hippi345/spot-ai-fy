from __future__ import annotations

from spot_backend.reply_grounding import reply_mentions_catalog_outside_tool_data


def test_flags_unquoted_track_by_artist_after_empty_search() -> None:
    tool = '{"available": true, "mode": "semantic", "results": []}'
    bad = reply_mentions_catalog_outside_tool_data(
        "Here is Unchained by Drake from your library.",
        [tool],
    )
    assert bad


def test_allows_honest_empty_reply() -> None:
    tool = '{"available": true, "results": []}'
    bad = reply_mentions_catalog_outside_tool_data(
        "I couldn't find any jazz in your liked songs.",
        [tool],
    )
    assert not bad
