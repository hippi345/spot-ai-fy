from __future__ import annotations

from spot_backend.library_index import format_fts_match_query


def test_fts_query_quotes_tokens_or_joined() -> None:
    q = format_fts_match_query("rainy day jazz")
    assert q == '"rainy" OR "day" OR "jazz"'


def test_fts_query_ac_dc_slash() -> None:
    q = format_fts_match_query("AC/DC back in black")
    assert '"AC"' in q and '"DC"' in q
    assert q.endswith('"black"')
    assert " OR " in q
    assert not q.endswith(" OR ")


def test_fts_query_hyphenated_artist() -> None:
    q = format_fts_match_query("jay-z")
    assert q == '"jay" OR "z"'
