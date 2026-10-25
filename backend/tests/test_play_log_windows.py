from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from spot_backend.play_log_windows import resolve_play_history_window


def test_this_week_monday_boundary() -> None:
    # Wednesday 2026-10-22 in US/Eastern
    tz = "America/New_York"
    now = datetime(2026, 10, 22, 15, 0, tzinfo=ZoneInfo(tz))
    tw = resolve_play_history_window("this_week", tz_name=tz, now=now)
    start_local = tw.start_utc.astimezone(ZoneInfo(tz))
    assert start_local.weekday() == 0  # Monday
    assert tw.label == "this_week"


def test_yesterday_utc() -> None:
    now = datetime(2026, 10, 24, 12, 0, tzinfo=timezone.utc)
    tw = resolve_play_history_window("yesterday", tz_name="UTC", now=now)
    assert tw.start_utc.day == 23
    assert tw.end_utc.day == 23
