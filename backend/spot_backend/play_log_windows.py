"""Resolve natural-language play-history windows to UTC ranges."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class TimeWindow:
    start_utc: datetime
    end_utc: datetime
    label: str


def _local_now(tz: ZoneInfo) -> datetime:
    return datetime.now(tz)


def _day_bounds_local(d: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    start = datetime.combine(d, time.min, tzinfo=tz)
    end = datetime.combine(d, time.max, tzinfo=tz)
    return start, end


def _to_utc_pair(start_local: datetime, end_local: datetime) -> tuple[datetime, datetime]:
    return (
        start_local.astimezone(timezone.utc),
        end_local.astimezone(timezone.utc),
    )


def resolve_play_history_window(
    window: str,
    *,
    tz_name: str = "UTC",
    now: datetime | None = None,
    explicit_start: str | None = None,
    explicit_end: str | None = None,
) -> TimeWindow:
    """Map window names to inclusive UTC [start, end]."""
    tz = ZoneInfo((tz_name or "UTC").strip() or "UTC")
    local_now = (now or _local_now(tz)).astimezone(tz)
    today = local_now.date()

    if explicit_start and explicit_end:
        s = datetime.fromisoformat(explicit_start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(explicit_end.replace("Z", "+00:00"))
        if s.tzinfo is None:
            s = s.replace(tzinfo=timezone.utc)
        if e.tzinfo is None:
            e = e.replace(tzinfo=timezone.utc)
        return TimeWindow(s.astimezone(timezone.utc), e.astimezone(timezone.utc), "custom")

    key = (window or "this_week").strip().lower().replace("-", "_").replace(" ", "_")
    key = re.sub(r"_+", "_", key)

    if key in ("today",):
        s, e = _day_bounds_local(today, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "today")

    if key in ("yesterday",):
        d = today - timedelta(days=1)
        s, e = _day_bounds_local(d, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "yesterday")

    if key in ("this_week", "week"):
        # ISO week: Monday start
        weekday = today.weekday()
        monday = today - timedelta(days=weekday)
        sunday = monday + timedelta(days=6)
        s, _ = _day_bounds_local(monday, tz)
        _, e = _day_bounds_local(sunday, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "this_week")

    if key in ("last_7_days", "last7", "past_week"):
        end_d = today
        start_d = today - timedelta(days=6)
        s, _ = _day_bounds_local(start_d, tz)
        _, e = _day_bounds_local(end_d, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "last_7_days")

    if key in ("last_30_days", "last30", "past_month"):
        end_d = today
        start_d = today - timedelta(days=29)
        s, _ = _day_bounds_local(start_d, tz)
        _, e = _day_bounds_local(end_d, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "last_30_days")

    if key in ("last_weekend", "weekend"):
        # Most recent Sat–Sun block (including current weekend if Sat/Sun)
        wd = today.weekday()
        if wd >= 5:
            saturday = today - timedelta(days=wd - 5)
        else:
            saturday = today - timedelta(days=wd + 2)
        sunday = saturday + timedelta(days=1)
        s, _ = _day_bounds_local(saturday, tz)
        _, e = _day_bounds_local(sunday, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, "last_weekend")

    m = re.match(r"last_(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", key)
    if m:
        names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        target = names.index(m.group(1))
        wd = today.weekday()
        days_back = (wd - target) % 7
        if days_back == 0:
            days_back = 7
        d = today - timedelta(days=days_back)
        s, e = _day_bounds_local(d, tz)
        a, b = _to_utc_pair(s, e)
        return TimeWindow(a, b, f"last_{m.group(1)}")

    if key in ("last_friday_night", "friday_night"):
        tw = resolve_play_history_window("last_friday", tz_name=tz_name, now=now)
        # 18:00–06:00 local Friday night into Saturday morning
        fri = tw.start_utc.astimezone(tz).date()
        start_local = datetime.combine(fri, time(18, 0), tzinfo=tz)
        end_local = datetime.combine(fri + timedelta(days=1), time(6, 0), tzinfo=tz)
        a, b = _to_utc_pair(start_local, end_local)
        return TimeWindow(a, b, "last_friday_night")

    raise ValueError(f"unknown play history window: {window!r}")
