from __future__ import annotations

from datetime import datetime, time as dt_time, timedelta

from . import config


def broker_epoch_to_utc(epoch_seconds: int) -> datetime:
    """Convert broker epoch seconds to UTC using configured offset."""
    broker_dt = datetime.utcfromtimestamp(epoch_seconds)
    return broker_dt - timedelta(hours=config.BROKER_UTC_OFFSET_HOURS)


def is_time_in_window(now_time: dt_time, start: dt_time, end: dt_time) -> bool:
    return start <= now_time < end


def session_from_utc(now_utc: datetime) -> str | None:
    now_time = now_utc.time()
    if is_time_in_window(now_time, config.LONDON_START_UTC, config.LONDON_END_UTC):
        return "LONDON"
    if config.NY_ENABLED and is_time_in_window(now_time, config.NY_START_UTC, config.NY_END_UTC):
        return "NY"
    return None
