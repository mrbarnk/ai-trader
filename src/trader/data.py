from __future__ import annotations

from datetime import datetime

from . import config
from . import mt5_client
from .models import Candle
from .time_utils import broker_epoch_to_utc
from .timeframes import TIMEFRAME_SECONDS


def rates_to_candles(rates) -> list[Candle]:
    candles: list[Candle] = []
    for rate in rates:
        time_utc = broker_epoch_to_utc(int(rate["time"]))
        candles.append(
            Candle(
                time_utc=time_utc,
                open=float(rate["open"]),
                high=float(rate["high"]),
                low=float(rate["low"]),
                close=float(rate["close"]),
            )
        )
    return candles


def drop_incomplete_candle(
    candles: list[Candle], timeframe_seconds: int, now_utc: datetime
) -> tuple[list[Candle], bool]:
    if not candles:
        return candles, False
    last = candles[-1]
    close_time = last.time_utc.timestamp() + timeframe_seconds
    if now_utc.timestamp() < close_time:
        return candles[:-1], True
    return candles, False


def get_closed_candles(symbol: str, timeframe: int, now_utc: datetime) -> tuple[list[Candle], bool]:
    rates = mt5_client.get_rates(symbol, timeframe, config.CANDLE_COUNT)
    candles = rates_to_candles(rates)
    timeframe_seconds = TIMEFRAME_SECONDS[timeframe]
    return drop_incomplete_candle(candles, timeframe_seconds, now_utc)
