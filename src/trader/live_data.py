from __future__ import annotations

from datetime import datetime

from .data import Candle, get_closed_candles


class LiveCandleProvider:
    def get_closed_candles(
        self, symbol: str, timeframe: int, now_utc: datetime
    ) -> tuple[list[Candle], bool]:
        return get_closed_candles(symbol, timeframe, now_utc)
