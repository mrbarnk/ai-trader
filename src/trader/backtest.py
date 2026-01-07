from __future__ import annotations

import argparse
import csv
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .logger import SignalLogger
from .models import Candle
from .rules_engine import SignalEngine
from .time_utils import session_from_utc
from .timeframes import (
    TIMEFRAME_H4,
    TIMEFRAME_M15,
    TIMEFRAME_M5,
    TIMEFRAME_M1,
    TIMEFRAME_SECONDS,
)


@dataclass(frozen=True)
class TimeframeSeries:
    timeframe: int
    candles: list[Candle]
    close_times: list[datetime]


class HistoricalCandleProvider:
    def __init__(self, series: dict[int, TimeframeSeries]):
        self.series = series

    def get_closed_candles(
        self, symbol: str, timeframe: int, now_utc: datetime
    ) -> tuple[list[Candle], bool]:
        tf_series = self.series.get(timeframe)
        if tf_series is None:
            return [], False
        idx = bisect_right(tf_series.close_times, now_utc)
        start = max(0, idx - config.CANDLE_COUNT)
        return tf_series.candles[start:idx], False


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip()
    if cleaned.endswith("Z"):
        cleaned = cleaned.replace("Z", "+00:00")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def load_csv_candles(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV missing header row")
        headers = {name.lower(): name for name in reader.fieldnames}
        time_key = None
        for key in ("timestamp_utc", "timestamp", "time", "datetime"):
            if key in headers:
                time_key = headers[key]
                break
        if time_key is None:
            raise ValueError("CSV must include a timestamp column")
        for row in reader:
            time_utc = parse_timestamp(row[time_key])
            candles.append(
                Candle(
                    time_utc=time_utc,
                    open=float(row[headers["open"]]),
                    high=float(row[headers["high"]]),
                    low=float(row[headers["low"]]),
                    close=float(row[headers["close"]]),
                )
            )
    candles.sort(key=lambda candle: candle.time_utc)
    return candles


def resample_candles(
    candles: list[Candle], source_seconds: int, target_seconds: int
) -> list[Candle]:
    if target_seconds % source_seconds != 0:
        raise ValueError("Target timeframe must be a multiple of source timeframe")

    candles_by_bucket: dict[int, list[Candle]] = {}
    for candle in candles:
        epoch = int(candle.time_utc.timestamp())
        bucket = epoch - (epoch % target_seconds)
        candles_by_bucket.setdefault(bucket, []).append(candle)

    expected = target_seconds // source_seconds
    resampled: list[Candle] = []
    for bucket in sorted(candles_by_bucket):
        bucket_candles = candles_by_bucket[bucket]
        if len(bucket_candles) != expected:
            continue
        bucket_candles.sort(key=lambda c: c.time_utc)
        resampled.append(
            Candle(
                time_utc=datetime.utcfromtimestamp(bucket),
                open=bucket_candles[0].open,
                high=max(c.high for c in bucket_candles),
                low=min(c.low for c in bucket_candles),
                close=bucket_candles[-1].close,
            )
        )
    return resampled


def build_series(candles: list[Candle], timeframe: int) -> TimeframeSeries:
    seconds = TIMEFRAME_SECONDS[timeframe]
    close_times = [candle.time_utc + timedelta(seconds=seconds) for candle in candles]
    return TimeframeSeries(timeframe=timeframe, candles=candles, close_times=close_times)


def run_backtest(
    csv_path: Path,
    source_minutes: int,
    output_path: Path,
    start: datetime | None,
    end: datetime | None,
) -> None:
    base_candles = load_csv_candles(csv_path)
    if start or end:
        base_candles = [
            candle
            for candle in base_candles
            if (start is None or candle.time_utc >= start)
            and (end is None or candle.time_utc <= end)
        ]

    source_seconds = source_minutes * 60
    candles_1m: list[Candle] | None = None
    if config.USE_1M_ENTRY:
        if source_seconds != TIMEFRAME_SECONDS[TIMEFRAME_M1]:
            raise ValueError("USE_1M_ENTRY requires 1-minute source candles")
        candles_1m = resample_candles(
            base_candles, source_seconds, TIMEFRAME_SECONDS[TIMEFRAME_M1]
        )
    candles_5m = resample_candles(
        base_candles, source_seconds, TIMEFRAME_SECONDS[TIMEFRAME_M5]
    )
    candles_15m = resample_candles(
        base_candles, source_seconds, TIMEFRAME_SECONDS[TIMEFRAME_M15]
    )
    candles_4h = resample_candles(
        base_candles, source_seconds, TIMEFRAME_SECONDS[TIMEFRAME_H4]
    )

    series = {
        TIMEFRAME_M5: build_series(candles_5m, TIMEFRAME_M5),
        TIMEFRAME_M15: build_series(candles_15m, TIMEFRAME_M15),
        TIMEFRAME_H4: build_series(candles_4h, TIMEFRAME_H4),
    }
    if config.USE_1M_ENTRY and candles_1m is not None:
        series[TIMEFRAME_M1] = build_series(candles_1m, TIMEFRAME_M1)

    provider = HistoricalCandleProvider(series)
    engine = SignalEngine(symbol=config.SYMBOL_VARIANTS[0], candle_provider=provider)
    logger = SignalLogger(str(output_path))

    step_timeframe = TIMEFRAME_M1 if config.USE_1M_ENTRY else TIMEFRAME_M5
    step_seconds = TIMEFRAME_SECONDS[step_timeframe]
    step_candles = candles_1m if config.USE_1M_ENTRY else candles_5m
    if step_candles is None:
        raise ValueError("USE_1M_ENTRY requires 1-minute candle series")
    for candle in step_candles:
        now_utc = candle.time_utc + timedelta(seconds=step_seconds)
        if start and now_utc < start:
            continue
        if end and now_utc > end:
            continue
        if session_from_utc(now_utc) is None:
            continue
        signal = engine.evaluate(now_utc)
        logger.log(signal)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest GU rules using CSV candles.")
    parser.add_argument("csv", type=Path, help="Path to CSV candle data (UTC).")
    parser.add_argument(
        "--source-minutes",
        type=int,
        default=1,
        help="Source candle interval in minutes (default: 1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/backtest_signals.jsonl"),
        help="Output JSONL path for signals.",
    )
    parser.add_argument("--start", type=str, default=None, help="Start time (ISO 8601 UTC).")
    parser.add_argument("--end", type=str, default=None, help="End time (ISO 8601 UTC).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = parse_timestamp(args.start) if args.start else None
    end = parse_timestamp(args.end) if args.end else None
    run_backtest(args.csv, args.source_minutes, args.output, start, end)


if __name__ == "__main__":
    main()
