from __future__ import annotations

import argparse
import csv
from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from . import config
from .logger import SignalLogger
from .models import Candle
from .rules_engine import SignalEngine, SignalOutput
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
    if "." in cleaned and "-" not in cleaned and "T" not in cleaned:
        cleaned = cleaned.replace(".", "-")
    dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def parse_datetime_parts(date_str: str, time_str: str) -> datetime:
    date_part = date_str.strip().replace(".", "-")
    time_part = time_str.strip()
    return parse_timestamp(f"{date_part} {time_part}")


def load_csv_candles(path: Path) -> list[Candle]:
    candles: list[Candle] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError("CSV is empty")
        header = [value.strip().lower() for value in first_row]
        has_header = {"open", "high", "low", "close"}.issubset(set(header))

        if has_header:
            handle.seek(0)
            dict_reader = csv.DictReader(handle)
            if dict_reader.fieldnames is None:
                raise ValueError("CSV missing header row")
            headers = {name.lower(): name for name in dict_reader.fieldnames}
            time_key = None
            for key in ("timestamp_utc", "timestamp", "time", "datetime"):
                if key in headers:
                    time_key = headers[key]
                    break
            if time_key is None:
                raise ValueError("CSV must include a timestamp column")
            for row in dict_reader:
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
        else:
            def parse_row(columns: list[str]) -> Candle | None:
                if len(columns) < 6:
                    return None
                if ":" in columns[0]:
                    time_utc = parse_timestamp(columns[0])
                    offset = 1
                else:
                    time_utc = parse_datetime_parts(columns[0], columns[1])
                    offset = 2
                open_price = float(columns[offset])
                high_price = float(columns[offset + 1])
                low_price = float(columns[offset + 2])
                close_price = float(columns[offset + 3])
                return Candle(
                    time_utc=time_utc,
                    open=open_price,
                    high=high_price,
                    low=low_price,
                    close=close_price,
                )

            first_candle = parse_row(first_row)
            if first_candle:
                candles.append(first_candle)
            for row in reader:
                candle = parse_row(row)
                if candle:
                    candles.append(candle)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
    outcome_path.write_text("", encoding="utf-8")

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
    outcome_logger = SignalLogger(str(outcome_path))

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
        if signal.decision == "TRADE":
            record = build_outcome_record(
                signal=signal,
                series=series,
                entry_timeframe=step_timeframe,
            )
            outcome_logger.log(record)


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


def build_outcome_record(
    signal: SignalOutput,
    series: dict[int, TimeframeSeries],
    entry_timeframe: int,
) -> dict:
    record = asdict(signal)
    tf_series = series.get(entry_timeframe)
    if tf_series is None:
        record.update({"outcome": "NO_DATA"})
        return record
    outcome = simulate_trade_outcome(record, tf_series)
    record.update(outcome)
    return record


def simulate_trade_outcome(record: dict, tf_series: TimeframeSeries) -> dict:
    direction = record.get("direction")
    entry_price = record.get("entry")
    stop_loss = record.get("stop_loss")
    tp1 = record.get("tp1_price")
    tp2 = record.get("tp2_price")
    if direction not in ("BUY", "SELL") or entry_price is None:
        return {"outcome": "INVALID_RECORD"}

    entry_time = parse_timestamp(record["timestamp_utc"])
    open_times = [c.time_utc for c in tf_series.candles]
    start_idx = bisect_left(open_times, entry_time)

    entry_idx = None
    entry_fill_time = None
    for i in range(start_idx, len(tf_series.candles)):
        candle = tf_series.candles[i]
        if candle.low <= entry_price <= candle.high:
            entry_idx = i
            entry_fill_time = candle.time_utc
            break
    if entry_idx is None:
        return {"outcome": "ENTRY_MISS"}

    tp1_hit = False
    tp2_hit = False
    tp1_time = None
    tp2_time = None
    sl_time = None
    active_stop = stop_loss

    for candle in tf_series.candles[entry_idx:]:
        hit_sl = False
        hit_tp1 = False
        hit_tp2 = False

        if direction == "BUY":
            hit_sl = active_stop is not None and candle.low <= active_stop
            hit_tp2 = tp2 is not None and candle.high >= tp2
            hit_tp1 = tp1 is not None and candle.high >= tp1
        else:
            hit_sl = active_stop is not None and candle.high >= active_stop
            hit_tp2 = tp2 is not None and candle.low <= tp2
            hit_tp1 = tp1 is not None and candle.low <= tp1

        if hit_tp2:
            hit_tp1 = True

        if not tp1_hit:
            if hit_sl and (hit_tp1 or hit_tp2):
                if first_hit_is_tp(candle, direction):
                    tp1_hit = True
                    tp1_time = candle.time_utc
                    if hit_tp2:
                        tp2_hit = True
                        tp2_time = candle.time_utc
                        return build_outcome(
                            "TP2_HIT", entry_fill_time, tp1_time, tp2_time, None
                        )
                    if entry_price is not None and config.ENABLE_BREAK_EVEN:
                        if direction == "BUY" and candle.low <= entry_price:
                            return build_outcome(
                                "TP1_THEN_BE", entry_fill_time, tp1_time, None, candle.time_utc
                            )
                        if direction == "SELL" and candle.high >= entry_price:
                            return build_outcome(
                                "TP1_THEN_BE", entry_fill_time, tp1_time, None, candle.time_utc
                            )
                sl_time = candle.time_utc
                return build_outcome("SL_HIT", entry_fill_time, None, None, sl_time)
            if hit_sl:
                sl_time = candle.time_utc
                return build_outcome("SL_HIT", entry_fill_time, None, None, sl_time)
            if hit_tp2:
                tp1_hit = True
                tp2_hit = True
                tp1_time = candle.time_utc
                tp2_time = candle.time_utc
                return build_outcome("TP2_HIT", entry_fill_time, tp1_time, tp2_time, None)
            if hit_tp1:
                tp1_hit = True
                tp1_time = candle.time_utc
                if config.ENABLE_BREAK_EVEN:
                    active_stop = entry_price
                if entry_price is not None and config.ENABLE_BREAK_EVEN:
                    if direction == "BUY" and candle.low <= entry_price and first_hit_is_tp(
                        candle, direction
                    ):
                        return build_outcome(
                            "TP1_THEN_BE", entry_fill_time, tp1_time, None, candle.time_utc
                        )
                    if direction == "SELL" and candle.high >= entry_price and first_hit_is_tp(
                        candle, direction
                    ):
                        return build_outcome(
                            "TP1_THEN_BE", entry_fill_time, tp1_time, None, candle.time_utc
                        )
                continue
        else:
            if hit_sl and hit_tp2:
                if first_hit_is_tp(candle, direction):
                    tp2_hit = True
                    tp2_time = candle.time_utc
                    return build_outcome(
                        "TP1_THEN_TP2", entry_fill_time, tp1_time, tp2_time, None
                    )
                sl_time = candle.time_utc
                return build_outcome(
                    "TP1_THEN_BE" if config.ENABLE_BREAK_EVEN else "TP1_THEN_SL",
                    entry_fill_time,
                    tp1_time,
                    None,
                    sl_time,
                )
            if hit_sl:
                sl_time = candle.time_utc
                return build_outcome(
                    "TP1_THEN_BE" if config.ENABLE_BREAK_EVEN else "TP1_THEN_SL",
                    entry_fill_time,
                    tp1_time,
                    None,
                    sl_time,
                )
            if hit_tp2:
                tp2_hit = True
                tp2_time = candle.time_utc
                return build_outcome(
                    "TP1_THEN_TP2", entry_fill_time, tp1_time, tp2_time, None
                )

    if tp1_hit:
        return build_outcome("TP1_OPEN", entry_fill_time, tp1_time, None, None)
    return build_outcome("OPEN", entry_fill_time, None, None, None)


def first_hit_is_tp(candle: Candle, direction: str) -> bool:
    bullish = candle.close >= candle.open
    if direction == "BUY":
        return bullish
    return not bullish



def build_outcome(
    outcome: str,
    entry_time: datetime | None,
    tp1_time: datetime | None,
    tp2_time: datetime | None,
    sl_time: datetime | None,
) -> dict:
    return {
        "outcome": outcome,
        "entry_filled_time": entry_time.isoformat() + "Z" if entry_time else None,
        "tp1_time": tp1_time.isoformat() + "Z" if tp1_time else None,
        "tp2_time": tp2_time.isoformat() + "Z" if tp2_time else None,
        "sl_time": sl_time.isoformat() + "Z" if sl_time else None,
    }


def main() -> None:
    args = parse_args()
    start = parse_timestamp(args.start) if args.start else None
    end = parse_timestamp(args.end) if args.end else None
    run_backtest(args.csv, args.source_minutes, args.output, start, end)


if __name__ == "__main__":
    main()
