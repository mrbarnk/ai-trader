from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from .models import Candle

SwingKind = Literal["high", "low"]
BreakDirection = Literal["bull", "bear"]
EventType = Literal["BOS", "CHoCH"]


@dataclass(frozen=True)
class SwingPoint:
    index: int
    time_utc: datetime
    price: float
    kind: SwingKind


@dataclass(frozen=True)
class BreakEvent:
    index: int
    time_utc: datetime
    direction: BreakDirection
    event_type: EventType
    break_level: float
    close_price: float
    defining_swing_index: int | None
    defining_swing_price: float | None


def find_swings(candles: list[Candle], left: int, right: int) -> list[SwingPoint]:
    swings: list[SwingPoint] = []
    if len(candles) < left + right + 1:
        return swings
    for i in range(left, len(candles) - right):
        high = candles[i].high
        low = candles[i].low
        is_high = all(high > candles[i - j].high for j in range(1, left + 1)) and all(
            high > candles[i + j].high for j in range(1, right + 1)
        )
        is_low = all(low < candles[i - j].low for j in range(1, left + 1)) and all(
            low < candles[i + j].low for j in range(1, right + 1)
        )
        if is_high:
            swings.append(SwingPoint(i, candles[i].time_utc, high, "high"))
        if is_low:
            swings.append(SwingPoint(i, candles[i].time_utc, low, "low"))
    return swings


def find_breaks(candles: list[Candle], swings: list[SwingPoint]) -> list[BreakEvent]:
    swings_by_index = {s.index: s for s in swings}
    last_swing_high: SwingPoint | None = None
    last_swing_low: SwingPoint | None = None
    last_swing_high_seen: SwingPoint | None = None
    last_swing_low_seen: SwingPoint | None = None
    breaks: list[BreakEvent] = []

    for i, candle in enumerate(candles):
        swing = swings_by_index.get(i)
        if swing:
            if swing.kind == "high":
                last_swing_high = swing
                last_swing_high_seen = swing
            else:
                last_swing_low = swing
                last_swing_low_seen = swing

        if last_swing_high and candle.close > last_swing_high.price:
            breaks.append(
                BreakEvent(
                    index=i,
                    time_utc=candle.time_utc,
                    direction="bull",
                    event_type="BOS",
                    break_level=last_swing_high.price,
                    close_price=candle.close,
                    defining_swing_index=(
                        last_swing_low_seen.index if last_swing_low_seen else None
                    ),
                    defining_swing_price=(
                        last_swing_low_seen.price if last_swing_low_seen else None
                    ),
                )
            )
            last_swing_high = None

        if last_swing_low and candle.close < last_swing_low.price:
            breaks.append(
                BreakEvent(
                    index=i,
                    time_utc=candle.time_utc,
                    direction="bear",
                    event_type="BOS",
                    break_level=last_swing_low.price,
                    close_price=candle.close,
                    defining_swing_index=(
                        last_swing_high_seen.index if last_swing_high_seen else None
                    ),
                    defining_swing_price=(
                        last_swing_high_seen.price if last_swing_high_seen else None
                    ),
                )
            )
            last_swing_low = None

    if not breaks:
        return breaks

    events: list[BreakEvent] = []
    prev_direction: BreakDirection | None = None
    for event in breaks:
        event_type: EventType = "BOS"
        if prev_direction and event.direction != prev_direction:
            event_type = "CHoCH"
        events.append(
            BreakEvent(
                index=event.index,
                time_utc=event.time_utc,
                direction=event.direction,
                event_type=event_type,
                break_level=event.break_level,
                close_price=event.close_price,
                defining_swing_index=event.defining_swing_index,
                defining_swing_price=event.defining_swing_price,
            )
        )
        prev_direction = event.direction
    return events


def latest_event(
    events: list[BreakEvent],
    direction: BreakDirection | None = None,
    event_types: tuple[EventType, ...] = ("BOS", "CHoCH"),
) -> BreakEvent | None:
    for event in reversed(events):
        if event.event_type not in event_types:
            continue
        if direction and event.direction != direction:
            continue
        return event
    return None


def has_liquidity_sweep(
    candles: list[Candle],
    swings: list[SwingPoint],
    end_index: int,
    direction: BreakDirection,
    pip_size: float,
    min_pips: int,
) -> bool:
    if end_index <= 0:
        return False
    if direction == "bear":
        swing_highs = [s for s in swings if s.kind == "high" and s.index < end_index]
        if not swing_highs:
            return False
        last_high = swing_highs[-1]
        sweep_level = last_high.price + (pip_size * min_pips)
        for candle in candles[last_high.index + 1 : end_index + 1]:
            if candle.high >= sweep_level and candle.close < last_high.price:
                return True
        return False

    swing_lows = [s for s in swings if s.kind == "low" and s.index < end_index]
    if not swing_lows:
        return False
    last_low = swing_lows[-1]
    sweep_level = last_low.price - (pip_size * min_pips)
    for candle in candles[last_low.index + 1 : end_index + 1]:
        if candle.low <= sweep_level and candle.close > last_low.price:
            return True
    return False
