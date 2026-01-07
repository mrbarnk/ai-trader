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
    """
    MODIFIED: Allow iBOS by tracking ALL unbroken swings
    """
    swings_by_index = {s.index: s for s in swings}
    
    # CHANGED: Track ALL unbroken swings, not just last one
    unbroken_highs: list[SwingPoint] = []  # All highs not yet broken
    unbroken_lows: list[SwingPoint] = []   # All lows not yet broken
    
    last_swing_high_seen: SwingPoint | None = None
    last_swing_low_seen: SwingPoint | None = None
    breaks: list[BreakEvent] = []
    
    # Track trend for CHoCH vs BOS
    current_trend: BreakDirection | None = None
    last_structure_high: float | None = None
    last_structure_low: float | None = None

    for i, candle in enumerate(candles):
        swing = swings_by_index.get(i)
        if swing:
            if swing.kind == "high":
                unbroken_highs.append(swing)  # Add to unbroken list
                last_swing_high_seen = swing
            else:
                unbroken_lows.append(swing)   # Add to unbroken list
                last_swing_low_seen = swing

        # CHANGED: Check if we break ANY unbroken high (bull break)
        broken_highs = [h for h in unbroken_highs if candle.close > h.price]
        if broken_highs:
            # Take the highest one broken (strongest break)
            highest_broken = max(broken_highs, key=lambda h: h.price)
            
            # Determine event type
            event_type: EventType = "BOS"
            if current_trend == "bear":
                event_type = "CHoCH"
                current_trend = "bull"
            elif current_trend == "bull":
                event_type = "BOS"
            else:
                current_trend = "bull"
                event_type = "BOS"
            
            breaks.append(
                BreakEvent(
                    index=i,
                    time_utc=candle.time_utc,
                    direction="bull",
                    event_type=event_type,
                    break_level=highest_broken.price,
                    close_price=candle.close,
                    defining_swing_index=(
                        last_swing_low_seen.index if last_swing_low_seen else None
                    ),
                    defining_swing_price=(
                        last_swing_low_seen.price if last_swing_low_seen else None
                    ),
                )
            )
            
            # Remove all broken highs from unbroken list
            unbroken_highs = [h for h in unbroken_highs if h.price >= highest_broken.price]

        # CHANGED: Check if we break ANY unbroken low (bear break)
        broken_lows = [l for l in unbroken_lows if candle.close < l.price]
        if broken_lows:
            # Take the lowest one broken (strongest break)
            lowest_broken = min(broken_lows, key=lambda l: l.price)
            
            # Determine event type
            event_type: EventType = "BOS"
            if current_trend == "bull":
                event_type = "CHoCH"
                current_trend = "bear"
            elif current_trend == "bear":
                event_type = "BOS"
            else:
                current_trend = "bear"
                event_type = "BOS"
            
            breaks.append(
                BreakEvent(
                    index=i,
                    time_utc=candle.time_utc,
                    direction="bear",
                    event_type=event_type,
                    break_level=lowest_broken.price,
                    close_price=candle.close,
                    defining_swing_index=(
                        last_swing_high_seen.index if last_swing_high_seen else None
                    ),
                    defining_swing_price=(
                        last_swing_high_seen.price if last_swing_high_seen else None
                    ),
                )
            )
            
            # Remove all broken lows from unbroken list
            unbroken_lows = [l for l in unbroken_lows if l.price <= lowest_broken.price]

    return breaks

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
    """
    FIXED: More strict liquidity sweep detection with wick requirement
    
    A true liquidity sweep should:
    1. Go beyond the swing point by min_pips
    2. Close back on the other side of the swing
    3. Have a significant wick showing rejection
    """
    if end_index <= 0:
        return False
    
    if direction == "bear":
        # For bear sweeps: looking for sweep ABOVE high (grab buy-side liquidity)
        swing_highs = [s for s in swings if s.kind == "high" and s.index < end_index]
        if not swing_highs:
            return False
        last_high = swing_highs[-1]
        sweep_level = last_high.price + (pip_size * min_pips)
        
        for candle in candles[last_high.index + 1 : end_index + 1]:
            # Must sweep above the level
            if candle.high >= sweep_level:
                # Must close back below the swing high (rejection)
                if candle.close < last_high.price:
                    # FIXED: Check wick size is significant
                    wick_size = (candle.high - candle.close) / pip_size
                    if wick_size >= min_pips:
                        return True
        return False
    
    # For bull sweeps: looking for sweep BELOW low (grab sell-side liquidity)
    swing_lows = [s for s in swings if s.kind == "low" and s.index < end_index]
    if not swing_lows:
        return False
    last_low = swing_lows[-1]
    sweep_level = last_low.price - (pip_size * min_pips)
    
    for candle in candles[last_low.index + 1 : end_index + 1]:
        # Must sweep below the level
        if candle.low <= sweep_level:
            # Must close back above the swing low (rejection)
            if candle.close > last_low.price:
                # FIXED: Check wick size is significant
                wick_size = (candle.close - candle.low) / pip_size
                if wick_size >= min_pips:
                    return True
    return False