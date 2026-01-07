from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from . import config
from .models import Candle
from .structure import (
    BreakEvent,
    BreakDirection,
    find_breaks,
    find_swings,
    has_liquidity_sweep,
    latest_event,
)
from .time_utils import session_from_utc
from .timeframes import TIMEFRAME_H4, TIMEFRAME_M15, TIMEFRAME_M5, TIMEFRAME_M1


BiasDirection = Literal["BUY", "SELL"]


@dataclass
class ActiveLeg:
    start_price: float
    end_price: float
    start_time: datetime
    end_time: datetime

    @property
    def low(self) -> float:
        return min(self.start_price, self.end_price)

    @property
    def high(self) -> float:
        return max(self.start_price, self.end_price)

    @property
    def range(self) -> float:
        return self.high - self.low

    def position(self, price: float) -> float | None:
        if self.range == 0:
            return None
        return (price - self.low) / self.range


@dataclass
class TradingState:
    bias: BiasDirection | None = None
    bias_established_time: datetime | None = None
    bias_established_session: str | None = None
    last_4h_close_time: datetime | None = None
    active_leg: ActiveLeg | None = None


@dataclass
class SignalOutput:
    decision: str
    timestamp_utc: str
    pair: str
    session: str | None
    direction: str | None
    entry: str | float
    stop_loss: float | None
    take_profit: str | None
    tp1_price: float | None
    tp2_price: float | None
    rules_passed: list[str]
    rules_failed: list[str]


class CandleProvider(Protocol):
    def get_closed_candles(
        self, symbol: str, timeframe: int, now_utc: datetime
    ) -> tuple[list[Candle], bool]:
        ...


class SignalEngine:
    def __init__(self, symbol: str, candle_provider: CandleProvider | None = None):
        self.symbol = symbol
        self.state = TradingState()
        if candle_provider is None:
            from .live_data import LiveCandleProvider

            candle_provider = LiveCandleProvider()
        self.candle_provider = candle_provider

    def evaluate(self, now_utc: datetime) -> SignalOutput:
        rules_passed: list[str] = []
        rules_failed: list[str] = []

        def fail(rule_id: str) -> SignalOutput:
            rules_failed.append(rule_id)
            return self._build_output(
                decision="NO_TRADE",
                now_utc=now_utc,
                session=session,
                direction=direction,
                entry="n/a",
                stop_loss=None,
                take_profit=None,
                tp1_price=None,
                tp2_price=None,
                rules_passed=rules_passed,
                rules_failed=rules_failed,
            )

        session = session_from_utc(now_utc)
        direction = self.state.bias

        # STEP 0 — TIMEFRAME LOCK
        candles_4h, _dropped_4h = self.candle_provider.get_closed_candles(
            self.symbol, TIMEFRAME_H4, now_utc
        )
        candles_15m, _dropped_15m = self.candle_provider.get_closed_candles(
            self.symbol, TIMEFRAME_M15, now_utc
        )
        candles_5m, _dropped_5m = self.candle_provider.get_closed_candles(
            self.symbol, TIMEFRAME_M5, now_utc
        )
        candles_1m: list[Candle] | None = None
        if config.USE_1M_ENTRY:
            candles_1m, _dropped_1m = self.candle_provider.get_closed_candles(
                self.symbol, TIMEFRAME_M1, now_utc
            )

        if not candles_4h or not candles_15m or not candles_5m:
            return fail("STEP_0_CANDLES_READY")
        if config.USE_1M_ENTRY and not candles_1m:
            return fail("STEP_0_1M_CANDLES_READY")
        rules_passed.append("STEP_0_COMPLETED_CANDLES_ONLY")

        # STEP 1 — ASSET & SESSION CHECK
        if self.symbol not in config.SYMBOL_VARIANTS:
            return fail("STEP_1_SYMBOL_NOT_GU")
        rules_passed.append("STEP_1_SYMBOL_OK")
        rules_passed.append("STEP_1_BROKER_UTC_OK")

        if session is None:
            return fail("STEP_1_SESSION_OUTSIDE")
        rules_passed.append("STEP_1_SESSION_OK")

        if session == "NY":
            rules_passed.append("STEP_1_NY_CONTINUATION_ONLY")

        # STEP 2 — 4H DIRECTION CHECK (BIAS)
        self._update_4h_bias(candles_4h)
        direction = self.state.bias
        if direction is None:
            return fail("STEP_2_4H_BIAS_NO_TRADE")
        rules_passed.append("STEP_2_4H_BIAS_CLEAR")

        if session == "NY" and self.state.bias_established_session != "LONDON":
            return fail("STEP_1_NY_REQUIRES_LONDON_BIAS")

        # STEP 3 — 4H ACTIVE LEG IDENTIFICATION
        active_leg = self.state.active_leg
        if active_leg is None:
            return fail("STEP_3_ACTIVE_LEG_MISSING")
        rules_passed.append("STEP_3_ACTIVE_LEG_OK")

        # STEP 4 — 15M LOCATION CHECK
        swings_15m = find_swings(candles_15m, config.SWING_LEFT, config.SWING_RIGHT)
        events_15m = find_breaks(candles_15m, swings_15m)
        desired_dir = "bear" if direction == "SELL" else "bull"
        structure_event = latest_event(
            events_15m, direction=desired_dir, event_types=("CHoCH",)
        )
        if structure_event is None:
            return fail("STEP_4_15M_CHOCH_MISSING")

        structure_pos = active_leg.position(structure_event.close_price)
        if structure_pos is None:
            return fail("STEP_4_15M_LOCATION_UNDEFINED")
        if structure_pos < 0 or structure_pos > 1:
            return fail("STEP_4_15M_OUTSIDE_LEG")

        if direction == "SELL":
            if 0.382 <= structure_pos < 0.5:
                return fail("STEP_4_15M_MID_RANGE")
            if structure_pos < 0.5:
                return fail("STEP_4_15M_LOCATION_INVALID")
        else:
            if 0.5 < structure_pos <= 0.618:
                return fail("STEP_4_15M_MID_RANGE")
            if structure_pos > 0.5:
                return fail("STEP_4_15M_LOCATION_INVALID")
        rules_passed.append("STEP_4_15M_LOCATION_VALID")
        rules_passed.append("STEP_4_15M_CHOCH_CONFIRMED")

        # STEP 5 — 15M QUALITY BOOST (NOT REQUIRED)
        if direction == "SELL" and structure_pos >= 0.7:
            rules_passed.append("STEP_5_15M_STRONG_ZONE")
        if direction == "BUY" and structure_pos <= 0.3:
            rules_passed.append("STEP_5_15M_STRONG_ZONE")

        swept = has_liquidity_sweep(
            candles=candles_15m,
            swings=swings_15m,
            end_index=structure_event.index,
            direction=desired_dir,
            pip_size=config.PIP_SIZE,
            min_pips=config.LIQUIDITY_SWEEP_PIPS,
        )
        if swept:
            rules_passed.append("STEP_5_LIQUIDITY_SWEEP")

        # STEP 6 — 5M TRIGGER CHECK (CHoCH)
        swings_5m = find_swings(candles_5m, config.SWING_LEFT, config.SWING_RIGHT)
        events_5m = find_breaks(candles_5m, swings_5m)
        choc_event = self._latest_event_after_time(
            events_5m, structure_event.time_utc, desired_dir, ("CHoCH",)
        )
        if choc_event is None:
            return fail("STEP_6_5M_CHOCH_MISSING")
        rules_passed.append("STEP_6_5M_CHOCH_FOUND")

        choc_pos = active_leg.position(choc_event.close_price)
        if choc_pos is None:
            return fail("STEP_6_5M_PULLBACK_UNDEFINED")
        if choc_pos < 0 or choc_pos > 1:
            return fail("STEP_6_5M_OUTSIDE_LEG")

        if direction == "SELL":
            if choc_pos < 0.382:
                return fail("STEP_6_5M_OUTSIDE_PULLBACK")
            if config.REQUIRE_5M_CHOCH_PREMIUM and choc_pos < 0.5:
                return fail("STEP_6_5M_PREMIUM_REQUIRED")
        else:
            if choc_pos > 0.618:
                return fail("STEP_6_5M_OUTSIDE_PULLBACK")
            if config.REQUIRE_5M_CHOCH_PREMIUM and choc_pos > 0.5:
                return fail("STEP_6_5M_PREMIUM_REQUIRED")
        rules_passed.append("STEP_6_5M_IN_PULLBACK")
        if config.REQUIRE_5M_CHOCH_PREMIUM:
            rules_passed.append("STEP_6_5M_PREMIUM_OK")

        if not self._choc_after_pullback(events_5m, choc_event):
            return fail("STEP_6_5M_AFTER_PULLBACK")
        rules_passed.append("STEP_6_5M_AFTER_PULLBACK")

        entry_event = choc_event
        if config.USE_1M_ENTRY:
            swings_1m = find_swings(
                candles_1m or [], config.SWING_LEFT, config.SWING_RIGHT
            )
            events_1m = find_breaks(candles_1m or [], swings_1m)
            entry_event = self._latest_event_after_time(
                events_1m, choc_event.time_utc, desired_dir, ("CHoCH",)
            )
            if entry_event is None:
                return fail("STEP_6_1M_CHOCH_MISSING")
            rules_passed.append("STEP_6_1M_CHOCH_FOUND")

            entry_pos = active_leg.position(entry_event.close_price)
            if entry_pos is None:
                return fail("STEP_6_1M_PULLBACK_UNDEFINED")
            if entry_pos < 0 or entry_pos > 1:
                return fail("STEP_6_1M_OUTSIDE_LEG")
            if direction == "SELL":
                if entry_pos < 0.382:
                    return fail("STEP_6_1M_OUTSIDE_PULLBACK")
                if config.REQUIRE_1M_CHOCH_PREMIUM and entry_pos < 0.5:
                    return fail("STEP_6_1M_PREMIUM_REQUIRED")
            else:
                if entry_pos > 0.618:
                    return fail("STEP_6_1M_OUTSIDE_PULLBACK")
                if config.REQUIRE_1M_CHOCH_PREMIUM and entry_pos > 0.5:
                    return fail("STEP_6_1M_PREMIUM_REQUIRED")
            rules_passed.append("STEP_6_1M_IN_PULLBACK")
            if config.REQUIRE_1M_CHOCH_PREMIUM:
                rules_passed.append("STEP_6_1M_PREMIUM_OK")

            if not self._choc_after_pullback(events_1m, entry_event):
                return fail("STEP_6_1M_AFTER_PULLBACK")
            rules_passed.append("STEP_6_1M_AFTER_PULLBACK")

        # STEP 7 — STOP LOSS VALIDITY CHECK
        if direction == "SELL":
            stop_loss = self._round_price(
                self._entry_high(candles_5m, candles_1m, entry_event) + config.PIP_SIZE
            )
        else:
            stop_loss = self._round_price(
                self._entry_low(candles_5m, candles_1m, entry_event) - config.PIP_SIZE
            )
        rules_passed.append("STEP_7_SL_VALID")

        # STEP 8 — TAKE PROFIT PLAN CHECK
        entry_price = self._round_price(entry_event.close_price)
        take_profit_plan = self._validate_take_profit_plan(
            direction, entry_price, active_leg
        )
        if take_profit_plan is None:
            return fail("STEP_8_TP_PLAN_MISSING")
        plan_name, tp1_price, tp2_price = take_profit_plan
        rules_passed.append("STEP_8_TP_PLAN_DEFINED")

        # STEP 9 — FINAL CONSISTENCY CHECK
        rules_passed.append("STEP_9_FINAL_CONSISTENCY")

        return self._build_output(
            decision="TRADE",
            now_utc=now_utc,
            session=session,
            direction=direction,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit=plan_name,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _update_4h_bias(self, candles_4h: list[Candle]) -> None:
        if not candles_4h:
            return
        last_closed = candles_4h[-1].time_utc
        if self.state.last_4h_close_time == last_closed:
            return

        swings = find_swings(candles_4h, config.SWING_LEFT, config.SWING_RIGHT)
        events = find_breaks(candles_4h, swings)
        if not events:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        last_event = events[-1]
        if last_event.defining_swing_price is None:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        if last_event.direction == "bull":
            invalidated = any(
                candle.close < last_event.defining_swing_price
                for candle in candles_4h[last_event.index + 1 :]
            )
        else:
            invalidated = any(
                candle.close > last_event.defining_swing_price
                for candle in candles_4h[last_event.index + 1 :]
            )

        if invalidated:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        direction: BiasDirection = "BUY" if last_event.direction == "bull" else "SELL"
        leg_start_price = last_event.defining_swing_price
        leg_start_time = (
            candles_4h[last_event.defining_swing_index].time_utc
            if last_event.defining_swing_index is not None
            else candles_4h[last_event.index].time_utc
        )
        if last_event.direction == "bull":
            leg_end_price = candles_4h[last_event.index].high
        else:
            leg_end_price = candles_4h[last_event.index].low

        self.state.bias = direction
        self.state.bias_established_time = candles_4h[last_event.index].time_utc
        self.state.bias_established_session = session_from_utc(
            self.state.bias_established_time
        )
        self.state.active_leg = ActiveLeg(
            start_price=leg_start_price,
            end_price=leg_end_price,
            start_time=leg_start_time,
            end_time=candles_4h[last_event.index].time_utc,
        )
        self.state.last_4h_close_time = last_closed

    def _choc_after_pullback(self, events: list[BreakEvent], choc_event: BreakEvent) -> bool:
        for event in events:
            if event.index == choc_event.index and event.time_utc == choc_event.time_utc:
                break
            if event.direction != choc_event.direction:
                return True
        return False

    def _latest_event_after_time(
        self,
        events: list[BreakEvent],
        after_time: datetime,
        direction: BreakDirection,
        event_types: tuple[str, ...],
    ) -> BreakEvent | None:
        for event in reversed(events):
            if event.time_utc < after_time:
                continue
            if event.direction != direction:
                continue
            if event.event_type not in event_types:
                continue
            return event
        return None

    def _entry_high(
        self,
        candles_5m: list[Candle],
        candles_1m: list[Candle] | None,
        entry_event: BreakEvent,
    ) -> float:
        if candles_1m:
            return candles_1m[entry_event.index].high
        return candles_5m[entry_event.index].high

    def _entry_low(
        self,
        candles_5m: list[Candle],
        candles_1m: list[Candle] | None,
        entry_event: BreakEvent,
    ) -> float:
        if candles_1m:
            return candles_1m[entry_event.index].low
        return candles_5m[entry_event.index].low

    def _validate_take_profit_plan(
        self, direction: BiasDirection, entry_price: float, active_leg: ActiveLeg
    ) -> tuple[str, float, float] | None:
        leg_mid = (active_leg.high + active_leg.low) / 2
        if direction == "SELL":
            tp1 = leg_mid
            tp2 = active_leg.low
            if not (entry_price > tp1 > tp2):
                return None
        else:
            tp1 = leg_mid
            tp2 = active_leg.high
            if not (entry_price < tp1 < tp2):
                return None
        return "PLAN_A", self._round_price(tp1), self._round_price(tp2)

    def _build_output(
        self,
        decision: str,
        now_utc: datetime,
        session: str | None,
        direction: str | None,
        entry: str | float,
        stop_loss: float | None,
        take_profit: str | None,
        tp1_price: float | None,
        tp2_price: float | None,
        rules_passed: list[str],
        rules_failed: list[str],
    ) -> SignalOutput:
        return SignalOutput(
            decision=decision,
            timestamp_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            pair=config.SYMBOL_CANONICAL,
            session=session,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _round_price(self, price: float) -> float:
        return round(price, 5)
