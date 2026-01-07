from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Literal, Protocol

from . import config
from .models import Candle
from .structure import BreakEvent, BreakDirection, find_breaks, find_swings, has_liquidity_sweep
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
    last_4h_event_type: str | None = None


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
    spread_pips: float | None
    choc_range_pips: float | None
    stop_distance_pips: float | None
    account_balance: float | None
    risk_amount: float | None
    position_size_lots: float | None
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
        spread_pips: float | None = None
        choc_range_pips: float | None = None
        stop_distance_pips: float | None = None
        account_balance: float | None = None
        risk_amount: float | None = None
        position_size_lots: float | None = None

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
                spread_pips=spread_pips,
                choc_range_pips=choc_range_pips,
                stop_distance_pips=stop_distance_pips,
                account_balance=account_balance,
                risk_amount=risk_amount,
                position_size_lots=position_size_lots,
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

        if config.ENABLE_SPREAD_FILTER:
            try:
                from .mt5_client import get_spread_pips
            except Exception:
                return fail("STEP_1_SPREAD_UNAVAILABLE")
            spread_pips = get_spread_pips(self.symbol)
            if spread_pips > config.MAX_SPREAD_PIPS:
                return fail("STEP_1_SPREAD_TOO_WIDE")
            rules_passed.append("STEP_1_SPREAD_OK")
        else:
            spread_pips = self._safe_spread_pips()

        # STEP 2 — 4H DIRECTION CHECK (BIAS)
        self._update_4h_bias(candles_4h)
        direction = self.state.bias
        if direction is None:
            if self.state.last_4h_event_type == "CHoCH":
                return fail("STEP_2_4H_LAST_EVENT_CHOCH")
            return fail("STEP_2_4H_BIAS_NO_TRADE")
        rules_passed.append("STEP_2_4H_BIAS_CLEAR")
        bias_time = self.state.bias_established_time
        if bias_time is None:
            return fail("STEP_2_4H_BIAS_TIME_MISSING")

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
        cross_level = (
            config.PREMIUM_CROSS_LEVEL
            if direction == "SELL"
            else config.DISCOUNT_CROSS_LEVEL
        )
        cross_time = self._first_cross_time(
            candles_15m, active_leg, cross_level, direction
        )
        if cross_time is None:
            if direction == "SELL":
                return fail("STEP_4_15M_PREMIUM_CROSS_MISSING")
            return fail("STEP_4_15M_DISCOUNT_CROSS_MISSING")
        if cross_time < bias_time:
            return fail("STEP_4_15M_CROSS_BEFORE_4H_BOS")
        if direction == "SELL":
            rules_passed.append("STEP_4_15M_PREMIUM_CROSS_OK")
        else:
            rules_passed.append("STEP_4_15M_DISCOUNT_CROSS_OK")

        structure_event = self._latest_event_after_time(
            events_15m, cross_time, desired_dir, ("CHoCH",)
        )
        if structure_event is None:
            return fail("STEP_4_15M_CHOCH_AFTER_CROSS_MISSING")
        if structure_event.time_utc < bias_time:
            return fail("STEP_4_15M_CHOCH_BEFORE_4H_BOS")

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
        rules_passed.append("STEP_4_15M_CHOCH_AFTER_CROSS")

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
        if direction == "SELL" and config.REQUIRE_LIQUIDITY_SWEEP_SELL and not swept:
            return fail("STEP_5_LIQUIDITY_SWEEP_REQUIRED")
        if config.REQUIRE_NO_LIQUIDITY_SWEEP and swept:
            return fail("STEP_5_LIQUIDITY_SWEEP_EXCLUDED")

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

        require_5m_premium = config.REQUIRE_5M_CHOCH_PREMIUM or (
            direction == "SELL" and config.REQUIRE_5M_CHOCH_PREMIUM_SELL
        )
        if direction == "SELL":
            if choc_pos < 0.382:
                return fail("STEP_6_5M_OUTSIDE_PULLBACK")
            if require_5m_premium and choc_pos < 0.5:
                return fail("STEP_6_5M_PREMIUM_REQUIRED")
        else:
            if choc_pos > 0.618:
                return fail("STEP_6_5M_OUTSIDE_PULLBACK")
            if require_5m_premium and choc_pos > 0.5:
                return fail("STEP_6_5M_PREMIUM_REQUIRED")
        rules_passed.append("STEP_6_5M_IN_PULLBACK")
        if require_5m_premium:
            rules_passed.append("STEP_6_5M_PREMIUM_OK")

        if not self._choc_after_pullback(events_5m, choc_event):
            return fail("STEP_6_5M_AFTER_PULLBACK")
        rules_passed.append("STEP_6_5M_AFTER_PULLBACK")

        entry_event = choc_event
        use_1m_entry = config.USE_1M_ENTRY and (
            direction != "SELL" or config.ENABLE_1M_ENTRY_SELL
        )
        if use_1m_entry:
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

            if config.ENABLE_CHOCH_RANGE_FILTER:
                if not self._choc_range_ok(candles_1m or [], entry_event):
                    return fail("STEP_6_1M_CHOCH_RANGE_TOO_SMALL")
                rules_passed.append("STEP_6_1M_CHOCH_RANGE_OK")

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

        if config.ENABLE_CHOCH_RANGE_FILTER:
            if not self._choc_range_ok(candles_5m, choc_event):
                return fail("STEP_6_5M_CHOCH_RANGE_TOO_SMALL")
            rules_passed.append("STEP_6_5M_CHOCH_RANGE_OK")

        entry_candles = candles_1m if use_1m_entry else candles_5m
        entry_candle = entry_candles[entry_event.index]
        entry_high = entry_candle.high
        entry_low = entry_candle.low
        if entry_event.break_level is None:
            return fail("STEP_6_ENTRY_BREAK_LEVEL_MISSING")
        entry_price = self._round_price(entry_event.break_level)

        # STEP 7 — STOP LOSS VALIDITY CHECK
        choch_high = entry_high
        choch_low = entry_low
        sl_buffer = config.SL_EXTRA_PIPS * config.PIP_SIZE
        if direction == "SELL":
            stop_loss = self._round_price(choch_high + sl_buffer)
        else:
            stop_loss = self._round_price(choch_low - sl_buffer)
        rules_passed.append("STEP_7_SL_VALID")

        choc_range_pips = self._choc_range_pips(entry_candles, entry_event)
        if choc_range_pips is None:
            choc_range_pips = 0.0

        stop_distance_pips = abs(entry_price - stop_loss) / config.PIP_SIZE
        if stop_distance_pips <= 0:
            return fail("STEP_7_STOP_DISTANCE_INVALID")

        if config.ENABLE_RISK_MANAGEMENT:
            account_balance = self._resolve_account_balance()
            if account_balance is None:
                return fail("STEP_7_BALANCE_UNAVAILABLE")
            risk_amount = account_balance * (config.RISK_PER_TRADE_PCT / 100)
            position_size_lots = self._position_size_lots(stop_distance_pips, risk_amount)
            if position_size_lots is None:
                return fail("STEP_7_POSITION_SIZE_INVALID")
            rules_passed.append("STEP_7_RISK_SIZING_OK")

        # STEP 8 — TAKE PROFIT PLAN CHECK
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
            spread_pips=spread_pips,
            choc_range_pips=choc_range_pips,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _update_4h_bias(self, candles_4h: list[Candle]) -> None:
        if not candles_4h:
            return
        last_closed = candles_4h[-1].time_utc
        if self.state.last_4h_close_time == last_closed:
            return

        swings = find_swings(
            candles_4h, config.SWING_LEFT_4H, config.SWING_RIGHT_4H
        )
        events = find_breaks(candles_4h, swings)
        if not events:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_event_type = None
            self.state.last_4h_close_time = last_closed
            return

        last_event = events[-1]
        self.state.last_4h_event_type = last_event.event_type
        if last_event.defining_swing_price is None:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return
        if last_event.event_type == "CHoCH":
            self.state.bias = None
            self.state.active_leg = None
            self.state.bias_established_time = None
            self.state.bias_established_session = None
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

    def _choc_range_pips(
        self, candles: list[Candle], choc_event: BreakEvent
    ) -> float | None:
        if choc_event.index >= len(candles):
            return None
        candle = candles[choc_event.index]
        return (candle.high - candle.low) / config.PIP_SIZE

    def _choc_range_ok(self, candles: list[Candle], choc_event: BreakEvent) -> bool:
        range_pips = self._choc_range_pips(candles, choc_event)
        if range_pips is None:
            return False
        return range_pips >= config.MIN_CHOCH_RANGE_PIPS

    def _first_cross_time(
        self,
        candles: list[Candle],
        active_leg: ActiveLeg,
        level: float,
        direction: BiasDirection,
    ) -> datetime | None:
        for candle in candles:
            if candle.time_utc < active_leg.start_time:
                continue
            pos = active_leg.position(candle.close)
            if pos is None or pos < 0 or pos > 1:
                continue
            if direction == "SELL":
                if pos >= level:
                    return candle.time_utc
            else:
                if pos <= level:
                    return candle.time_utc
        return None

    def _validate_take_profit_plan(
        self, direction: BiasDirection, entry_price: float, active_leg: ActiveLeg
    ) -> tuple[str, float, float] | None:
        leg_range = active_leg.high - active_leg.low
        if leg_range <= 0:
            return None
        if direction == "SELL":
            tp1 = active_leg.high - (leg_range * 0.5)
            tp2 = active_leg.high - (leg_range * 0.9)
            if not (entry_price > tp1 > tp2):
                return None
        else:
            tp1 = active_leg.low + (leg_range * 0.5)
            tp2 = active_leg.low + (leg_range * 0.9)
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
        spread_pips: float | None,
        choc_range_pips: float | None,
        stop_distance_pips: float | None,
        account_balance: float | None,
        risk_amount: float | None,
        position_size_lots: float | None,
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
            spread_pips=spread_pips,
            choc_range_pips=choc_range_pips,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _round_price(self, price: float) -> float:
        return round(price, 5)

    def _safe_spread_pips(self) -> float | None:
        if config.ASSUME_ZERO_SPREAD:
            return 0.0
        try:
            from .mt5_client import get_spread_pips
        except Exception:
            return None
        try:
            return get_spread_pips(self.symbol)
        except Exception:
            return None

    def _resolve_account_balance(self) -> float | None:
        if config.ACCOUNT_BALANCE_OVERRIDE is not None:
            return float(config.ACCOUNT_BALANCE_OVERRIDE)
        try:
            from .mt5_client import get_account_balance
        except Exception:
            return None
        try:
            return float(get_account_balance())
        except Exception:
            return None

    def _position_size_lots(self, stop_distance_pips: float, risk_amount: float) -> float | None:
        if stop_distance_pips <= 0 or config.PIP_VALUE_PER_LOT <= 0:
            return None
        raw_lots = risk_amount / (stop_distance_pips * config.PIP_VALUE_PER_LOT)
        if raw_lots <= 0:
            return None
        stepped = math.floor(raw_lots / config.LOT_STEP) * config.LOT_STEP
        if stepped < config.MIN_LOT_SIZE or stepped > config.MAX_LOT_SIZE:
            return None
        return round(stepped, 4)
