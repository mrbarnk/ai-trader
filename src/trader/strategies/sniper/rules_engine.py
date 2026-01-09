from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Literal, Protocol

from ... import config
from ...models import Candle
from ...structure import BreakEvent, BreakDirection, find_breaks, find_swings, has_liquidity_sweep
from ...time_utils import session_from_utc
from ...timeframes import TIMEFRAME_D1, TIMEFRAME_H4, TIMEFRAME_M1, TIMEFRAME_M15


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
class OrderBlock:
    """
    Order Block = Last opposite candle before structure break.
    """

    high: float
    low: float
    created_time: datetime
    direction: BiasDirection

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    @property
    def range(self) -> float:
        return self.high - self.low


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
    model_mode: str
    model_tag: str
    model_magic: int
    session: str | None
    direction: str | None
    entry: str | float
    stop_loss: float | None
    take_profit: str | None
    tp1_price: float | None
    tp2_price: float | None
    tp3_price: float | None
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
    MODEL_MODE = "sniper"

    def __init__(self, symbol: str, candle_provider: CandleProvider | None = None):
        self.symbol = symbol
        self.state = TradingState()
        if candle_provider is None:
            from ...live_data import LiveCandleProvider

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
        tp3_price: float | None = None

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
                tp3_price=None,
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
        candles_1m, _dropped_1m = self.candle_provider.get_closed_candles(
            self.symbol, TIMEFRAME_M1, now_utc
        )
        candles_d1: list[Candle] | None = None
        if config.TP3_ENABLED and config.TP3_LEG_SOURCE == "D1":
            candles_d1, _dropped_d1 = self.candle_provider.get_closed_candles(
                self.symbol, TIMEFRAME_D1, now_utc
            )

        if not candles_4h or not candles_15m or not candles_1m:
            return fail("STEP_0_CANDLES_READY")
        if config.TP3_ENABLED and config.TP3_LEG_SOURCE == "D1" and not candles_d1:
            return fail("STEP_0_D1_CANDLES_READY")
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
                from ...mt5_client import get_spread_pips
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
            return fail("STEP_2_4H_BIAS_NO_TRADE")
        rules_passed.append("STEP_2_4H_BIAS_CLEAR")
        bias_time = self.state.bias_established_time
        if bias_time is None:
            return fail("STEP_2_4H_BIAS_TIME_MISSING")

        # Check if bias is still valid
        if self.state.active_leg:
            current_price = candles_15m[-1].close
            if direction == "BUY":
                defining_low = self.state.active_leg.start_price
                if current_price < defining_low:
                    self.state.bias = None
                    self.state.active_leg = None
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
            else:
                defining_high = self.state.active_leg.start_price
                if current_price > defining_high:
                    self.state.bias = None
                    self.state.active_leg = None
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
        rules_passed.append("STEP_2_4H_BIAS_STILL_VALID")

        if session == "NY" and self.state.bias_established_session != "LONDON":
            return fail("STEP_1_NY_REQUIRES_LONDON_BIAS")

        # STEP 3 — 4H ACTIVE LEG IDENTIFICATION
        active_leg = self.state.active_leg
        if active_leg is None:
            return fail("STEP_3_ACTIVE_LEG_MISSING")
        rules_passed.append("STEP_3_ACTIVE_LEG_OK")

        # STEP 4 — 15M CONFIRMATION & ZONE
        swings_15m = find_swings(candles_15m, config.SWING_LEFT, config.SWING_RIGHT)
        events_15m = find_breaks(candles_15m, swings_15m)
        desired_dir = "bear" if direction == "SELL" else "bull"
        structure_event = self._latest_event_after_time(
            events_15m, bias_time, desired_dir, ("BOS", "CHoCH")
        )
        if structure_event is None:
            return fail("STEP_4_15M_NO_CONFIRMATION")

        structure_pos = active_leg.position(structure_event.close_price)
        if structure_pos is None or structure_pos < -0.1 or structure_pos > 1.1:
            return fail("STEP_4_15M_OUTSIDE_4H_LEG")
        rules_passed.append("STEP_4_15M_STRUCTURE_CONFIRMED")

        ob_zone = self._find_order_block(candles_15m, structure_event, direction)
        zone_high: float | None = None
        zone_low: float | None = None
        zone_rule = "STEP_4_15M_OB_DEFINED"
        if ob_zone is not None:
            zone_high, zone_low = ob_zone
        else:
            fvg_zone = self._find_fvg_zone(candles_15m, structure_event, direction)
            if fvg_zone is None:
                return fail("STEP_4_15M_ZONE_MISSING")
            zone_high, zone_low = fvg_zone
            zone_rule = "STEP_4_15M_FVG_DEFINED"

        zone_ob = OrderBlock(
            high=zone_high,
            low=zone_low,
            created_time=structure_event.time_utc,
            direction=direction,
        )
        rules_passed.append(zone_rule)

        # STEP 5 — PRICE TAPS 15M ZONE
        zone_touch_index = self._find_zone_touch_index(
            candles_1m, structure_event.time_utc, zone_ob
        )
        if zone_touch_index is None:
            return fail("STEP_5_15M_ZONE_NOT_TOUCHED")
        rules_passed.append("STEP_5_15M_ZONE_TOUCHED")

        # STEP 6 — 1M LIQUIDITY SWEEP + BOS
        swings_1m = find_swings(candles_1m, config.SWING_LEFT, config.SWING_RIGHT)
        events_1m = find_breaks(candles_1m, swings_1m)
        entry_event = self._latest_event_with_sweep(
            events_1m,
            candles_1m,
            swings_1m,
            zone_touch_index,
            desired_dir,
        )
        if entry_event is None:
            return fail("STEP_6_1M_SWEEP_BOS_MISSING")
        rules_passed.append("STEP_6_1M_SWEEP_BOS_CONFIRMED")

        entry_pos = active_leg.position(entry_event.close_price)
        if entry_pos is None or entry_pos < -0.1 or entry_pos > 1.1:
            return fail("STEP_6_1M_OUTSIDE_4H_LEG")
        rules_passed.append("STEP_6_1M_IN_4H_LEG")

        if entry_event.break_level is None:
            return fail("STEP_6_ENTRY_BREAK_LEVEL_MISSING")
        entry_price = self._round_price(entry_event.break_level)

        # STEP 7 — STOP LOSS VALIDITY CHECK
        sl_buffer = config.SL_EXTRA_PIPS * config.PIP_SIZE
        stop_loss = self._stop_from_order_block(
            candles_1m, entry_event, direction, entry_price, sl_buffer
        )
        if stop_loss is None:
            stop_loss = self._stop_from_defining_swing(
                candles_1m, entry_event, direction, entry_price, sl_buffer
            )
        if stop_loss is None:
            return fail("STEP_7_STOP_LOSS_MISSING")

        rules_passed.append("STEP_7_SL_VALID")

        choc_range_pips = self._choc_range_pips(candles_1m, entry_event)
        if choc_range_pips is None:
            choc_range_pips = 0.0

        stop_distance_pips = abs(entry_price - stop_loss) / config.PIP_SIZE
        if stop_distance_pips <= 0:
            return fail("STEP_7_STOP_DISTANCE_INVALID")

        if config.ENABLE_RISK_MANAGEMENT:
            account_balance = self._resolve_account_balance()
            if account_balance is None:
                return fail("STEP_7_BALANCE_UNAVAILABLE")
            risk_pct = getattr(
                config, "SNIPER_RISK_PER_TRADE_PCT", config.RISK_PER_TRADE_PCT
            )
            risk_amount = account_balance * (risk_pct / 100)
            position_size_lots = self._position_size_lots(stop_distance_pips, risk_amount)
            if position_size_lots is None:
                return fail("STEP_7_POSITION_SIZE_INVALID")
            rules_passed.append("STEP_7_RISK_SIZING_OK")

        # STEP 8 — TAKE PROFIT PLAN CHECK
        tp_leg = self._resolve_tp_leg(active_leg, structure_event, candles_15m, direction)
        if tp_leg is None and config.TP_LEG_FALLBACK_TO_4H:
            tp_leg = active_leg
            rules_passed.append("STEP_8_TP_LEG_FALLBACK_4H")

        take_profit_plan = (
            self._validate_take_profit_plan(direction, entry_price, tp_leg)
            if tp_leg is not None
            else None
        )
        if take_profit_plan is None:
            take_profit_plan = self._rr_take_profit_plan(direction, entry_price, stop_loss)
            if take_profit_plan is None:
                return fail("STEP_8_TP_PLAN_MISSING")
            rules_passed.append("STEP_8_TP_PLAN_RR")
        else:
            rules_passed.append("STEP_8_TP_PLAN_DEFINED")

        plan_name, tp1_price, tp2_price = take_profit_plan

        if config.TP3_ENABLED:
            tp3_leg = self._resolve_tp3_leg(
                active_leg, structure_event, candles_15m, candles_d1, direction
            )
            if tp3_leg is not None:
                tp3_price = self._leg_target_price(direction, tp3_leg, config.TP3_LEG_PERCENT)
                if tp3_price is not None:
                    tp3_price = self._round_price(tp3_price)
                    rules_passed.append("STEP_8_TP3_DEFINED")
            if tp3_price is None:
                tp3_price = self._rr_target_price(
                    direction,
                    entry_price,
                    stop_loss,
                    getattr(config, "SNIPER_TP3_RR", 10.0),
                )
                if tp3_price is None:
                    return fail("STEP_8_TP3_INVALID")
                tp3_price = self._round_price(tp3_price)
                rules_passed.append("STEP_8_TP3_DEFINED_RR")

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
            tp3_price=tp3_price,
            spread_pips=spread_pips,
            choc_range_pips=choc_range_pips,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _find_order_block(
        self,
        candles: list[Candle],
        break_event: BreakEvent,
        direction: BiasDirection,
    ) -> tuple[float, float] | None:
        if break_event.index == 0:
            return None

        break_index = break_event.index
        if direction == "SELL":
            for i in range(break_index - 1, max(0, break_index - 10), -1):
                candle = candles[i]
                if candle.close > candle.open:
                    return (candle.high, candle.low)
        else:
            for i in range(break_index - 1, max(0, break_index - 10), -1):
                candle = candles[i]
                if candle.close < candle.open:
                    return (candle.high, candle.low)
        return None

    def _find_fvg_zone(
        self,
        candles: list[Candle],
        break_event: BreakEvent,
        direction: BiasDirection,
    ) -> tuple[float, float] | None:
        if break_event.index < 2:
            return None
        start_index = max(2, break_event.index - 10)
        for i in range(break_event.index, start_index - 1, -1):
            prev2 = candles[i - 2]
            current = candles[i]
            if direction == "BUY":
                if prev2.high < current.low:
                    return (current.low, prev2.high)
            else:
                if prev2.low > current.high:
                    return (prev2.low, current.high)
        return None

    def _find_zone_touch_index(
        self, candles: list[Candle], after_time: datetime, ob: OrderBlock
    ) -> int | None:
        for i, candle in enumerate(candles):
            if candle.time_utc < after_time:
                continue
            if self._candle_touches_ob(candle, ob):
                return i
        return None

    def _candle_touches_ob(
        self, candle: Candle, ob: OrderBlock, tolerance_pips: float = 1.0
    ) -> bool:
        tolerance = tolerance_pips * config.PIP_SIZE
        return candle.low <= ob.high + tolerance and candle.high >= ob.low - tolerance

    def _latest_event_with_sweep(
        self,
        events: list[BreakEvent],
        candles: list[Candle],
        swings: list,
        zone_touch_index: int,
        direction: BreakDirection,
    ) -> BreakEvent | None:
        for event in reversed(events):
            if event.index <= zone_touch_index:
                continue
            if event.direction != direction:
                continue
            if event.event_type != "BOS":
                continue
            relevant_swings = [
                swing for swing in swings if zone_touch_index <= swing.index < event.index
            ]
            if not relevant_swings:
                continue
            if has_liquidity_sweep(
                candles,
                relevant_swings,
                event.index,
                direction,
                config.PIP_SIZE,
                config.LIQUIDITY_SWEEP_PIPS,
            ):
                return event
        return None

    def _stop_from_order_block(
        self,
        candles: list[Candle],
        entry_event: BreakEvent,
        direction: BiasDirection,
        entry_price: float,
        sl_buffer: float,
    ) -> float | None:
        ob_zone = self._find_order_block(candles, entry_event, direction)
        if ob_zone is None:
            return None
        ob_high, ob_low = ob_zone
        if direction == "BUY":
            stop_loss = self._round_price(ob_low - sl_buffer)
            if stop_loss >= entry_price:
                return None
            return stop_loss
        stop_loss = self._round_price(ob_high + sl_buffer)
        if stop_loss <= entry_price:
            return None
        return stop_loss

    def _stop_from_defining_swing(
        self,
        candles: list[Candle],
        entry_event: BreakEvent,
        direction: BiasDirection,
        entry_price: float,
        sl_buffer: float,
    ) -> float | None:
        if entry_event.defining_swing_price is None:
            return None
        if direction == "BUY":
            swing_low = entry_event.defining_swing_price
            if (
                entry_event.defining_swing_index is not None
                and 0 <= entry_event.defining_swing_index < len(candles)
            ):
                swing_low = candles[entry_event.defining_swing_index].low
            stop_loss = self._round_price(swing_low - sl_buffer)
            if stop_loss >= entry_price:
                return None
            return stop_loss
        swing_high = entry_event.defining_swing_price
        if (
            entry_event.defining_swing_index is not None
            and 0 <= entry_event.defining_swing_index < len(candles)
        ):
            swing_high = candles[entry_event.defining_swing_index].high
        stop_loss = self._round_price(swing_high + sl_buffer)
        if stop_loss <= entry_price:
            return None
        return stop_loss

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

    def _resolve_tp_leg(
        self,
        active_leg: ActiveLeg,
        structure_event: BreakEvent | None,
        candles_15m: list[Candle],
        direction: BiasDirection,
    ) -> ActiveLeg | None:
        if config.TP_LEG_SOURCE != "15M":
            return active_leg
        if structure_event is None or structure_event.defining_swing_price is None:
            return None
        start_time = (
            candles_15m[structure_event.defining_swing_index].time_utc
            if structure_event.defining_swing_index is not None
            and 0 <= structure_event.defining_swing_index < len(candles_15m)
            else candles_15m[structure_event.index].time_utc
        )
        if direction == "SELL":
            end_price = candles_15m[structure_event.index].low
        else:
            end_price = candles_15m[structure_event.index].high
        return ActiveLeg(
            start_price=structure_event.defining_swing_price,
            end_price=end_price,
            start_time=start_time,
            end_time=structure_event.time_utc,
        )

    def _resolve_tp3_leg(
        self,
        active_leg: ActiveLeg,
        structure_event: BreakEvent | None,
        candles_15m: list[Candle],
        candles_d1: list[Candle] | None,
        direction: BiasDirection,
    ) -> ActiveLeg | None:
        if config.TP3_LEG_SOURCE == "4H":
            return active_leg
        if config.TP3_LEG_SOURCE == "15M":
            return self._resolve_tp_leg(active_leg, structure_event, candles_15m, direction)
        if config.TP3_LEG_SOURCE != "D1":
            return None
        if not candles_d1:
            return None
        swings_d1 = find_swings(candles_d1, config.SWING_LEFT_4H, config.SWING_RIGHT_4H)
        events_d1 = find_breaks(candles_d1, swings_d1)
        desired_dir = "bear" if direction == "SELL" else "bull"
        for event in reversed(events_d1):
            if event.direction != desired_dir:
                continue
            if event.defining_swing_price is None:
                continue
            start_time = (
                candles_d1[event.defining_swing_index].time_utc
                if event.defining_swing_index is not None
                and 0 <= event.defining_swing_index < len(candles_d1)
                else candles_d1[event.index].time_utc
            )
            end_price = (
                candles_d1[event.index].low
                if direction == "SELL"
                else candles_d1[event.index].high
            )
            return ActiveLeg(
                start_price=event.defining_swing_price,
                end_price=end_price,
                start_time=start_time,
                end_time=event.time_utc,
            )
        return None

    def _leg_target_price(
        self, direction: BiasDirection, leg: ActiveLeg, percent: float
    ) -> float | None:
        if percent <= 0 or percent > 1:
            return None
        if direction == "SELL":
            return leg.high - (leg.range * percent)
        return leg.low + (leg.range * percent)

    def _validate_take_profit_plan(
        self,
        direction: BiasDirection,
        entry_price: float,
        tp_leg: ActiveLeg,
    ) -> tuple[str, float, float] | None:
        if not (
            0 < config.TP1_LEG_PERCENT <= 2.0 and 0 < config.TP2_LEG_PERCENT <= 2.0
        ):
            return None

        if direction == "SELL":
            target = tp_leg.end_price
            distance_to_target = entry_price - target
            if distance_to_target <= 0:
                return None
            tp1 = entry_price - (distance_to_target * config.TP1_LEG_PERCENT)
            tp2 = entry_price - (distance_to_target * config.TP2_LEG_PERCENT)
            if tp1 >= entry_price or tp2 >= entry_price:
                return None
            if tp2 >= tp1:
                return None
        else:
            target = tp_leg.end_price
            distance_to_target = target - entry_price
            if distance_to_target <= 0:
                return None
            tp1 = entry_price + (distance_to_target * config.TP1_LEG_PERCENT)
            tp2 = entry_price + (distance_to_target * config.TP2_LEG_PERCENT)
            if tp1 <= entry_price or tp2 <= entry_price:
                return None
            if tp2 <= tp1:
                return None

        return "PLAN_A", self._round_price(tp1), self._round_price(tp2)

    def _rr_take_profit_plan(
        self, direction: BiasDirection, entry_price: float, stop_loss: float
    ) -> tuple[str, float, float] | None:
        risk = abs(entry_price - stop_loss)
        if risk <= 0:
            return None
        rr1 = getattr(config, "SNIPER_TP1_RR", 3.0)
        rr2 = getattr(config, "SNIPER_TP2_RR", 5.0)
        if rr1 <= 0 or rr2 <= 0 or rr2 <= rr1:
            return None
        if direction == "SELL":
            tp1 = entry_price - (risk * rr1)
            tp2 = entry_price - (risk * rr2)
            if tp1 >= entry_price or tp2 >= entry_price:
                return None
        else:
            tp1 = entry_price + (risk * rr1)
            tp2 = entry_price + (risk * rr2)
            if tp1 <= entry_price or tp2 <= entry_price:
                return None
        return "RR", self._round_price(tp1), self._round_price(tp2)

    def _rr_target_price(
        self,
        direction: BiasDirection,
        entry_price: float,
        stop_loss: float,
        rr: float,
    ) -> float | None:
        risk = abs(entry_price - stop_loss)
        if risk <= 0 or rr <= 0:
            return None
        if direction == "SELL":
            return entry_price - (risk * rr)
        return entry_price + (risk * rr)

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
        tp3_price: float | None,
        spread_pips: float | None,
        choc_range_pips: float | None,
        stop_distance_pips: float | None,
        account_balance: float | None,
        risk_amount: float | None,
        position_size_lots: float | None,
        rules_passed: list[str],
        rules_failed: list[str],
    ) -> SignalOutput:
        model_mode, model_tag, model_magic = self._model_meta()
        return SignalOutput(
            decision=decision,
            timestamp_utc=now_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            pair=config.SYMBOL_CANONICAL,
            model_mode=model_mode,
            model_tag=model_tag,
            model_magic=model_magic,
            session=session,
            direction=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=tp3_price,
            spread_pips=spread_pips,
            choc_range_pips=choc_range_pips,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _model_meta(self) -> tuple[str, str, int]:
        mode = self.MODEL_MODE
        tag = config.MODEL_TAGS.get(mode, mode.upper())
        magic = config.MODEL_MAGICS.get(mode, 0)
        return mode, tag, magic

    def _round_price(self, price: float) -> float:
        return round(price, 5)

    def _safe_spread_pips(self) -> float | None:
        if config.ASSUME_ZERO_SPREAD:
            return 0.0
        try:
            from ...mt5_client import get_spread_pips
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
            from ...mt5_client import get_account_balance
        except Exception:
            return None
        try:
            return float(get_account_balance())
        except Exception:
            return None

    def _position_size_lots(
        self, stop_distance_pips: float, risk_amount: float
    ) -> float | None:
        if stop_distance_pips <= 0 or config.PIP_VALUE_PER_LOT <= 0:
            return None
        raw_lots = risk_amount / (stop_distance_pips * config.PIP_VALUE_PER_LOT)
        if raw_lots <= 0:
            return None
        stepped = math.floor(raw_lots / config.LOT_STEP) * config.LOT_STEP
        if stepped < config.MIN_LOT_SIZE or stepped > config.MAX_LOT_SIZE:
            return None
        return round(stepped, 4)
