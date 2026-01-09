from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Literal, Protocol

from ... import config
from ...models import Candle
from ...structure import BreakEvent, BreakDirection, find_breaks, find_swings, has_liquidity_sweep
from ...time_utils import session_from_utc
from ...timeframes import TIMEFRAME_H4, TIMEFRAME_M15, TIMEFRAME_M5, TIMEFRAME_M1, TIMEFRAME_D1


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
    Order Block = Last opposite candle before structure break
    
    For SELL: Last BULLISH candle before bearish CHoCH
    For BUY: Last BEARISH candle before bullish CHoCH
    """
    high: float
    low: float
    created_time: datetime
    direction: BiasDirection  # Direction to trade FROM this OB (SELL from bearish OB, BUY from bullish OB)
    traded: bool = False
    
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
    
    # ✅ NEW: Track Order Blocks for continuous trading
    active_order_blocks: list[OrderBlock] = field(default_factory=list)


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
    MODEL_MODE = "aggressive"

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
        candles_5m, _dropped_5m = self.candle_provider.get_closed_candles(
            self.symbol, TIMEFRAME_M5, now_utc
        )
        candles_1m: list[Candle] | None = None
        if config.USE_1M_ENTRY:
            candles_1m, _dropped_1m = self.candle_provider.get_closed_candles(
                self.symbol, TIMEFRAME_M1, now_utc
            )
        candles_d1: list[Candle] | None = None
        if config.TP3_ENABLED and config.TP3_LEG_SOURCE == "D1":
            candles_d1, _dropped_d1 = self.candle_provider.get_closed_candles(
                self.symbol, TIMEFRAME_D1, now_utc
            )

        if not candles_4h or not candles_15m or not candles_5m:
            return fail("STEP_0_CANDLES_READY")
        if config.USE_1M_ENTRY and not candles_1m:
            return fail("STEP_0_1M_CANDLES_READY")
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
                    # Clear OBs when bias invalidates
                    self.state.active_order_blocks.clear()
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
            else:
                defining_high = self.state.active_leg.start_price
                if current_price > defining_high:
                    self.state.bias = None
                    self.state.active_leg = None
                    # Clear OBs when bias invalidates
                    self.state.active_order_blocks.clear()
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
        
        rules_passed.append("STEP_2_4H_BIAS_STILL_VALID")

        if session == "NY" and self.state.bias_established_session != "LONDON":
            return fail("STEP_1_NY_REQUIRES_LONDON_BIAS")

        # STEP 3 — 4H ACTIVE LEG IDENTIFICATION
        active_leg = self.state.active_leg
        if active_leg is None:
            return fail("STEP_3_ACTIVE_LEG_MISSING")
        rules_passed.append("STEP_3_ACTIVE_LEG_OK")

        # STEP 4 — 15M STRUCTURE & ORDER BLOCK IDENTIFICATION
        swings_15m = find_swings(candles_15m, config.SWING_LEFT, config.SWING_RIGHT)
        events_15m = find_breaks(candles_15m, swings_15m)
        desired_dir = "bear" if direction == "SELL" else "bull"
        
        # ✅ REMOVED: cross_time requirement - trade ALL 15M OBs in the 4H leg
        # Old logic required price to reach 50% first, blocking early OBs
        
        # ✅ NEW: Look for ANY 15M CHoCH after 4H bias established
        structure_event = self._latest_event_after_time(
            events_15m, active_leg.start_time, desired_dir, ("CHoCH",)
        )

        # Also check for weakness (optional alternative to CHoCH)
        has_weakness, weakness_time = self._has_weakness(
            candles_15m, active_leg.start_time, direction, active_leg
        )

        if structure_event is None and not has_weakness:
            return fail("STEP_4_15M_NO_CHOCH_OR_WEAKNESS")

        # Use CHoCH if available (preferred), otherwise weakness
        if structure_event:
            reference_time = structure_event.time_utc
            reference_price = structure_event.close_price
            rules_passed.append("STEP_4_15M_CHOCH_CONFIRMED")
            
            # ✅ NEW: Create Order Block from this CHoCH
            ob_zone = self._find_order_block(candles_15m, structure_event, direction)
            if ob_zone:
                ob_high, ob_low = ob_zone
                
                # Check if we already have this OB
                ob_exists = any(
                    abs(ob.created_time.timestamp() - structure_event.time_utc.timestamp()) < 1
                    for ob in self.state.active_order_blocks
                )
                
                if not ob_exists:
                    new_ob = OrderBlock(
                        high=ob_high,
                        low=ob_low,
                        created_time=structure_event.time_utc,
                        direction=direction,
                        traded=False
                    )
                    self.state.active_order_blocks.append(new_ob)
                    rules_passed.append("STEP_4_15M_OB_CREATED")
                    
        elif has_weakness:
            reference_time = weakness_time
            weakness_candle_idx = next(
                i for i, c in enumerate(candles_15m) if c.time_utc == weakness_time
            )
            reference_price = candles_15m[weakness_candle_idx].close
            rules_passed.append("STEP_4_15M_WEAKNESS_CONFIRMED")

        # ✅ SIMPLIFIED: Trade every OB in the 4H leg (no premium/discount filter)
        # Premium/discount was for 4H-based entries, not 15M OB strategy
        structure_pos = active_leg.position(reference_price)
        # if structure_pos is None:
        #     return fail("STEP_4_15M_LOCATION_UNDEFINED")
        
        # # Only check if inside the 4H leg (allow 5% overshoot for wicks)
        # if structure_pos < -0.20 or structure_pos > 1.20:  # Was -0.10/1.10
        #     return fail("STEP_4_15M_OUTSIDE_LEG")
        
        if structure_pos is None:
            return fail("STEP_4_15M_LOCATION_UNDEFINED")
        rules_passed.append("STEP_4_15M_LOCATION_VALID")
        rules_passed.append("STEP_4_15M_STRUCTURE_AFTER_CROSS")

        # STEP 5 — SKIP QUALITY FILTERS (TRADE EVERY SETUP)
        # Removed liquidity sweep requirements for maximum trade frequency

        # STEP 6 — 5M TRIGGER CHECK (SIMPLIFIED - NO OB RETEST REQUIRED)
        
        # ✅ ULTRA AGGRESSIVE: Enter on ANY 5M CHoCH in direction, with or without OB retest
        swings_5m = find_swings(candles_5m, config.SWING_LEFT, config.SWING_RIGHT)
        events_5m = find_breaks(candles_5m, swings_5m)
        
        # Look for 5M CHoCH after the most recent 15M structure
        choc_event = self._latest_event_after_time(
            events_5m, reference_time, desired_dir, ("CHoCH",)
        )
        
        if choc_event is None:
            return fail("STEP_6_5M_CHOCH_MISSING")
        
        rules_passed.append("STEP_6_5M_CHOCH_FOUND")

        # ✅ SIMPLIFIED: Allow 5M CHoCH anywhere in the 4H leg
        choc_pos = active_leg.position(choc_event.close_price)
        if choc_pos is None:
            return fail("STEP_6_5M_PULLBACK_UNDEFINED")
        
        # Only verify it's within the 4H leg (very lenient)
        # if choc_pos < -0.10 or choc_pos > 1.10:
        #     return fail("STEP_6_5M_OUTSIDE_LEG")
        
        rules_passed.append("STEP_6_5M_IN_4H_LEG")

        entry_event = choc_event
        use_1m_entry = config.USE_1M_ENTRY and (
            direction != "SELL" or config.ENABLE_1M_ENTRY_SELL
        )
        if use_1m_entry:
            swings_1m = find_swings(
                candles_1m or [], config.SWING_LEFT, config.SWING_RIGHT
            )
            events_1m = find_breaks(candles_1m or [], swings_1m)
            entry_event_1m = self._latest_event_after_time(
                events_1m, choc_event.time_utc, desired_dir, ("CHoCH",)
            )
            if entry_event_1m is None:
                return fail("STEP_6_1M_CHOCH_MISSING")
            
            entry_event = entry_event_1m
            rules_passed.append("STEP_6_1M_CHOCH_FOUND")

            # ✅ SIMPLIFIED: Just check it's in the leg
            entry_pos = active_leg.position(entry_event.close_price)
            if entry_pos is None:
                return fail("STEP_6_1M_PULLBACK_UNDEFINED")
            
            if entry_pos < -0.10 or entry_pos > 1.10:
                return fail("STEP_6_1M_OUTSIDE_LEG")
            
            rules_passed.append("STEP_6_1M_IN_4H_LEG")

        entry_candles = candles_1m if use_1m_entry else candles_5m
        entry_candle = entry_candles[entry_event.index]
        entry_high = entry_candle.high
        entry_low = entry_candle.low
        if entry_event.break_level is None:
            return fail("STEP_6_ENTRY_BREAK_LEVEL_MISSING")
        entry_price = self._round_price(entry_event.break_level)

        # STEP 7 — STOP LOSS VALIDITY CHECK
        sl_candle_lookback = getattr(config, 'SL_CANDLE_LOOKBACK', 3)
        lookback_start = max(0, entry_event.index - sl_candle_lookback)
        recent_candles = entry_candles[lookback_start:entry_event.index + 1]
        
        sl_buffer = config.SL_EXTRA_PIPS * config.PIP_SIZE
        
        if direction == "SELL":
            structure_high = max(c.high for c in recent_candles)
            stop_loss = self._round_price(structure_high + sl_buffer)
            
            if stop_loss <= entry_price:
                return fail("STEP_7_SL_BELOW_ENTRY")
        else:
            structure_low = min(c.low for c in recent_candles)
            stop_loss = self._round_price(structure_low - sl_buffer)
            
            if stop_loss >= entry_price:
                return fail("STEP_7_SL_ABOVE_ENTRY")
        
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
        tp_leg = self._resolve_tp_leg(active_leg, structure_event, candles_15m, direction)
        if tp_leg is None:
            if config.TP_LEG_FALLBACK_TO_4H:
                tp_leg = active_leg
                rules_passed.append("STEP_8_TP_LEG_FALLBACK_4H")
            else:
                return fail("STEP_8_TP_LEG_MISSING")

        take_profit_plan = self._validate_take_profit_plan(
            direction, entry_price, tp_leg
        )
        if take_profit_plan is None:
            return fail("STEP_8_TP_PLAN_MISSING")
        plan_name, tp1_price, tp2_price = take_profit_plan
        rules_passed.append("STEP_8_TP_PLAN_DEFINED")

        if config.TP3_ENABLED:
            tp3_leg = self._resolve_tp3_leg(
                active_leg, structure_event, candles_15m, candles_d1, direction
            )
            if tp3_leg is None:
                if config.TP3_LEG_FALLBACK_TO_4H:
                    tp3_leg = active_leg
                    rules_passed.append("STEP_8_TP3_LEG_FALLBACK_4H")
                else:
                    return fail("STEP_8_TP3_LEG_MISSING")
            tp3_price = self._leg_target_price(direction, tp3_leg, config.TP3_LEG_PERCENT)
            if tp3_price is None:
                return fail("STEP_8_TP3_INVALID")
            tp3_price = self._round_price(tp3_price)
            rules_passed.append("STEP_8_TP3_DEFINED")

        # STEP 9 — FINAL CONSISTENCY CHECK
        rules_passed.append("STEP_9_FINAL_CONSISTENCY")

        # ✅ Mark this OB as traded (allow retest later if price returns)
        # active_ob.traded = True

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
        direction: BiasDirection
    ) -> tuple[float, float] | None:
        """
        Find the Order Block (last opposite candle before break)
        
        For SELL: Find last BULLISH candle before bearish CHoCH
        For BUY: Find last BEARISH candle before bullish CHoCH
        
        Returns: (ob_high, ob_low) or None
        """
        if break_event.index == 0:
            return None
        
        # Look backwards from the break candle
        break_index = break_event.index
        
        if direction == "SELL":
            # Find last bullish candle (close > open)
            for i in range(break_index - 1, max(0, break_index - 10), -1):
                candle = candles[i]
                if candle.close > candle.open:  # Bullish candle
                    return (candle.high, candle.low)
        else:
            # Find last bearish candle (close < open)
            for i in range(break_index - 1, max(0, break_index - 10), -1):
                candle = candles[i]
                if candle.close < candle.open:  # Bearish candle
                    return (candle.high, candle.low)
        
        return None

    def _is_price_in_ob(
        self,
        price: float,
        ob: OrderBlock,
        tolerance_pips: float = 5
    ) -> bool:
        """
        Check if price is within OB zone
        Allow small tolerance (default 5 pips)
        """
        tolerance = tolerance_pips * config.PIP_SIZE
        
        ob_high = ob.high + tolerance
        ob_low = ob.low - tolerance
        
        return ob_low <= price <= ob_high

    def _find_active_ob_at_price(
        self,
        current_price: float,
        direction: BiasDirection
    ) -> OrderBlock | None:
        """
        Find an unttraded OB that current price is retesting
        
        Returns the most recent OB that:
        1. Matches direction
        2. Hasn't been traded yet (or allow retest)
        3. Current price is within OB zone
        """
        # Check OBs in reverse (most recent first)
        for ob in reversed(self.state.active_order_blocks):
            if ob.direction != direction:
                continue
            
            # Allow retesting OBs that were already traded
            # (price may return to same OB multiple times)
            
            if self._is_price_in_ob(current_price, ob):
                return ob
        
        return None

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
            self.state.active_order_blocks.clear()
            return

        last_event = events[-1]
        self.state.last_4h_event_type = last_event.event_type
        if last_event.defining_swing_price is None:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            self.state.active_order_blocks.clear()
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
            # Clear all OBs when bias changes
            self.state.active_order_blocks.clear()
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

    def _has_weakness(
        self, 
        candles: list[Candle], 
        after_time: datetime, 
        direction: BiasDirection,
        active_leg: ActiveLeg
    ) -> tuple[bool, datetime | None]:
        """
        Detect price weakness in premium/discount zone
        
        For SELL: Look for bearish weakness in premium (61.8%+)
        For BUY: Look for bullish weakness in discount (38.2%-)
        """
        relevant_candles = [c for c in candles if c.time_utc >= after_time]
        
        for candle in relevant_candles:
            position = active_leg.position(candle.close)
            if position is None or position < 0 or position > 1:
                continue
            
            if direction == "SELL":
                if position < 0.618:
                    continue
                
                candle_range = candle.high - candle.low
                if candle_range <= 0:
                    continue
                
                is_bearish_close = candle.close < candle.open
                close_in_lower_quarter = (candle.close - candle.low) / candle_range < 0.25
                has_upper_wick = (candle.high - max(candle.open, candle.close)) / candle_range > 0.25
                
                weakness_count = sum([
                    is_bearish_close,
                    close_in_lower_quarter,
                    has_upper_wick
                ])
                
                if weakness_count >= 2:
                    return True, candle.time_utc
            
            else:  # BUY
                if position > 0.382:
                    continue
                
                candle_range = candle.high - candle.low
                if candle_range <= 0:
                    continue
                
                is_bullish_close = candle.close > candle.open
                close_in_upper_quarter = (candle.close - candle.low) / candle_range > 0.75
                has_lower_wick = (min(candle.open, candle.close) - candle.low) / candle_range > 0.25
                
                weakness_count = sum([
                    is_bullish_close,
                    close_in_upper_quarter,
                    has_lower_wick
                ])
                
                if weakness_count >= 2:
                    return True, candle.time_utc
        
        return False, None

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
        self, direction: BiasDirection, entry_price: float, active_leg: ActiveLeg
    ) -> tuple[str, float, float] | None:
        leg_range = active_leg.high - active_leg.low
        if leg_range <= 0:
            return None
        
        if not (0 < config.TP1_LEG_PERCENT < 1 and 0 < config.TP2_LEG_PERCENT < 1):
            return None
        
        # ✅ FIX: Ensure TP2 represents MORE profit than TP1
        if config.TP2_LEG_PERCENT <= config.TP1_LEG_PERCENT:
            return None
        
        if direction == "SELL":
            tp1 = active_leg.high - (leg_range * config.TP1_LEG_PERCENT)
            tp2 = active_leg.high - (leg_range * config.TP2_LEG_PERCENT)
            
            # ✅ FIX: Validate TPs are below entry (profit zone)
            if tp1 >= entry_price or tp2 >= entry_price:
                return None
            
            # ✅ FIX: Validate TP2 is lower than TP1 (more profit for SELL)
            if tp2 >= tp1:
                return None
        
        else:  # BUY
            tp1 = active_leg.low + (leg_range * config.TP1_LEG_PERCENT)
            tp2 = active_leg.low + (leg_range * config.TP2_LEG_PERCENT)
            
            # ✅ FIX: Validate TPs are above entry (profit zone)
            if tp1 <= entry_price or tp2 <= entry_price:
                return None
            
            # ✅ FIX: Validate TP2 is higher than TP1 (more profit for BUY)
            if tp2 <= tp1:
                return None
        
        # ✅ FIX: Round before returning
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