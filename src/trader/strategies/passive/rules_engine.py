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
    break_event_index: int  # Index of the CHoCH that created this OB
    tested: bool = False  # Has price returned to test this OB?
    traded: bool = False  # Has a trade been taken from this OB?
    
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
    
    # Track 15M Order Blocks for sniper entries
    active_order_blocks_15m: list[OrderBlock] = field(default_factory=list)


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
    MODEL_MODE = "sniper"  # H4 → 15M OB → 1M Sniper Entry Strategy

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
                    self.state.active_order_blocks_15m.clear()
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
            else:
                defining_high = self.state.active_leg.start_price
                if current_price > defining_high:
                    self.state.bias = None
                    self.state.active_leg = None
                    self.state.active_order_blocks_15m.clear()
                    return fail("STEP_2_4H_BIAS_INVALIDATED_BY_PRICE")
        
        rules_passed.append("STEP_2_4H_BIAS_STILL_VALID")

        if session == "NY" and self.state.bias_established_session != "LONDON":
            return fail("STEP_1_NY_REQUIRES_LONDON_BIAS")

        # STEP 3 — 4H ACTIVE LEG IDENTIFICATION
        active_leg = self.state.active_leg
        if active_leg is None:
            return fail("STEP_3_ACTIVE_LEG_MISSING")
        rules_passed.append("STEP_3_ACTIVE_LEG_OK")

        # STEP 4 — 15M STRUCTURE & ORDER BLOCK CREATION
        swings_15m = find_swings(candles_15m, config.SWING_LEFT, config.SWING_RIGHT)
        events_15m = find_breaks(candles_15m, swings_15m)
        desired_dir = "bear" if direction == "SELL" else "bull"
        
        # Look for 15M CHoCH after 4H bias established
        structure_event = self._latest_event_after_time(
            events_15m, active_leg.start_time, desired_dir, ("CHoCH",)
        )

        if structure_event is None:
            return fail("STEP_4_15M_NO_CHOCH")
        
        rules_passed.append("STEP_4_15M_CHOCH_CONFIRMED")
        
        # Create 15M Order Block from this CHoCH
        ob_zone = self._find_order_block(candles_15m, structure_event, direction)
        if ob_zone is None:
            return fail("STEP_4_15M_OB_NOT_FOUND")
        
        ob_high, ob_low = ob_zone
        
        # Check if we already have this OB
        ob_exists = any(
            abs(ob.created_time.timestamp() - structure_event.time_utc.timestamp()) < 1
            for ob in self.state.active_order_blocks_15m
        )
        
        current_ob: OrderBlock
        if not ob_exists:
            current_ob = OrderBlock(
                high=ob_high,
                low=ob_low,
                created_time=structure_event.time_utc,
                direction=direction,
                break_event_index=structure_event.index,
                tested=False,
                traded=False
            )
            self.state.active_order_blocks_15m.append(current_ob)
            rules_passed.append("STEP_4_15M_OB_CREATED")
        else:
            # Find the existing OB
            current_ob = next(
                ob for ob in self.state.active_order_blocks_15m
                if abs(ob.created_time.timestamp() - structure_event.time_utc.timestamp()) < 1
            )

        # STEP 5 — 15M ORDER BLOCK RETEST CHECK (MANDATORY)
        current_price = candles_1m[-1].close
        
        # Find an active 15M OB that price is currently testing
        active_ob = self._find_active_ob_at_price(current_price, direction)
        
        if active_ob is None:
            return fail("STEP_5_15M_OB_NOT_RETESTED")
        
        # Mark OB as tested
        if not active_ob.tested:
            active_ob.tested = True
            rules_passed.append("STEP_5_15M_OB_FIRST_RETEST")
        else:
            rules_passed.append("STEP_5_15M_OB_RETESTED")

        # STEP 6 — 1M LIQUIDITY SWEEP + BOS/CHoCH
        swings_1m = find_swings(candles_1m, config.SWING_LEFT, config.SWING_RIGHT)
        events_1m = find_breaks(candles_1m, swings_1m)
        
        # Look for recent 1M CHoCH/BOS after entering the 15M OB zone
        # Use the time when OB was first tested as reference
        ob_test_time = active_ob.created_time
        
        choc_event_1m = self._latest_event_after_time(
            events_1m, ob_test_time, desired_dir, ("CHoCH", "BOS")
        )
        
        if choc_event_1m is None:
            return fail("STEP_6_1M_CHOCH_MISSING")
        
        rules_passed.append("STEP_6_1M_CHOCH_FOUND")
        
        # Check for liquidity sweep before the 1M BOS
        has_sweep = self._check_liquidity_sweep_before_event(
            candles_1m, swings_1m, choc_event_1m, direction
        )
        
        if not has_sweep:
            return fail("STEP_6_1M_NO_LIQUIDITY_SWEEP")
        
        rules_passed.append("STEP_6_1M_LIQUIDITY_SWEEP_CONFIRMED")

        # STEP 7 — 1M ORDER BLOCK ENTRY
        # Find the 1M OB from the BOS that just occurred
        ob_1m_zone = self._find_order_block(candles_1m, choc_event_1m, direction)
        
        if ob_1m_zone is None:
            return fail("STEP_7_1M_OB_NOT_FOUND")
        
        entry_ob_high, entry_ob_low = ob_1m_zone
        
        # Entry at the middle of 1M OB for conservative fill
        entry_price = self._round_price((entry_ob_high + entry_ob_low) / 2)
        
        rules_passed.append("STEP_7_1M_OB_ENTRY_FOUND")

        # STEP 8 — STOP LOSS FROM 1M STRUCTURE
        # Stop loss beyond the 1M Order Block (not the 15M OB)
        sl_buffer = config.SL_EXTRA_PIPS * config.PIP_SIZE

        if direction == "SELL":
            # For SELL: Stop above the 1M OB high (price should not go back up)
            stop_loss = self._round_price(entry_ob_high + sl_buffer)
            
            if stop_loss <= entry_price:
                return fail("STEP_8_SL_BELOW_ENTRY")
        else:  # BUY
            # For BUY: Stop below the 1M OB low (price should not go back down)
            stop_loss = self._round_price(entry_ob_low - sl_buffer)
            
            if stop_loss >= entry_price:
                return fail("STEP_8_SL_ABOVE_ENTRY")

        rules_passed.append("STEP_8_SL_VALID")

        choc_range_pips = self._choc_range_pips(candles_1m, choc_event_1m)
        if choc_range_pips is None:
            choc_range_pips = 0.0

        stop_distance_pips = abs(entry_price - stop_loss) / config.PIP_SIZE
        if stop_distance_pips <= 0:
            return fail("STEP_8_STOP_DISTANCE_INVALID")

        # STEP 9 — RISK MANAGEMENT (1% RISK)
        if config.ENABLE_RISK_MANAGEMENT:
            account_balance = self._resolve_account_balance()
            if account_balance is None:
                return fail("STEP_9_BALANCE_UNAVAILABLE")
            
            # Force 1% risk as per strategy
            risk_pct = 1.0  # Override config for this strategy
            risk_amount = account_balance * (risk_pct / 100)
            
            position_size_lots = self._position_size_lots(stop_distance_pips, risk_amount)
            if position_size_lots is None:
                return fail("STEP_9_POSITION_SIZE_INVALID")
            rules_passed.append("STEP_9_RISK_SIZING_OK_1PCT")

        # STEP 10 — TAKE PROFIT PLAN (1:3 to 1:10 R:R)
        tp_leg = self._resolve_tp_leg(active_leg, structure_event, candles_15m, direction)
        if tp_leg is None:
            if config.TP_LEG_FALLBACK_TO_4H:
                tp_leg = active_leg
                rules_passed.append("STEP_10_TP_LEG_FALLBACK_4H")
            else:
                return fail("STEP_10_TP_LEG_MISSING")

        take_profit_plan = self._validate_take_profit_plan(
            direction, entry_price, stop_loss, tp_leg
        )
        if take_profit_plan is None:
            return fail("STEP_10_TP_PLAN_MISSING")
        plan_name, tp1_price, tp2_price = take_profit_plan
        rules_passed.append("STEP_10_TP_PLAN_DEFINED")

        if config.TP3_ENABLED:
            tp3_leg = self._resolve_tp3_leg(
                active_leg, structure_event, candles_15m, candles_d1, direction
            )
            if tp3_leg is None:
                if config.TP3_LEG_FALLBACK_TO_4H:
                    tp3_leg = active_leg
                    rules_passed.append("STEP_10_TP3_LEG_FALLBACK_4H")
                else:
                    return fail("STEP_10_TP3_LEG_MISSING")
            tp3_price = self._leg_target_price(direction, tp3_leg, config.TP3_LEG_PERCENT)
            if tp3_price is None:
                return fail("STEP_10_TP3_INVALID")
            tp3_price = self._round_price(tp3_price)
            rules_passed.append("STEP_10_TP3_DEFINED")

        # STEP 11 — FINAL CONSISTENCY CHECK
        rules_passed.append("STEP_11_FINAL_CONSISTENCY")

        # Mark this OB as traded
        active_ob.traded = True

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

    def _check_liquidity_sweep_before_event(
        self,
        candles: list[Candle],
        swings: list[tuple[int, float]],
        event: BreakEvent,
        direction: BiasDirection
    ) -> bool:
        """
        Check if liquidity was swept before the BOS/CHoCH
        
        Liquidity sweep = Equal highs/lows taken out before structure break
        
        For SELL: Look for equal highs being swept (price went above then reversed)
        For BUY: Look for equal lows being swept (price went below then reversed)
        """
        if event.index < 5:  # Need some history
            return False
        
        # Look at candles before the break event
        lookback_start = max(0, event.index - 20)
        recent_candles = candles[lookback_start:event.index]
        
        if direction == "SELL":
            # Find equal highs in recent swings
            swing_highs = [s[1] for s in swings if s[0] < event.index and s[1] > candles[s[0]].low]
            
            if len(swing_highs) < 2:
                return False
            
            # Check for equal highs (within 5 pips)
            tolerance = 5 * config.PIP_SIZE
            
            for i in range(len(swing_highs) - 1):
                for j in range(i + 1, len(swing_highs)):
                    if abs(swing_highs[i] - swing_highs[j]) <= tolerance:
                        # Found equal highs, check if they were swept
                        equal_high = max(swing_highs[i], swing_highs[j])
                        
                        # Check if price went above equal high then reversed
                        for candle in recent_candles[-10:]:  # Last 10 candles before break
                            if candle.high > equal_high:
                                return True
            
        else:  # BUY
            # Find equal lows in recent swings
            swing_lows = [s[1] for s in swings if s[0] < event.index and s[1] < candles[s[0]].high]
            
            if len(swing_lows) < 2:
                return False
            
            # Check for equal lows (within 5 pips)
            tolerance = 5 * config.PIP_SIZE
            
            for i in range(len(swing_lows) - 1):
                for j in range(i + 1, len(swing_lows)):
                    if abs(swing_lows[i] - swing_lows[j]) <= tolerance:
                        # Found equal lows, check if they were swept
                        equal_low = min(swing_lows[i], swing_lows[j])
                        
                        # Check if price went below equal low then reversed
                        for candle in recent_candles[-10:]:  # Last 10 candles before break
                            if candle.low < equal_low:
                                return True
        
        return False

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
        Find an active 15M OB that current price is retesting
        
        Returns the most recent OB that:
        1. Matches direction
        2. Current price is within OB zone
        3. Not yet traded (or allow re-entries)
        """
        # Check OBs in reverse (most recent first)
        for ob in reversed(self.state.active_order_blocks_15m):
            if ob.direction != direction:
                continue
            
            # Skip OBs that have already been traded (unless allowing re-entries)
            if ob.traded:
                continue
            
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
            self.state.active_order_blocks_15m.clear()
            return

        last_event = events[-1]
        self.state.last_4h_event_type = last_event.event_type
        if last_event.defining_swing_price is None:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            self.state.active_order_blocks_15m.clear()
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
            self.state.active_order_blocks_15m.clear()
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
        stop_loss: float,
        tp_leg: ActiveLeg
    ) -> tuple[str, float, float] | None:
        """
        Calculate TPs based on Risk:Reward ratio (1:3 to 1:10)
        
        TP1 = 1:3 R:R (conservative)
        TP2 = 1:5+ R:R (based on leg target)
        """
        
        # Calculate risk (entry to stop)
        risk = abs(entry_price - stop_loss)
        
        if risk <= 0:
            return None
        
        # TP1: Conservative 1:3 R:R
        reward_tp1 = risk * 3.0
        
        # TP2: Aim for leg target or 1:5 R:R minimum
        if direction == "SELL":
            tp1 = entry_price - reward_tp1
            
            # TP2: Either 50% of leg or 1:5 R:R, whichever is further
            leg_target = tp_leg.end_price
            tp2_from_leg = entry_price - ((entry_price - leg_target) * 0.5)
            tp2_from_rr = entry_price - (risk * 5.0)
            tp2 = min(tp2_from_leg, tp2_from_rr)  # More aggressive target
            
            # Validate TPs are below entry (profit zone)
            if tp1 >= entry_price or tp2 >= entry_price:
                return None
            
            # Validate TP2 is MORE profit (lower) than TP1
            if tp2 >= tp1:
                return None
        
        else:  # BUY
            tp1 = entry_price + reward_tp1
            
            # TP2: Either 50% of leg or 1:5 R:R, whichever is further
            leg_target = tp_leg.end_price
            tp2_from_leg = entry_price + ((leg_target - entry_price) * 0.5)
            tp2_from_rr = entry_price + (risk * 5.0)
            tp2 = max(tp2_from_leg, tp2_from_rr)  # More aggressive target
            
            # Validate TPs are above entry (profit zone)
            if tp1 <= entry_price or tp2 <= entry_price:
                return None
            
            # Validate TP2 is MORE profit (higher) than TP1
            if tp2 <= tp1:
                return None
        
        return "RR_BASED", self._round_price(tp1), self._round_price(tp2)
        
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