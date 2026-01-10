from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import math
from typing import Literal, Protocol

from ... import config
from ...models import Candle
from ...structure import BreakEvent, BreakDirection, find_breaks, find_swings
from ...time_utils import session_from_utc
from ...timeframes import TIMEFRAME_H4, TIMEFRAME_M15, TIMEFRAME_M5, TIMEFRAME_M1


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
        """Returns position as percentage (0.0 to 1.0) within the leg range"""
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
    last_15m_bos_time: datetime | None = None  # Track latest 15M BOS


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
    fib_level: float | None
    entry_fib_position: float | None
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


class SimplifiedSignalEngine:
    MODEL_MODE = "simplified_bos_choch"

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
        fib_level: float | None = None
        entry_fib_position: float | None = None
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
                tp3_price=None,
                spread_pips=spread_pips,
                fib_level=fib_level,
                entry_fib_position=entry_fib_position,
                stop_distance_pips=stop_distance_pips,
                account_balance=account_balance,
                risk_amount=risk_amount,
                position_size_lots=position_size_lots,
                rules_passed=rules_passed,
                rules_failed=rules_failed,
            )

        session = session_from_utc(now_utc)
        direction = self.state.bias

        # STEP 0 — GET CANDLES
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
        rules_passed.append("STEP_0_CANDLES_READY")

        # STEP 1 — BASIC CHECKS
        if self.symbol not in config.SYMBOL_VARIANTS:
            return fail("STEP_1_SYMBOL_NOT_SUPPORTED")
        if session is None:
            return fail("STEP_1_SESSION_OUTSIDE")
        rules_passed.append("STEP_1_BASIC_CHECKS_OK")

        # Check spread if enabled
        if config.ENABLE_SPREAD_FILTER:
            try:
                from ...mt5_client import get_spread_pips
                spread_pips = get_spread_pips(self.symbol)
                if spread_pips > config.MAX_SPREAD_PIPS:
                    return fail("STEP_1_SPREAD_TOO_WIDE")
                rules_passed.append("STEP_1_SPREAD_OK")
            except Exception:
                return fail("STEP_1_SPREAD_UNAVAILABLE")

        # STEP 2 — 4H BIAS (SIMPLIFIED)
        self._update_4h_bias(candles_4h)
        direction = self.state.bias
        if direction is None:
            return fail("STEP_2_NO_4H_BIAS")
        
        active_leg = self.state.active_leg
        if active_leg is None:
            return fail("STEP_2_NO_ACTIVE_LEG")

        # Check for bias invalidation by 15M closes beyond key levels
        invalidation_result = self._check_bias_invalidation(candles_15m, direction, active_leg)
        if invalidation_result:
            self.state.bias = None
            self.state.active_leg = None
            return fail(f"STEP_2_BIAS_INVALIDATED_{invalidation_result}")
        
        # Basic price validation  
        current_price = candles_15m[-1].close
        if direction == "BUY" and current_price < active_leg.start_price:
            return fail("STEP_2_BIAS_INVALIDATED_PRICE")
        elif direction == "SELL" and current_price > active_leg.start_price:
            return fail("STEP_2_BIAS_INVALIDATED_PRICE")
        
        rules_passed.append("STEP_2_4H_BIAS_VALID")

        # STEP 3 — 15M BOS DETECTION
        swings_15m = find_swings(candles_15m, config.SWING_LEFT, config.SWING_RIGHT)
        events_15m = find_breaks(candles_15m, swings_15m)
        desired_dir = "bear" if direction == "SELL" else "bull"

        # Find latest 15M BOS in the active 4H leg
        latest_15m_bos = self._find_latest_bos_in_leg(
            events_15m, active_leg, desired_dir
        )
        if latest_15m_bos is None:
            return fail("STEP_3_NO_15M_BOS")

        # Update our tracking of 15M BOS
        self.state.last_15m_bos_time = latest_15m_bos.time_utc
        rules_passed.append("STEP_3_15M_BOS_FOUND")

        # STEP 4 — 5M CHoCH FROM FIB LEVEL
        swings_5m = find_swings(candles_5m, config.SWING_LEFT, config.SWING_RIGHT)
        events_5m = find_breaks(candles_5m, swings_5m)

        # Look for 5M CHoCH after the 15M BOS
        choc_5m = self._find_choch_after_time(
            events_5m, latest_15m_bos.time_utc, desired_dir
        )
        if choc_5m is None:
            return fail("STEP_4_NO_5M_CHOCH")

        # Check if 5M CHoCH is from acceptable fib level
        fib_threshold = getattr(config, 'MIN_FIB_LEVEL', 0.5)  # Default 50%
        choch_fib_position = active_leg.position(choc_5m.close_price)
        
        if choch_fib_position is None:
            return fail("STEP_4_CHOCH_OUTSIDE_LEG")

        if direction == "SELL":
            if choch_fib_position < fib_threshold:
                return fail("STEP_4_CHOCH_BELOW_FIB_THRESHOLD")
        else:  # BUY
            if choch_fib_position > (1.0 - fib_threshold):
                return fail("STEP_4_CHOCH_ABOVE_FIB_THRESHOLD")

        fib_level = choch_fib_position
        entry_fib_position = fib_level
        rules_passed.append("STEP_4_5M_CHOCH_FROM_FIB_OK")

        # STEP 5 — OPTIONAL 1M CHoCH (Must be within 5M CHoCH candle range)
        entry_event = choc_5m
        if config.USE_1M_ENTRY and candles_1m:
            swings_1m = find_swings(candles_1m, config.SWING_LEFT, config.SWING_RIGHT)
            events_1m = find_breaks(candles_1m, swings_1m)
            
            # Get 5M CHoCH candle range
            choc_5m_candle = candles_5m[choc_5m.index]
            choc_5m_high = choc_5m_candle.high
            choc_5m_low = choc_5m_candle.low
            
            # Find 1M CHoCH within 5M candle range
            choc_1m = self._find_choch_within_range(
                events_1m, choc_5m.time_utc, desired_dir, 
                choc_5m_high, choc_5m_low, candles_1m
            )
            if choc_1m is None:
                return fail("STEP_5_NO_1M_CHOCH_IN_5M_RANGE")
                
            entry_event = choc_1m
            rules_passed.append("STEP_5_1M_CHOCH_WITHIN_5M_RANGE")

        # STEP 6 — ENTRY VALIDATION
        entry_candles = candles_1m if config.USE_1M_ENTRY else candles_5m
        entry_candle = entry_candles[entry_event.index]
        
        if entry_event.break_level is None:
            return fail("STEP_6_ENTRY_LEVEL_MISSING")
            
        # Validate strong body break
        if not self._body_breaks_level(entry_candle, desired_dir, entry_event.break_level):
            return fail("STEP_6_WEAK_ENTRY_SIGNAL")

        entry_price = self._round_price(entry_candle.close)
        rules_passed.append("STEP_6_ENTRY_VALIDATED")

        # STEP 7 — STOP LOSS
        sl_buffer = config.SL_EXTRA_PIPS * config.PIP_SIZE
        stop_loss = self._calculate_stop_loss(
            entry_candles, entry_event, direction, entry_price, sl_buffer
        )
        if stop_loss is None:
            return fail("STEP_7_INVALID_STOP_LOSS")

        stop_distance_pips = abs(entry_price - stop_loss) / config.PIP_SIZE
        if stop_distance_pips <= 0:
            return fail("STEP_7_INVALID_STOP_DISTANCE")
        rules_passed.append("STEP_7_STOP_LOSS_OK")

        # STEP 8 — TAKE PROFIT (Target end of active leg)
        if direction == "SELL":
            tp1_price = self._round_price(active_leg.low)
            if tp1_price >= entry_price:
                return fail("STEP_8_INVALID_TP_TARGET")
        else:
            tp1_price = self._round_price(active_leg.high)
            if tp1_price <= entry_price:
                return fail("STEP_8_INVALID_TP_TARGET")

        tp2_price = tp1_price  # Simple target for now
        rules_passed.append("STEP_8_TAKE_PROFIT_OK")

        # STEP 9 — RISK MANAGEMENT
        if config.ENABLE_RISK_MANAGEMENT:
            account_balance = self._resolve_account_balance()
            if account_balance is None:
                return fail("STEP_9_BALANCE_UNAVAILABLE")
            risk_amount = account_balance * (config.RISK_PER_TRADE_PCT / 100)
            position_size_lots = self._position_size_lots(stop_distance_pips, risk_amount)
            if position_size_lots is None:
                return fail("STEP_9_INVALID_POSITION_SIZE")
            rules_passed.append("STEP_9_RISK_MANAGEMENT_OK")

        return self._build_output(
            decision="TRADE",
            now_utc=now_utc,
            session=session,
            direction=direction,
            entry=entry_price,
            stop_loss=stop_loss,
            take_profit="LEG_TARGET",
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp3_price=None,
            spread_pips=spread_pips,
            fib_level=fib_level,
            entry_fib_position=entry_fib_position,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _update_4h_bias(self, candles_4h: list[Candle]) -> None:
        """Simplified 4H bias detection - just look for last CHoCH"""
        if not candles_4h:
            return
            
        last_closed = candles_4h[-1].time_utc
        if self.state.last_4h_close_time == last_closed:
            return

        swings = find_swings(candles_4h, config.SWING_LEFT_4H, config.SWING_RIGHT_4H)
        events = find_breaks(candles_4h, swings)
        
        if not events:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        # Find the last valid CHoCH
        last_choch = None
        for event in reversed(events):
            if event.event_type == "CHoCH" and event.defining_swing_price is not None:
                last_choch = event
                break

        if last_choch is None:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        # Check if bias is still valid (no invalidation)
        if last_choch.direction == "bull":
            invalidated = any(
                candle.close < last_choch.defining_swing_price
                for candle in candles_4h[last_choch.index + 1:]
            )
        else:
            invalidated = any(
                candle.close > last_choch.defining_swing_price
                for candle in candles_4h[last_choch.index + 1:]
            )

        if invalidated:
            self.state.bias = None
            self.state.active_leg = None
            self.state.last_4h_close_time = last_closed
            return

        # Set bias and active leg
        direction: BiasDirection = "BUY" if last_choch.direction == "bull" else "SELL"
        leg_start_price = last_choch.defining_swing_price
        leg_start_time = (
            candles_4h[last_choch.defining_swing_index].time_utc
            if last_choch.defining_swing_index is not None
            else candles_4h[last_choch.index].time_utc
        )
        
        if last_choch.direction == "bull":
            leg_end_price = candles_4h[last_choch.index].high
        else:
            leg_end_price = candles_4h[last_choch.index].low

        self.state.bias = direction
        self.state.bias_established_time = candles_4h[last_choch.index].time_utc
        self.state.bias_established_session = session_from_utc(self.state.bias_established_time)
        self.state.active_leg = ActiveLeg(
            start_price=leg_start_price,
            end_price=leg_end_price,
            start_time=leg_start_time,
            end_time=candles_4h[last_choch.index].time_utc,
        )
        self.state.last_4h_close_time = last_closed

    def _find_latest_bos_in_leg(
        self,
        events: list[BreakEvent],
        active_leg: ActiveLeg,
        direction: BreakDirection
    ) -> BreakEvent | None:
        """Find the latest BOS in the current active leg"""
        for event in reversed(events):
            if event.time_utc < active_leg.start_time:
                continue
            if event.direction != direction:
                continue
            if event.event_type != "BOS":
                continue
            return event
        return None

    def _find_choch_within_range(
        self,
        events: list[BreakEvent],
        after_time: datetime,
        direction: BreakDirection,
        range_high: float,
        range_low: float,
        candles: list[Candle]
    ) -> BreakEvent | None:
        """Find CHoCH after time that occurs within specified price range"""
        for event in reversed(events):
            if event.time_utc <= after_time:
                continue
            if event.direction != direction:
                continue
            if event.event_type != "CHoCH":
                continue
                
            # Check if CHoCH occurs within the 5M candle range
            event_candle = candles[event.index]
            if (event_candle.low >= range_low and event_candle.high <= range_high):
                return event
        return None

    def _check_bias_invalidation(
        self, 
        candles_15m: list[Candle], 
        direction: BiasDirection,
        active_leg: ActiveLeg
    ) -> str | None:
        """
        Check if 15M closes beyond key levels invalidating bias
        
        For SELL: If 15M closes above key resistance during pullback
        For BUY: If 15M closes below key support during pullback
        """
        if not candles_15m or not active_leg:
            return None
            
        # Look at recent 15M closes (last 10 candles for invalidation check)
        recent_candles = candles_15m[-10:] if len(candles_15m) >= 10 else candles_15m
        
        if direction == "SELL":
            # For SELL bias, invalidated if 15M closes above the leg start (key resistance)
            invalidation_level = active_leg.start_price
            for candle in recent_candles:
                if candle.close > invalidation_level:
                    return "15M_CLOSE_ABOVE_RESISTANCE"
        else:  # BUY
            # For BUY bias, invalidated if 15M closes below the leg start (key support)  
            invalidation_level = active_leg.start_price
            for candle in recent_candles:
                if candle.close < invalidation_level:
                    return "15M_CLOSE_BELOW_SUPPORT"
                    
        return None

    def _find_choch_after_time(
        self,
        events: list[BreakEvent],
        after_time: datetime,
        direction: BreakDirection
    ) -> BreakEvent | None:
        """
        Find the latest CHoCH after specified time that breaks actual swing levels
        
        CHoCH must:
        1. Break a previous swing (high for bullish, low for bearish)
        2. Have a valid defining swing price
        3. Occur after the specified time
        """
        for event in reversed(events):
            if event.time_utc <= after_time:
                continue
            if event.direction != direction:
                continue
            if event.event_type != "CHoCH":
                continue
            
            # Ensure CHoCH has valid swing level it's breaking
            if event.defining_swing_price is None:
                continue
                
            # Ensure it's a meaningful break (not just any level)
            if event.defining_swing_index is None:
                continue
                
            return event
        return None

    def _body_breaks_level(
        self, candle: Candle, direction: BreakDirection, level: float
    ) -> bool:
        """Check if candle body strongly breaks the level"""
        if direction == "bull":
            return candle.open > level and candle.close > level
        return candle.open < level and candle.close < level

    def _calculate_stop_loss(
        self,
        candles: list[Candle],
        entry_event: BreakEvent,
        direction: BiasDirection,
        entry_price: float,
        sl_buffer: float,
    ) -> float | None:
        """Calculate stop loss based on the broken swing"""
        if entry_event.defining_swing_price is None:
            return None

        if direction == "SELL":
            stop_loss = self._round_price(entry_event.defining_swing_price + sl_buffer)
            if stop_loss <= entry_price:
                return None
            return stop_loss
        else:
            stop_loss = self._round_price(entry_event.defining_swing_price - sl_buffer)
            if stop_loss >= entry_price:
                return None
            return stop_loss

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
        fib_level: float | None,
        entry_fib_position: float | None,
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
            fib_level=fib_level,
            entry_fib_position=entry_fib_position,
            stop_distance_pips=stop_distance_pips,
            account_balance=account_balance,
            risk_amount=risk_amount,
            position_size_lots=position_size_lots,
            rules_passed=rules_passed,
            rules_failed=rules_failed,
        )

    def _model_meta(self) -> tuple[str, str, int]:
        mode = self.MODEL_MODE
        tag = config.MODEL_TAGS.get(mode, "SIMPLIFIED")
        magic = config.MODEL_MAGICS.get(mode, 9999)
        return mode, tag, magic

    def _round_price(self, price: float) -> float:
        return round(price, 5)

    def _resolve_account_balance(self) -> float | None:
        if config.ACCOUNT_BALANCE_OVERRIDE is not None:
            return float(config.ACCOUNT_BALANCE_OVERRIDE)
        try:
            from ...mt5_client import get_account_balance
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


# Keep compatibility with engine_factory imports.
SignalEngine = SimplifiedSignalEngine
