from __future__ import annotations

import argparse
import csv
import json
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
    TIMEFRAME_D1,
    TIMEFRAME_SECONDS,
)


from datetime import date
from collections import defaultdict


class DailyRiskManager:
    """
    FIXED: Manages daily risk limits for trading with real-time balance tracking
    
    Features:
    - Daily drawdown % limit (primary protection)
    - Consecutive loss limit (early warning)
    - Max daily losses limit (secondary safety)
    - Daily profit target (lock gains)
    - Real-time balance updates
    """
    
    def __init__(self, starting_balance: float):
        self.starting_balance = starting_balance
        self.daily_stats = defaultdict(lambda: {
            'starting_balance': starting_balance,
            'current_balance': starting_balance,
            'total_losses': 0,
            'consecutive_losses': 0,
            'total_wins': 0,
            'total_trades': 0,
            'buy_trades': 0,
            'sell_trades': 0,
            'daily_pnl': 0.0,
            'daily_r': 0.0,
            'stopped': False,
            'stop_reason': None,
            'stop_time': None
        })
    
    def reset_if_new_day(self, current_date: date, current_balance: float):
        """
        FIXED: Initialize stats for a new trading day with CURRENT balance
        """
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = {
                'starting_balance': current_balance,  # ✅ Use actual current balance
                'current_balance': current_balance,
                'total_losses': 0,
                'consecutive_losses': 0,
                'total_wins': 0,
                'total_trades': 0,
                'buy_trades': 0,
                'sell_trades': 0,
                'daily_pnl': 0.0,
                'daily_r': 0.0,
                'stopped': False,
                'stop_reason': None,
                'stop_time': None
            }
    
    def should_stop_trading(
        self, 
        current_date: date, 
        current_time,
        current_balance: float  # ✅ Pass actual balance
    ) -> tuple[bool, str | None]:
        """
        FIXED: Check all daily limit conditions using REAL-TIME balance
        
        Returns:
            (should_stop, reason): Boolean and optional stop reason
        """
        from . import config
        
        if not config.ENABLE_DAILY_RISK_MANAGEMENT:
            return False, None
        
        stats = self.daily_stats[current_date]
        
        # Already stopped?
        if stats['stopped']:
            return True, stats['stop_reason']
        
        # ✅ FIXED: Use actual current_balance passed in, not stale stats
        daily_pnl = current_balance - stats['starting_balance']
        daily_pnl_pct = (daily_pnl / stats['starting_balance']) * 100
        
        limits_hit = []
        
        # Check 1: Daily Drawdown % (MOST IMPORTANT) ⭐
        if config.ENABLE_DAILY_DRAWDOWN_LIMIT:
            if daily_pnl_pct <= -config.MAX_DAILY_DRAWDOWN_PERCENT:
                loss_amount = abs(daily_pnl)
                limits_hit.append(
                    ("DAILY_DRAWDOWN_LIMIT", 
                     f"-{config.MAX_DAILY_DRAWDOWN_PERCENT}% (${loss_amount:.2f})")
                )
        
        # Check 2: Consecutive Losses (Early Warning)
        if config.ENABLE_CONSECUTIVE_LOSS_LIMIT:
            if stats['consecutive_losses'] >= config.MAX_CONSECUTIVE_LOSSES:
                limits_hit.append(
                    ("CONSECUTIVE_LOSS_LIMIT", 
                     f"{config.MAX_CONSECUTIVE_LOSSES} losses in a row")
                )
        
        # Check 3: Max Daily Losses (Secondary Safety)
        if config.ENABLE_MAX_LOSSES_LIMIT:
            if stats['total_losses'] >= config.MAX_DAILY_LOSSES:
                limits_hit.append(
                    ("MAX_DAILY_LOSSES", 
                     f"{config.MAX_DAILY_LOSSES} total losses")
                )
        
        # Check 4: Daily Profit Target (Lock Gains)
        if config.ENABLE_DAILY_PROFIT_TARGET:
            if daily_pnl_pct >= config.DAILY_PROFIT_TARGET_PERCENT:
                profit_amount = daily_pnl
                limits_hit.append(
                    ("PROFIT_TARGET_HIT", 
                     f"+{config.DAILY_PROFIT_TARGET_PERCENT}% (${profit_amount:.2f})")
                )
        
        # If any limit hit
        if limits_hit:
            reason, detail = limits_hit[0]  # Use first limit hit
            
            stats['stopped'] = True
            stats['stop_reason'] = f"{reason}: {detail}"
            stats['stop_time'] = current_time
            
            # ✅ Update final balance
            stats['current_balance'] = current_balance
            
            # Log if enabled
            if config.LOG_DAILY_LIMITS:
                self._log_daily_stop(current_date, current_time, stats)
            
            return True, stats['stop_reason']
        
        return False, None
    
    def record_trade_result(
        self, 
        current_date: date, 
        pnl: float,
        r_result: float,
        current_balance: float,  # ✅ ADDED: Pass actual balance
        direction: str,
    ):
        """
        FIXED: Update daily statistics after trade closes with real balance
        
        Args:
            current_date: Trading date
            pnl: Dollar profit/loss
            r_result: R-multiple result
            current_balance: Actual account balance after trade
        """
        stats = self.daily_stats[current_date]
        
        stats['total_trades'] += 1
        stats['daily_pnl'] += pnl
        stats['daily_r'] += r_result
        stats['current_balance'] = current_balance  # ✅ FIXED: Update to actual balance
        if direction == "BUY":
            stats['buy_trades'] += 1
        elif direction == "SELL":
            stats['sell_trades'] += 1
        
        if r_result < -0.1:  # Loss (small buffer for BE with tiny loss)
            stats['total_losses'] += 1
            stats['consecutive_losses'] += 1
        else:  # Win or BE
            stats['total_wins'] += 1
            stats['consecutive_losses'] = 0  # Reset consecutive counter
    
    def _log_daily_stop(self, current_date: date, stop_time, stats: dict):
        """Log when daily limit is hit"""
        from . import config
        
        log_entry = {
            'date': str(current_date),
            'stop_time': str(stop_time),
            'reason': stats['stop_reason'],
            'trades': stats['total_trades'],
            'wins': stats['total_wins'],
            'losses': stats['total_losses'],
            'buy_trades': stats['buy_trades'],
            'sell_trades': stats['sell_trades'],
            'consecutive_losses': stats['consecutive_losses'],
            'daily_pnl': round(stats['daily_pnl'], 2),
            'daily_r': round(stats['daily_r'], 2),
            'starting_balance': stats['starting_balance'],
            'ending_balance': round(stats['current_balance'], 2),
            'drawdown_pct': round(
                (stats['current_balance'] - stats['starting_balance']) 
                / stats['starting_balance'] * 100, 2
            )
        }
        
        with open(config.LOG_DAILY_LIMIT_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def get_daily_summary(self) -> str:
        """Generate human-readable summary report"""
        lines = ["\n" + "="*100]
        lines.append("DAILY RISK MANAGEMENT SUMMARY")
        lines.append("="*100)
        lines.append(
            f"{'Date':<12} {'Trades':>7} {'B/S':>7} {'W/L':>7} {'Start Bal':>12} "
            f"{'End Bal':>12} {'Daily P&L':>12} {'Daily %':>9} "
            f"{'R-Multiple':>11} {'Status':<50}"
        )
        lines.append("-"*100)
        
        total_stopped_days = 0
        stop_reasons = defaultdict(int)
        
        for day in sorted(self.daily_stats.keys()):
            stats = self.daily_stats[day]
            drawdown_pct = (
                (stats['current_balance'] - stats['starting_balance']) 
                / stats['starting_balance'] * 100
            )
            
            w_l = f"{stats['total_wins']}/{stats['total_losses']}"
            
            if stats['stopped']:
                status = f"✗ STOPPED: {stats['stop_reason']}"
                total_stopped_days += 1
                stop_reasons[stats['stop_reason'].split(':')[0]] += 1
            else:
                status = "✓ Active"
            
            bs = f"{stats['buy_trades']}/{stats['sell_trades']}"
            lines.append(
                f"{day!s:<12} {stats['total_trades']:>7} {bs:>7} {w_l:>7} "
                f"${stats['starting_balance']:>10.2f} ${stats['current_balance']:>10.2f} "
                f"${stats['daily_pnl']:>10.2f} {drawdown_pct:>8.2f}% "
                f"{stats['daily_r']:>10.2f}R {status:<50}"
            )
        
        lines.append("="*100)
        lines.append(f"Total Days: {len(self.daily_stats)}")
        lines.append(f"Stopped Days: {total_stopped_days} ({total_stopped_days/len(self.daily_stats)*100:.1f}%)")
        
        if stop_reasons:
            lines.append("\nStop Reasons Breakdown:")
            for reason, count in sorted(stop_reasons.items(), key=lambda x: -x[1]):
                pct = count / total_stopped_days * 100 if total_stopped_days > 0 else 0
                lines.append(f"  {reason:<30}: {count:>3} days ({pct:>5.1f}%)")
        
        lines.append("="*100 + "\n")
        
        return "\n".join(lines)


def calculate_position_size(
    balance: float, 
    entry_price: float,
    stop_loss: float,
    risk_percent: float = 1.0
) -> float:
    """
    ✅ NEW: Calculate position size based on % risk per trade
    
    Args:
        balance: Current account balance
        entry_price: Trade entry price
        stop_loss: Stop loss price
        risk_percent: % of balance to risk (default 1%)
    
    Returns:
        Position size in lots
    """
    from . import config
    
    # Calculate risk amount in dollars
    risk_amount = balance * (risk_percent / 100)
    
    # Calculate stop distance in pips
    stop_distance_pips = abs(entry_price - stop_loss) / config.PIP_SIZE
    
    if stop_distance_pips <= 0:
        return 0.01  # Minimum size
    
    # Calculate position size
    # risk_amount = pips * pip_value * lots
    # lots = risk_amount / (pips * pip_value)
    position_size = risk_amount / (stop_distance_pips * config.PIP_VALUE_PER_LOT)
    
    # Apply min/max limits
    position_size = max(0.01, position_size)  # Min 0.01 lots
    position_size = min(position_size, 100.0)  # Max 100 lots
    
    # Round to 2 decimal places
    position_size = round(position_size, 2)
    
    return position_size


def check_trade_outcome(trade_data: dict, current_candle, series: dict, now_utc):
    """
    Check if trade hit TP or SL
    
    Returns outcome dict or None if still open
    """
    from . import config
    
    direction = trade_data['direction']
    entry_price = trade_data['entry']
    stop_loss = trade_data.get('stop_loss')
    tp1_price = trade_data.get('tp1_price')
    tp2_price = trade_data.get('tp2_price')
    
    # Check if moved to breakeven
    moved_to_be = trade_data.get('moved_to_be', False)
    
    if direction == "SELL":
        # Check stop loss first
        if current_candle.high >= stop_loss:
            return {
                'type': 'TP1_THEN_BE' if moved_to_be else 'SL_HIT',
                'exit_price': stop_loss,
                'exit_time': now_utc
            }
        
        # Check TP1
        if tp1_price and current_candle.low <= tp1_price:
            if not moved_to_be:
                # First time hitting TP1 - move to BE
                trade_data['moved_to_be'] = True
                if config.ENABLE_BREAK_EVEN:
                    trade_data['stop_loss'] = entry_price - (2 * config.PIP_SIZE)
                # Don't close yet
                return None
        
        # Check TP2
        if tp2_price and current_candle.low <= tp2_price:
            return {
                'type': 'TP1_THEN_TP2',
                'exit_price': tp2_price,
                'exit_time': now_utc
            }
    
    else:  # BUY
        # Check stop loss first
        if current_candle.low <= stop_loss:
            return {
                'type': 'TP1_THEN_BE' if moved_to_be else 'SL_HIT',
                'exit_price': stop_loss,
                'exit_time': now_utc
            }
        
        # Check TP1
        if tp1_price and current_candle.high >= tp1_price:
            if not moved_to_be:
                # First time hitting TP1 - move to BE
                trade_data['moved_to_be'] = True
                if config.ENABLE_BREAK_EVEN:
                    trade_data['stop_loss'] = entry_price + (2 * config.PIP_SIZE)
                return None
        
        # Check TP2
        if tp2_price and current_candle.high >= tp2_price:
            return {
                'type': 'TP1_THEN_TP2',
                'exit_price': tp2_price,
                'exit_time': now_utc
            }
    
    return None  # Still open

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


def trade_identity(record: dict) -> tuple:
    def round_price(value: float | None) -> float | None:
        if value is None:
            return None
        return round(float(value), 5)

    return (
        record.get("direction"),
        round_price(record.get("entry")),
        round_price(record.get("stop_loss")),
        round_price(record.get("tp1_price")),
        round_price(record.get("tp2_price")),
    )


def run_backtest(
    csv_path: Path,
    source_minutes: int,
    output_path: Path,
    start: datetime | None,
    end: datetime | None,
) -> None:
    """
    ✅ FIXED: Main backtest loop with proper risk management
    """
    from . import config
    
    # Initialize account and risk manager
    starting_balance = config.ACCOUNT_BALANCE_OVERRIDE or 10000.0
    current_balance = starting_balance
    risk_manager = DailyRiskManager(starting_balance)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
    outcome_path.write_text("", encoding="utf-8")
    if config.LOG_DAILY_LIMITS:
        Path(config.LOG_DAILY_LIMIT_FILE).write_text("", encoding="utf-8")

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
    candles_d1: list[Candle] | None = None
    if config.TP3_ENABLED and config.TP3_LEG_SOURCE == "D1":
        candles_d1 = resample_candles(
            base_candles, source_seconds, TIMEFRAME_SECONDS[TIMEFRAME_D1]
        )

    series = {
        TIMEFRAME_M5: build_series(candles_5m, TIMEFRAME_M5),
        TIMEFRAME_M15: build_series(candles_15m, TIMEFRAME_M15),
        TIMEFRAME_H4: build_series(candles_4h, TIMEFRAME_H4),
    }
    if config.USE_1M_ENTRY and candles_1m is not None:
        series[TIMEFRAME_M1] = build_series(candles_1m, TIMEFRAME_M1)
    if candles_d1 is not None:
        series[TIMEFRAME_D1] = build_series(candles_d1, TIMEFRAME_D1)

    provider = HistoricalCandleProvider(series)
    engine = SignalEngine(symbol=config.SYMBOL_VARIANTS[0], candle_provider=provider)
    logger = SignalLogger(str(output_path))
    outcome_logger = SignalLogger(str(outcome_path))

    step_timeframe = TIMEFRAME_M1 if config.USE_1M_ENTRY else TIMEFRAME_M5
    step_seconds = TIMEFRAME_SECONDS[step_timeframe]
    step_candles = candles_1m if config.USE_1M_ENTRY else candles_5m
    if step_candles is None:
        raise ValueError("USE_1M_ENTRY requires 1-minute candle series")
    
    # Track open trades
    open_trades = []
    seen_trade_keys: set[tuple] = set()
    
    # ✅ Main loop with FIXED risk management
    for i, candle in enumerate(step_candles):
        if i < 1:
            continue
        
        now_utc = step_candles[i-1].time_utc + timedelta(seconds=step_seconds + 1)
        current_date = now_utc.date()
        
        if start and now_utc < start:
            continue
        if end and now_utc > end:
            continue
        if session_from_utc(now_utc) is None:
            continue
        
        # ✅ FIXED: Reset daily counters with CURRENT balance
        risk_manager.reset_if_new_day(current_date, current_balance)
        
        # ✅ Check if we should stop trading BEFORE processing anything
        if config.ENABLE_DAILY_RISK_MANAGEMENT:
            should_stop, reason = risk_manager.should_stop_trading(
                current_date, 
                now_utc, 
                current_balance
            )
            
            if should_stop:
                # Skip ALL activity for rest of day
                continue
        
        # ✅ Process existing open trades
        closed_indices = []
        for idx, trade_data in enumerate(open_trades):
            # Check if trade hit TP or SL this candle
            outcome = check_trade_outcome(
                trade_data,
                candle,
                series,
                now_utc
            )
            
            if outcome:
                # Trade closed!
                trade_data['outcome'] = outcome['type']
                trade_data['exit_time'] = outcome['exit_time']
                trade_data['exit_price'] = outcome['exit_price']
                
                # Calculate P&L
                entry_price = trade_data['entry']
                stop_loss = trade_data['stop_loss']
                exit_price = outcome['exit_price']
                direction = trade_data['direction']
                
                # Calculate risk in pips
                if direction == "SELL":
                    risk_pips = (stop_loss - entry_price) / config.PIP_SIZE
                    result_pips = (entry_price - exit_price) / config.PIP_SIZE
                else:
                    risk_pips = (entry_price - stop_loss) / config.PIP_SIZE
                    result_pips = (exit_price - entry_price) / config.PIP_SIZE
                
                # Calculate R-multiple
                r_multiple = result_pips / risk_pips if risk_pips > 0 else 0
                
                # ✅ Calculate dollar P&L using ACTUAL position size
                position_size = trade_data.get('position_size_lots', 0.01)
                pnl = result_pips * config.PIP_VALUE_PER_LOT * position_size
                
                # ✅ Update balance IMMEDIATELY
                current_balance += pnl
                
                # ✅ Record in risk manager with ACTUAL balance
                risk_manager.record_trade_result(
                    current_date, 
                    pnl, 
                    r_multiple,
                    current_balance,
                    direction,
                )
                
                # Store results
                trade_data['pnl'] = pnl
                trade_data['r_multiple'] = r_multiple
                trade_data['balance_after'] = current_balance
                
                # Log
                outcome_logger.log(trade_data)
                
                # Mark for removal
                closed_indices.append(idx)
                
                # ✅ CHECK LIMITS IMMEDIATELY AFTER EACH TRADE CLOSES
                if config.ENABLE_DAILY_RISK_MANAGEMENT:
                    should_stop, reason = risk_manager.should_stop_trading(
                        current_date,
                        now_utc,
                        current_balance
                    )
                    
                    if should_stop:
                        # Stop processing immediately
                        break
        
        # Remove closed trades (reverse order to maintain indices)
        for idx in reversed(closed_indices):
            open_trades.pop(idx)
        
        # ✅ Re-check if we should stop (in case last trade triggered limit)
        if config.ENABLE_DAILY_RISK_MANAGEMENT:
            should_stop, reason = risk_manager.should_stop_trading(
                current_date,
                now_utc,
                current_balance
            )
            
            if should_stop:
                continue
        
        # ✅ Evaluate for new signals
        signal = engine.evaluate(now_utc)
        logger.log(signal)
        
        if signal.decision == "TRADE":
            signal_key = trade_identity(
                {
                    "direction": signal.direction,
                    "entry": signal.entry,
                    "stop_loss": signal.stop_loss,
                    "tp1_price": signal.tp1_price,
                    "tp2_price": signal.tp2_price,
                }
            )
            if signal_key in seen_trade_keys:
                continue
            if any(trade_identity(trade) == signal_key for trade in open_trades):
                continue
            # ✅ Calculate position size based on current balance
            position_size = calculate_position_size(
                balance=current_balance,
                entry_price=signal.entry,
                stop_loss=signal.stop_loss,
                risk_percent=1.0  # Risk 1% per trade
            )
            
            record = build_outcome_record(
                signal=signal,
                series=series,
                entry_timeframe=step_timeframe,
            )
            record['balance_before'] = current_balance
            record['position_size_lots'] = position_size  # ✅ Store calculated size
            open_trades.append(record)
            seen_trade_keys.add(signal_key)
    
    # Print summary
    total_pnl = current_balance - starting_balance
    total_return = (total_pnl / starting_balance * 100) if starting_balance else 0.0
    print("\n" + "=" * 100)
    print("BACKTEST COMPLETED")
    print("=" * 100)
    print(f"Starting Balance: ${starting_balance:,.2f}")
    print(f"Ending Balance:   ${current_balance:,.2f}")
    print(f"Total P&L:        ${total_pnl:+,.2f}")
    print(f"Return:           {total_return:+.2f}%")
    print("=" * 100 + "\n")
    print(risk_manager.get_daily_summary())


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
    """
    FIXED VERSION: Realistic trade simulation with spread, slippage, and conservative logic
    """
    direction = record.get("direction")
    entry_price = record.get("entry")
    stop_loss = record.get("stop_loss")
    tp1 = record.get("tp1_price")
    tp2 = record.get("tp2_price")
    
    if direction not in ("BUY", "SELL") or entry_price is None:
        return {"outcome": "INVALID_RECORD"}

    # FIXED: Add realistic spread and slippage costs
    SPREAD_PIPS = 1.5  # Average spread for GU
    ENTRY_SLIPPAGE_PIPS = 0.5  # Entry slippage
    SL_SLIPPAGE_PIPS = 2.0  # Stop loss slippage (worse in fast markets)
    TP_SLIPPAGE_PIPS = 0.5  # Take profit slippage
    
    spread_cost = SPREAD_PIPS * config.PIP_SIZE
    entry_slip = ENTRY_SLIPPAGE_PIPS * config.PIP_SIZE
    sl_slip = SL_SLIPPAGE_PIPS * config.PIP_SIZE
    tp_slip = TP_SLIPPAGE_PIPS * config.PIP_SIZE
    
    # Adjust prices for spread and slippage
    if direction == "BUY":
        # Buy at ASK (higher price)
        entry_price_actual = entry_price + spread_cost + entry_slip
        # Stop loss hits at worse price
        stop_loss_actual = stop_loss - sl_slip if stop_loss else None
        # Take profit hits at worse price
        tp1_actual = tp1 - spread_cost - tp_slip if tp1 else None
        tp2_actual = tp2 - spread_cost - tp_slip if tp2 else None
    else:  # SELL
        # Sell at BID (lower price)
        entry_price_actual = entry_price - spread_cost - entry_slip
        # Stop loss hits at worse price
        stop_loss_actual = stop_loss + sl_slip if stop_loss else None
        # Take profit hits at worse price (need to buy back at ASK)
        tp1_actual = tp1 + spread_cost + tp_slip if tp1 else None
        tp2_actual = tp2 + spread_cost + tp_slip if tp2 else None

    entry_time = parse_timestamp(record["timestamp_utc"])
    open_times = [c.time_utc for c in tf_series.candles]
    start_idx = bisect_left(open_times, entry_time)

    # FIXED: More realistic entry fill check
    entry_idx = None
    entry_fill_time = None
    for i in range(start_idx, len(tf_series.candles)):
        candle = tf_series.candles[i]
        
        # Check if entry would have filled with spread/slippage considered
        if check_entry_fill(candle, entry_price, entry_price_actual, direction):
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
    active_stop = stop_loss_actual

    for candle in tf_series.candles[entry_idx:]:
        hit_sl = False
        hit_tp1 = False
        hit_tp2 = False

        if direction == "BUY":
            hit_sl = active_stop is not None and candle.low <= active_stop
            hit_tp2 = tp2_actual is not None and candle.high >= tp2_actual
            hit_tp1 = tp1_actual is not None and candle.high >= tp1_actual
        else:
            hit_sl = active_stop is not None and candle.high >= active_stop
            hit_tp2 = tp2_actual is not None and candle.low <= tp2_actual
            hit_tp1 = tp1_actual is not None and candle.low <= tp1_actual

        if hit_tp2:
            hit_tp1 = True

        if not tp1_hit:
            # FIXED: Use conservative intra-candle logic
            if hit_sl and (hit_tp1 or hit_tp2):
                # When both SL and TP are hit in same candle, use conservative logic
                sequence = estimate_intra_candle_sequence(
                    candle, direction, entry_price_actual, active_stop, 
                    tp1_actual if hit_tp1 else tp2_actual
                )
                
                if sequence == "TP_FIRST":
                    tp1_hit = True
                    tp1_time = candle.time_utc
                    if hit_tp2:
                        tp2_hit = True
                        tp2_time = candle.time_utc
                        return build_outcome(
                            "TP2_HIT", entry_fill_time, tp1_time, tp2_time, None
                        )
                    # FIXED: Move stop to breakeven with buffer
                    if config.ENABLE_BREAK_EVEN:
                        be_buffer = 2 * config.PIP_SIZE
                        if direction == "BUY":
                            active_stop = entry_price_actual + be_buffer
                        else:
                            active_stop = entry_price_actual - be_buffer
                    continue
                else:
                    # Conservative: assume SL hit first
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
                # FIXED: Move stop to breakeven with buffer
                if config.ENABLE_BREAK_EVEN:
                    be_buffer = 2 * config.PIP_SIZE
                    if direction == "BUY":
                        active_stop = entry_price_actual + be_buffer
                    else:
                        active_stop = entry_price_actual - be_buffer
                continue
        else:
            # TP1 already hit, checking for TP2 or BE
            if hit_sl and hit_tp2:
                # FIXED: Conservative logic for TP2 vs BE
                sequence = estimate_intra_candle_sequence(
                    candle, direction, entry_price_actual, active_stop, tp2_actual
                )
                
                if sequence == "TP_FIRST":
                    tp2_hit = True
                    tp2_time = candle.time_utc
                    return build_outcome(
                        "TP1_THEN_TP2", entry_fill_time, tp1_time, tp2_time, None
                    )
                else:
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


def check_entry_fill(candle: Candle, entry_price: float, entry_price_actual: float, direction: str) -> bool:
    """
    FIXED: More realistic entry fill check
    Accounts for spread and checks if price actually reached our adjusted entry level
    """
    if direction == "BUY":
        # For buys, we need price to go low enough to hit our adjusted ask price
        # But not gap over it completely
        if candle.low <= entry_price_actual <= candle.high:
            # Check we didn't gap over
            if candle.open > entry_price_actual and candle.low > entry_price_actual:
                return False
            return True
        return False
    else:  # SELL
        # For sells, we need price to go high enough to hit our adjusted bid price
        if candle.low <= entry_price_actual <= candle.high:
            # Check we didn't gap over
            if candle.open < entry_price_actual and candle.high < entry_price_actual:
                return False
            return True
        return False


def estimate_intra_candle_sequence(
    candle: Candle, 
    direction: str, 
    entry: float, 
    sl: float, 
    tp: float
) -> str:
    """
    FIXED: Conservative intra-candle sequence estimation
    
    When we can't know for certain which was hit first (SL or TP),
    we use a CONSERVATIVE approach that assumes the worst case.
    
    This gives more realistic backtest results than optimistic assumptions.
    """
    # Calculate distances from entry
    if direction == "BUY":
        sl_distance = entry - sl  # Distance below entry
        tp_distance = tp - entry  # Distance above entry
        
        # If candle closed below entry, very likely hit SL first
        if candle.close < entry:
            return "SL_FIRST"
        
        # If candle is strongly bullish (close near high), likely hit TP first
        body_position = (candle.close - candle.low) / (candle.high - candle.low) if candle.high > candle.low else 0.5
        if body_position > 0.75:  # Close in upper 25% of range
            return "TP_FIRST"
        
        # Default: conservative assumption
        return "SL_FIRST"
    
    else:  # SELL
        sl_distance = sl - entry  # Distance above entry
        tp_distance = entry - tp  # Distance below entry
        
        # If candle closed above entry, very likely hit SL first
        if candle.close > entry:
            return "SL_FIRST"
        
        # If candle is strongly bearish (close near low), likely hit TP first
        body_position = (candle.close - candle.low) / (candle.high - candle.low) if candle.high > candle.low else 0.5
        if body_position < 0.25:  # Close in lower 25% of range
            return "TP_FIRST"
        
        # Default: conservative assumption
        return "SL_FIRST"


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
