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
    TIMEFRAME_SECONDS,
)


from datetime import date
from collections import defaultdict


class DailyRiskManager:
    """
    Manages daily risk limits for trading
    
    Features:
    - Daily drawdown % limit (primary protection)
    - Consecutive loss limit (early warning)
    - Max daily losses limit (secondary safety)
    - Daily profit target (lock gains)
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
            'daily_pnl': 0.0,
            'daily_r': 0.0,
            'stopped': False,
            'stop_reason': None,
            'stop_time': None
        })
    
    def reset_if_new_day(self, current_date: date, current_balance: float):
        """Initialize stats for a new trading day"""
        if current_date not in self.daily_stats:
            self.daily_stats[current_date] = {
                'starting_balance': current_balance,
                'current_balance': current_balance,
                'total_losses': 0,
                'consecutive_losses': 0,
                'total_wins': 0,
                'total_trades': 0,
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
        current_balance: float
    ) -> tuple[bool, str | None]:
        """
        Check all daily limit conditions
        
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
        
        # Calculate daily drawdown %
        daily_pnl_pct = (
            (stats['current_balance'] - stats['starting_balance']) 
            / stats['starting_balance'] 
            * 100
        )
        
        limits_hit = []
        
        # Check 1: Daily Drawdown % (MOST IMPORTANT) ⭐
        if config.ENABLE_DAILY_DRAWDOWN_LIMIT:
            if daily_pnl_pct <= -config.MAX_DAILY_DRAWDOWN_PERCENT:
                loss_amount = stats['starting_balance'] - stats['current_balance']
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
                profit_amount = stats['current_balance'] - stats['starting_balance']
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
            
            # Log if enabled
            if config.LOG_DAILY_LIMITS:
                self._log_daily_stop(current_date, current_time, stats)
            
            return True, stats['stop_reason']
        
        return False, None
    
    def record_trade_result(
        self, 
        current_date: date, 
        pnl: float,
        r_result: float
    ):
        """
        Update daily statistics after trade closes
        
        Args:
            current_date: Trading date
            pnl: Dollar profit/loss
            r_result: R-multiple result
        """
        stats = self.daily_stats[current_date]
        
        stats['total_trades'] += 1
        stats['daily_pnl'] += pnl
        stats['daily_r'] += r_result
        stats['current_balance'] += pnl
        
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
            f"{'Date':<12} {'Trades':>7} {'W/L':>7} {'Daily P&L':>12} "
            f"{'Daily %':>9} {'R-Multiple':>11} {'Status':<50}"
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
            
            lines.append(
                f"{day!s:<12} {stats['total_trades']:>7} {w_l:>7} "
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
    """
    Run backtest with daily risk management
    
    Features:
    - Daily drawdown limits
    - Consecutive loss limits
    - Max daily losses
    - Profit target locks
    """
    from . import config
    
    # Initialize risk manager
    starting_balance = config.ACCOUNT_BALANCE_OVERRIDE or 10000.0
    current_balance = starting_balance
    risk_manager = DailyRiskManager(starting_balance)

    # Setup output files
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("", encoding="utf-8")
    outcome_path = output_path.with_name(output_path.stem + "_outcomes.jsonl")
    outcome_path.write_text("", encoding="utf-8")

    # Load and filter candles
    base_candles = load_csv_candles(csv_path)
    if start or end:
        base_candles = [
            candle
            for candle in base_candles
            if (start is None or candle.time_utc >= start)
            and (end is None or candle.time_utc <= end)
        ]

    # Resample to all timeframes
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

    # Build series for all timeframes
    series = {
        TIMEFRAME_M5: build_series(candles_5m, TIMEFRAME_M5),
        TIMEFRAME_M15: build_series(candles_15m, TIMEFRAME_M15),
        TIMEFRAME_H4: build_series(candles_4h, TIMEFRAME_H4),
    }
    if config.USE_1M_ENTRY and candles_1m is not None:
        series[TIMEFRAME_M1] = build_series(candles_1m, TIMEFRAME_M1)

    # Initialize engine and loggers
    provider = HistoricalCandleProvider(series)
    engine = SignalEngine(symbol=config.SYMBOL_VARIANTS[0], candle_provider=provider)
    logger = SignalLogger(str(output_path))
    outcome_logger = SignalLogger(str(outcome_path))

    # Determine evaluation timeframe
    step_timeframe = TIMEFRAME_M1 if config.USE_1M_ENTRY else TIMEFRAME_M5
    step_seconds = TIMEFRAME_SECONDS[step_timeframe]
    step_candles = candles_1m if config.USE_1M_ENTRY else candles_5m
    if step_candles is None:
        raise ValueError("USE_1M_ENTRY requires 1-minute candle series")
    
    # Track open positions for risk management
    open_positions = {}  # {entry_time: position_data}
    
    print(f"\n{'='*100}")
    print(f"BACKTEST STARTED")
    print(f"{'='*100}")
    print(f"Period: {start} to {end}")
    print(f"Candles: {len(step_candles)} ({step_timeframe}m intervals)")
    print(f"Starting Balance: ${starting_balance:,.2f}")
    print(f"Risk Management: {'ENABLED' if config.ENABLE_DAILY_RISK_MANAGEMENT else 'DISABLED'}")
    if config.ENABLE_DAILY_RISK_MANAGEMENT:
        print(f"  - Max Daily Drawdown: {config.MAX_DAILY_DRAWDOWN_PERCENT}%")
        print(f"  - Max Consecutive Losses: {config.MAX_CONSECUTIVE_LOSSES}")
        print(f"  - Max Daily Losses: {config.MAX_DAILY_LOSSES}")
        print(f"  - Daily Profit Target: {config.DAILY_PROFIT_TARGET_PERCENT}%")
    print(f"{'='*100}\n")
    
    # Main evaluation loop with 1 candle delay to avoid lookahead bias
    for i, candle in enumerate(step_candles):
        if i < 1:
            continue  # Skip first candle to ensure we have previous candle data
        
        # Evaluate using previous candle's close + small delay
        # This simulates realistic reaction time to candle close
        now_utc = step_candles[i-1].time_utc + timedelta(seconds=step_seconds + 1)
        current_date = now_utc.date()
        
        # Filter by date range
        if start and now_utc < start:
            continue
        if end and now_utc > end:
            continue
        
        # Filter by session
        if session_from_utc(now_utc) is None:
            continue
        
        # ⭐⭐⭐ DAILY RISK MANAGEMENT CHECK ⭐⭐⭐
        if config.ENABLE_DAILY_RISK_MANAGEMENT:
            # Reset if new day
            risk_manager.reset_if_new_day(current_date, current_balance)
            
            # Check if we should stop trading today
            should_stop, stop_reason = risk_manager.should_stop_trading(
                current_date, 
                now_utc, 
                current_balance
            )
            
            if should_stop:
                # Skip evaluation for rest of day
                # (Only print once per day when first stopped)
                if not risk_manager.daily_stats[current_date].get('stop_logged', False):
                    print(f"[{now_utc.strftime('%Y-%m-%d %H:%M')}] 🛑 {stop_reason}")
                    risk_manager.daily_stats[current_date]['stop_logged'] = True
                continue
        
        # Manage existing open positions (check for TP/SL hits)
        closed_positions = []
        for entry_time, position in list(open_positions.items()):
            # Get current candle for this position's timeframe
            current_candle = candle
            
            # Check if position hit TP or SL
            outcome, exit_price, exit_time = check_position_exit(
                position, 
                current_candle, 
                now_utc
            )
            
            if outcome:
                # Position closed
                pnl = calculate_pnl(position, exit_price)
                r_result = calculate_r_multiple(position, exit_price)
                
                # Update balance
                current_balance += pnl
                
                # ⭐ Record in risk manager ⭐
                if config.ENABLE_DAILY_RISK_MANAGEMENT:
                    risk_manager.record_trade_result(current_date, pnl, r_result)
                
                # Log outcome
                position['outcome'] = outcome
                position['exit_price'] = exit_price
                position['exit_time'] = exit_time
                position['pnl'] = pnl
                position['r_multiple'] = r_result
                outcome_logger.log(position)
                
                closed_positions.append(entry_time)
        
        # Remove closed positions
        for entry_time in closed_positions:
            del open_positions[entry_time]
        
        # Evaluate for new trade signals
        signal = engine.evaluate(now_utc)
        logger.log(signal)
        
        if signal.decision == "TRADE":
            # Build trade record
            record = build_outcome_record(
                signal=signal,
                series=series,
                entry_timeframe=step_timeframe,
            )
            
            # Add to open positions
            open_positions[now_utc] = record
    
    # Close any remaining open positions at end of backtest
    print(f"\nClosing {len(open_positions)} remaining open positions...")
    for entry_time, position in open_positions.items():
        # Force close at last available price
        last_candle = step_candles[-1]
        exit_price = last_candle.close
        pnl = calculate_pnl(position, exit_price)
        r_result = calculate_r_multiple(position, exit_price)
        
        current_balance += pnl
        
        # Record final trades
        if config.ENABLE_DAILY_RISK_MANAGEMENT:
            trade_date = entry_time.date()
            risk_manager.record_trade_result(trade_date, pnl, r_result)
        
        position['outcome'] = 'FORCED_CLOSE'
        position['exit_price'] = exit_price
        position['exit_time'] = last_candle.time_utc
        position['pnl'] = pnl
        position['r_multiple'] = r_result
        outcome_logger.log(position)
    
    # Print final summary
    print(f"\n{'='*100}")
    print(f"BACKTEST COMPLETED")
    print(f"{'='*100}")
    print(f"Starting Balance: ${starting_balance:,.2f}")
    print(f"Ending Balance: ${current_balance:,.2f}")
    print(f"Total P&L: ${current_balance - starting_balance:+,.2f}")
    print(f"Return: {(current_balance - starting_balance) / starting_balance * 100:+.2f}%")
    print(f"{'='*100}")
    
    # Print daily risk management summary
    if config.ENABLE_DAILY_RISK_MANAGEMENT:
        print(risk_manager.get_daily_summary())


# ============================================================================
# HELPER FUNCTIONS (add these if you don't have them)
# ============================================================================

def check_position_exit(position: dict, current_candle: Candle, now_utc: datetime):
    """
    Check if position hit TP or SL
    
    Returns:
        (outcome, exit_price, exit_time) or (None, None, None) if still open
    """
    direction = position['direction']
    entry_price = position['entry']
    stop_loss = position['stop_loss']
    tp1_price = position.get('tp1_price')
    tp2_price = position.get('tp2_price')
    
    # Check if we've moved SL to breakeven after TP1
    moved_to_be = position.get('moved_to_breakeven', False)
    
    if direction == "SELL":
        # Check SL (above entry)
        if current_candle.high >= stop_loss:
            return "SL_HIT", stop_loss, now_utc
        
        # Check TP1
        if tp1_price and current_candle.low <= tp1_price:
            if not moved_to_be:
                # Hit TP1 for first time
                position['moved_to_breakeven'] = True
                position['stop_loss'] = entry_price - (2 * config.PIP_SIZE)  # Move to BE
                # Continue (don't close position yet)
                return None, None, None
            else:
                # Already hit TP1, now check TP2
                if tp2_price and current_candle.low <= tp2_price:
                    return "TP1_THEN_TP2", tp2_price, now_utc
                # Otherwise still open
                return None, None, None
        
    else:  # BUY
        # Check SL (below entry)
        if current_candle.low <= stop_loss:
            return "SL_HIT", stop_loss, now_utc
        
        # Check TP1
        if tp1_price and current_candle.high >= tp1_price:
            if not moved_to_be:
                # Hit TP1 for first time
                position['moved_to_breakeven'] = True
                position['stop_loss'] = entry_price + (2 * config.PIP_SIZE)  # Move to BE
                return None, None, None
            else:
                # Already hit TP1, now check TP2
                if tp2_price and current_candle.high >= tp2_price:
                    return "TP1_THEN_TP2", tp2_price, now_utc
                return None, None, None
    
    return None, None, None


def calculate_pnl(position: dict, exit_price: float) -> float:
    """Calculate dollar P&L for position"""
    entry_price = position['entry']
    position_size = position.get('position_size_lots', 0.01)
    direction = position['direction']
    
    if direction == "SELL":
        pips = (entry_price - exit_price) / config.PIP_SIZE
    else:  # BUY
        pips = (exit_price - entry_price) / config.PIP_SIZE
    
    pnl = pips * config.PIP_VALUE_PER_LOT * position_size
    return pnl


def calculate_r_multiple(position: dict, exit_price: float) -> float:
    """Calculate R-multiple for position"""
    entry_price = position['entry']
    stop_loss = position['stop_loss']
    direction = position['direction']
    
    if direction == "SELL":
        risk = (stop_loss - entry_price) / config.PIP_SIZE
        pips = (entry_price - exit_price) / config.PIP_SIZE
    else:  # BUY
        risk = (entry_price - stop_loss) / config.PIP_SIZE
        pips = (exit_price - entry_price) / config.PIP_SIZE
    
    if risk <= 0:
        return 0.0
    
    return pips / risk

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