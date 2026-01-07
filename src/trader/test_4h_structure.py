# test_ibos_detection.py
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import sys

# Allow running as a script from repo root.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trader.backtest import load_csv_candles, resample_candles
from trader.structure import find_swings, find_breaks
from trader import config
from trader.timeframes import TIMEFRAME_SECONDS, TIMEFRAME_H4

parser = ArgumentParser(description="Inspect 4H structure for a date range.")
parser.add_argument("--csv", type=Path, default=ROOT / "data" / "gu_1m.csv")
parser.add_argument("--source-minutes", type=int, default=1)
parser.add_argument("--start", type=str, default="2025-10-01")
parser.add_argument("--end", type=str, default="2025-11-30")
args = parser.parse_args()

# Load data
csv_path = args.csv
candles = load_csv_candles(csv_path)

# Filter to your test range
start = datetime.fromisoformat(args.start)
end = datetime.fromisoformat(args.end)
candles = [c for c in candles if start <= c.time_utc <= end]

print(f"Loaded {len(candles)} 1M candles")
if not candles:
    raise SystemExit("No candles found for the selected range.")
print(f"Date range: {candles[0].time_utc} to {candles[-1].time_utc}")

# Resample to 4H
source_seconds = args.source_minutes * 60
candles_4h = resample_candles(
    candles,
    source_seconds,
    TIMEFRAME_SECONDS[TIMEFRAME_H4],
)

print(f"\n4H candles: {len(candles_4h)}")
if candles_4h:
    print(f"First 4H: {candles_4h[0].time_utc}")
    print(f"Last 4H: {candles_4h[-1].time_utc}")

# Find swings
swings = find_swings(candles_4h, config.SWING_LEFT_4H, config.SWING_RIGHT_4H)
print(f"\n4H Swing points found: {len(swings)}")
print(f"  Swing highs: {len([s for s in swings if s.kind == 'high'])}")
print(f"  Swing lows: {len([s for s in swings if s.kind == 'low'])}")

if swings:
    print("\nAll swings in chronological order:")
    for i, swing in enumerate(swings):
        print(f"  {i+1}. {swing.kind.upper()}: {swing.price:.5f} at {swing.time_utc} (index {swing.index})")

# Find breaks
events = find_breaks(candles_4h, swings)
print(f"\n{'='*80}")
print(f"4H Break events found: {len(events)}")
print(f"{'='*80}")

if events:
    print("\n🔍 DETAILED BREAK ANALYSIS:\n")
    for i, event in enumerate(events):
        print(f"{'='*80}")
        print(f"EVENT #{i+1}: {event.event_type} {event.direction.upper()}")
        print(f"{'='*80}")
        print(f"  Time: {event.time_utc}")
        print(f"  Break level: {event.break_level:.5f}")
        print(f"  Close price: {event.close_price:.5f}")
        print(f"  Defining swing price: {event.defining_swing_price}")
        print(f"  Defining swing index: {event.defining_swing_index}")
        
        # CHECK FOR IBOS - Key validation
        print(f"\n  🔍 VALIDATION CHECKS:")
        
        # Check 1: Is this breaking the most recent relevant swing?
        if event.direction == "bull":
            # Bull break should break most recent swing high
            recent_highs = [s for s in swings if s.kind == "high" and s.index < event.index]
            if recent_highs:
                last_high = recent_highs[-1]
                print(f"  ✓ Last swing high before break: {last_high.price:.5f} at {last_high.time_utc}")
                if abs(last_high.price - event.break_level) < 0.00001:
                    print(f"  ✓ Breaking the correct swing (most recent high)")
                else:
                    print(f"  ⚠️  Break level {event.break_level:.5f} != Last high {last_high.price:.5f}")
                    print(f"  ⚠️  POSSIBLE iBOS: Not breaking most recent swing!")
        else:
            # Bear break should break most recent swing low
            recent_lows = [s for s in swings if s.kind == "low" and s.index < event.index]
            if recent_lows:
                last_low = recent_lows[-1]
                print(f"  ✓ Last swing low before break: {last_low.price:.5f} at {last_low.time_utc}")
                if abs(last_low.price - event.break_level) < 0.00001:
                    print(f"  ✓ Breaking the correct swing (most recent low)")
                else:
                    print(f"  ⚠️  Break level {event.break_level:.5f} != Last low {last_low.price:.5f}")
                    print(f"  ⚠️  POSSIBLE iBOS: Not breaking most recent swing!")
        
        # Check 2: Invalidation check (does it stay valid?)
        future_candles = candles_4h[event.index + 1:]
        
        if event.direction == "bull":
            # For bull BOS, should not close below defining swing low
            if event.defining_swing_price:
                violated = [c for c in future_candles if c.close < event.defining_swing_price]
                if violated:
                    print(f"\n  ❌ BIAS INVALIDATED!")
                    print(f"     Defining swing low: {event.defining_swing_price:.5f}")
                    print(f"     First violation at: {violated[0].time_utc}")
                    print(f"     Closed at: {violated[0].close:.5f}")
                    print(f"     This BOS becomes invalid for trading!")
                else:
                    print(f"\n  ✅ BIAS REMAINS VALID")
                    print(f"     Price stayed above defining low {event.defining_swing_price:.5f}")
        else:
            # For bear BOS, should not close above defining swing high
            if event.defining_swing_price:
                violated = [c for c in future_candles if c.close > event.defining_swing_price]
                if violated:
                    print(f"\n  ❌ BIAS INVALIDATED!")
                    print(f"     Defining swing high: {event.defining_swing_price:.5f}")
                    print(f"     First violation at: {violated[0].time_utc}")
                    print(f"     Closed at: {violated[0].close:.5f}")
                    print(f"     This BOS becomes invalid for trading!")
                else:
                    print(f"\n  ✅ BIAS REMAINS VALID")
                    print(f"     Price stayed below defining high {event.defining_swing_price:.5f}")
        
        # Check 3: How long does this BOS stay valid?
        if i < len(events) - 1:
            next_event = events[i + 1]
            time_valid = (next_event.time_utc - event.time_utc).total_seconds() / 3600
            print(f"\n  ⏰ Valid for: {time_valid:.1f} hours until next BOS")
        else:
            print(f"\n  ⏰ Still valid at end of test period")
        
        print()

    # Summary
    print(f"\n{'='*80}")
    print("📊 SUMMARY:")
    print(f"{'='*80}")
    
    bos_count = len([e for e in events if e.event_type == "BOS"])
    choch_count = len([e for e in events if e.event_type == "CHoCH"])
    bull_count = len([e for e in events if e.direction == "bull"])
    bear_count = len([e for e in events if e.direction == "bear"])
    
    print(f"Total events: {len(events)}")
    print(f"  BOS: {bos_count}")
    print(f"  CHoCH: {choch_count}")
    print(f"  Bullish: {bull_count}")
    print(f"  Bearish: {bear_count}")
    
    print("\n⚠️  KEY INSIGHTS:")
    print(f"  - If you see 'POSSIBLE iBOS' warnings, those breaks are suspect")
    print(f"  - If you see 'BIAS INVALIDATED', no trades should happen after that point")
    print(f"  - Check if BOS events happen during your trading sessions")
    print(f"  - Your sessions: London {config.LONDON_START_UTC}-{config.LONDON_END_UTC}, NY {config.NY_START_UTC if config.NY_ENABLED else 'DISABLED'}")
    
else:
    print("  ❌ NO BREAK EVENTS FOUND!")
    print("\nThis is why you have no trades.")
    print("Possible reasons:")
    print("  1. Not enough 4H candles in the test period")
    print("  2. SWING_LEFT_4H/RIGHT_4H too large (need more candles)")
    print("  3. Market was ranging (no clear structure breaks)")
