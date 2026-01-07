from datetime import time

SYMBOL_CANONICAL = "GU"
SYMBOL_VARIANTS = ("GBPUSD", "GBPUSDm", "GBPUSD.r")

PIP_SIZE = 0.0001
# FIXED: Increase minimum sweep size to avoid noise
LIQUIDITY_SWEEP_PIPS = 5  # Was 1, now 5 for actual sweeps

POLL_SECONDS = 60
CANDLE_COUNT = 300

SWING_LEFT = 2
SWING_RIGHT = 2
SWING_LEFT_4H = 4
SWING_RIGHT_4H = 4

BROKER_UTC_OFFSET_HOURS = 0

# FIXED: 24-hour trading for better coverage
LONDON_START_UTC = time(0, 0)  # Was time(7, 0)
LONDON_END_UTC = time(23, 59)  # Was time(11, 0)

NY_ENABLED = False  # Disabled since London covers 24h
NY_START_UTC = time(12, 0)
NY_END_UTC = time(14, 0)

# FIXED: Enable 1M entry for sells to get better entries
USE_1M_ENTRY = True
REQUIRE_1M_CHOCH_PREMIUM = True
REQUIRE_5M_CHOCH_PREMIUM = True

# FIXED: Enable 1M entry for sells (was False, now True)
ENABLE_1M_ENTRY_SELL = True

# FIXED: Don't require premium on 5M for sells - too restrictive
REQUIRE_5M_CHOCH_PREMIUM_SELL = False

# FIXED: Adjusted premium/discount levels for better zones
# Premium = upper zone (for sells) - should be 61.8%+ for true premium
# Discount = lower zone (for buys) - should be 38.2%- for true discount
PREMIUM_CROSS_LEVEL = 0.75  # Need to reach 75%+ of leg initially
DISCOUNT_CROSS_LEVEL = 0.25  # Need to reach 25%- of leg initially

ENABLE_SPREAD_FILTER = False
MAX_SPREAD_PIPS = 4.0
ASSUME_ZERO_SPREAD = False  # Account for spread in backtest

# FIXED: Increase SL buffer to account for spread and slippage in backtest
SL_EXTRA_PIPS = 3.0

ENABLE_BREAK_EVEN = True  # Move SL to BE after TP1

# FIXED: Don't require liquidity sweep for sells - too restrictive
REQUIRE_NO_LIQUIDITY_SWEEP = False
REQUIRE_LIQUIDITY_SWEEP_SELL = False

ENABLE_CHOCH_RANGE_FILTER = False
MIN_CHOCH_RANGE_PIPS = 6.0

ENABLE_RISK_MANAGEMENT = True
RISK_PER_TRADE_PCT = 1.0
ACCOUNT_BALANCE_OVERRIDE = 10000.0
PIP_VALUE_PER_LOT = 10.0
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 5.0
LOT_STEP = 0.01

# NEW: Add these for better stop loss placement
SL_CANDLE_LOOKBACK = 3  # Look back 3 candles for structure high/low

# ==================================================
# DAILY RISK MANAGEMENT SYSTEM
# ==================================================

# Enable/disable the entire system
ENABLE_DAILY_RISK_MANAGEMENT = True

# --- Daily Drawdown Limit (PRIMARY PROTECTION) ⭐⭐⭐ ---
ENABLE_DAILY_DRAWDOWN_LIMIT = True
MAX_DAILY_DRAWDOWN_PERCENT = 3.0
# Stops when daily loss reaches -3% of account
# Purpose: Capital preservation - MOST IMPORTANT LIMIT!
# Example: $10,000 account → stops at -$300 loss

# --- Consecutive Loss Limit (Early Warning) ---
ENABLE_CONSECUTIVE_LOSS_LIMIT = True
MAX_CONSECUTIVE_LOSSES = 4
# Stops after 4 losses in a row
# Purpose: Detect when strategy is broken for current conditions
# Red flag that something is off with market conditions

# --- Max Daily Losses (Secondary Safety) ---
ENABLE_MAX_LOSSES_LIMIT = True
MAX_DAILY_LOSSES = 10
# Stops after 10 total losses today (regardless of wins)
# Purpose: Prevent overtrading on bad days
# With 48% loss rate, 10 losses = below-average day

# --- Daily Profit Target (Lock Gains) ---
ENABLE_DAILY_PROFIT_TARGET = False
DAILY_PROFIT_TARGET_PERCENT = 1000.0
# Stops when daily profit reaches +8% of account
# Purpose: Lock in great days, avoid giving back profits
# Example: $10,000 account → stops at +$800 profit

# --- Limit Priority ---
LIMIT_PRIORITY = "FIRST"
# "FIRST" = Stop at whichever limit hits first
# "STRICTEST" = Use the strictest limit (not implemented yet)

# --- Reset Time ---
DAILY_RESET_HOUR_UTC = 0  # Reset at midnight UTC
# Defines when "new day" starts for limit tracking

# --- Logging ---
LOG_DAILY_LIMITS = True
LOG_DAILY_LIMIT_FILE = "daily_limits_log.jsonl"
# Logs when daily limits are hit for analysis