Perfect. Below is **everything**, merged into **one single plain-English file**.
This is the **goal document**.
If something is not here, it is **not part of Version 1**.

---

# VERSION 1 — TRADING SYSTEM GOAL FILE

*(Locked scope, no extras)*

---

## 1. WHAT WE ARE BUILDING

We are building a rule-based trading system that:

* Connects to MT5
* Reads price every **60 seconds**
* Uses **GU (GBPUSD)** only (accept broker variants: GBPUSD, GBPUSDm, GBPUSD.r)
* Trades **London session** (UTC)
* Produces **trade signals only when all rules align**
* Does nothing when conditions are not perfect
* Uses **completed candles only** (no intra-candle or tick-based logic)
* Signal-only (no auto execution or order placement)
* Optional filters: max spread and minimum CHoCH candle range (config)

The system’s purpose is **selectivity**, not frequency.

---

## 2. TIMEFRAME MODEL (LOCKED)

Each timeframe has **one job only**.

* **4H** → Direction (bias only)
* **15M** → Location + setup
* **5M** → Entry trigger (default)
* **1M** → Optional entry trigger (config toggle)

---

## 3. 4H RULES — DIRECTION ONLY

4H answers **one question**:

> Are we only looking for BUYs, only SELLs, or NO TRADE?

Rules:

* Direction must be clear
* If unclear → no trades on all lower timeframes
* 4H is checked **only when a new 4H candle closes**
* Once bias is set, it stays until invalidated
* Bias is clear only if ALL are true:

  * At least one confirmed BOS in one direction
  * Last valid swing high/low is respected
  * No opposite BOS after the last impulse
* Bias is invalidated if:

  * An opposite-direction BOS occurs
  * OR price closes beyond the last valid swing that defined the bias

4H does **not**:

* Trigger trades
* Define entries
* Define stop loss or take profit

---

## 4. PREMIUM / DISCOUNT RULE (CORE EDGE)

Once 4H direction is known, the **active 4H leg** is defined.

Definition:

* The most recent impulse move that caused the last BOS
* Start = swing low (bullish leg) or swing high (bearish leg)
* End = impulse high/low that caused BOS

From that leg:

* Top 50% = **Premium**
* Bottom 50% = **Discount**

This rule controls **where structure is allowed**.

---

## 5. 15M RULES — SETUP LOCATION

15M structure is only valid **if it happens in the correct location**.

Structure definitions (15M and 5M):

* A swing high/low requires **at least 2 candles on each side**
* BOS = candle **closes beyond** prior swing high/low (wick-only breaks do not count)
* CHoCH = BOS **against the prior micro-trend**
* No candle close = no structure break

15M pullback definition:

* Price retraces **against the 4H bias**
* Retrace must be **at least 38.2%** of the active 4H leg
* Structure must be within **0% – 100%** of the active 4H leg

### If 4H bias is SELL:

* Price must **first close into premium** at **50% or 70%** (config)
* 15M **CHoCH** must occur **after** that premium close
* 15M **CHoCH** **must happen in the top 50%**
* Valid zone = **50% – 100%** of the leg
* Strong zone = **70% – 100%**
* Stronger signal if:

  * Happens at **70% or higher**
  * Happens **after a liquidity sweep of highs**

### If 4H bias is BUY:

* Price must **first close into discount** at **50% or 30%** (config)
* 15M **CHoCH** must occur **after** that discount close
* 15M **CHoCH** **must happen in the bottom 50%**
* Valid zone = **0% – 50%** of the leg
* Strong zone = **0% – 30%**
* Stronger signal if:

  * Happens at **30% or lower**
  * Happens **after a liquidity sweep of lows**

If 15M structure happens in mid-range:

* **Ignore it completely**
* Mid-range (invalid) = **38.2% – 50%** for SELL bias, **50% – 61.8%** for BUY bias

15M does **not** enter trades.

---

## 6. 5M RULES — TRIGGER (CHoCH)

5M is the **execution trigger** (default), but it must respect 15M.

Rules:

* 5M CHoCH must be **in the same direction as 4H**
* It must happen **inside the 15M pullback**
* It must be within **0% – 100%** of the active 4H leg
* 5M CHoCH premium check is **optional** (config)
* Mid-pullback or late CHoCH → **no trade**
* CHoCH must occur after a pullback, not in consolidation
* CHoCH candle range filter is **optional** (config)

No CHoCH → no trade.

### Optional 1M entry

If enabled:

* 1M CHoCH must occur **after** the 5M CHoCH
* It must be **in the same direction as 4H**
* It must happen **inside the 15M pullback**
* 1M CHoCH premium check is **optional** (config)
* SL is based on the chosen entry timeframe (1M or 5M)
* CHoCH candle range filter is **optional** (config)

---

## 7. LIQUIDITY SWEEP RULE (QUALITY FILTER)

Liquidity sweep is **not optional context** — it ranks signal quality.

Liquidity sweep definition (GU-specific):

* Price wicks beyond a prior equal/obvious high/low
* Wick exceeds that level by **>= 3 pips**
* Candle closes back inside the prior range

Signal strength ranking:

1. CHoCH after liquidity sweep at 70%+
2. CHoCH in correct 50%
3. Everything else → weak / ignore

The system prefers quality, not quantity.

---

## 8. STOP LOSS RULE (NON-NEGOTIABLE)

Stop loss must:

* Be placed **beyond the entry CHoCH** (1M or 5M)

  * SELL → above CHoCH high
  * BUY → below CHoCH low

If SL cannot be placed cleanly:

* No trade

No widening stops. Ever.

---

## 8B. RISK & POSITION SIZE

Risk sizing is required:

* Fixed risk per trade (config)
* Account balance available (or override set)
* Position size within min/max lot bounds

If sizing is invalid → **NO TRADE**

---

## 9. TAKE PROFIT RULES

Each trade uses **one plan only**.

### TP Plan A — Partial + 4H swing extreme

* TP1: **50% of the active 4H leg**
* TP2: **4H LL** (SELL) or **4H HH** (BUY)

If the plan is not valid relative to entry:

* No trade

---

## 10. SESSION RULE

Trades are allowed only during:

* **London session** (UTC 07:00 – 11:00)
* Early NY continuation (UTC 12:00 – 14:00, enabled by default)

Early NY can be disabled later (configurable).

Outside these hours:

* System does nothing

Early NY continuation is allowed only if:

* 4H bias was already established during London

NY is continuation only, not reversal hunting.

UTC is enforced (no manual DST logic).

No volatility filter is needed because:

* Session + location already handles it

---

## 11. SIGNAL DECISION RULE (FINAL GATE)

A trade signal is allowed **only if ALL are true**:

1. 4H bias is clear
2. 15M structure is in correct 50%
3. 5M CHoCH confirms direction
4. Location is near extreme (not mid-range)
5. SL is beyond CHoCH
6. TP plan is defined

If **any one fails** → **NO TRADE**

---

## 12. OUTPUT EXPECTATION

The system outputs only **one structured record** with these top-level fields:

* decision: TRADE / NO_TRADE
* timestamp_utc: ISO 8601 (UTC)
* pair: GU
* session: LONDON / NY
* direction: BUY / SELL (required if TRADE, optional if NO_TRADE)
* entry: short explanation string OR price
* stop_loss: price (required if TRADE)
* take_profit: PLAN_A
* tp1_price: price (required if TRADE)
* tp2_price: price (required if TRADE)
* spread_pips: number (optional)
* choc_range_pips: number (optional)
* stop_distance_pips: number (required if TRADE)
* account_balance: number (required if risk management enabled)
* risk_amount: number (required if risk management enabled)
* position_size_lots: number (required if risk management enabled)
* rules_passed: array of rule identifiers
* rules_failed: array of rule identifiers (empty if TRADE, non-empty if NO_TRADE)

No guessing. No freestyle.
No extra fields, commentary, or confidence language.

---

## 13. WHAT THIS SYSTEM IS NOT

* Not a scalper
* Not a prediction engine
* Not a high-frequency trader
* Not emotional
* Not reactive

It waits. It filters. It strikes only when location + structure align.

---

## 14. THE GOAL (ONE SENTENCE)

**Only trade GU when structure breaks at the correct part of the higher-timeframe leg during London session — everything else is ignored.**

---

This file is now your **north star**.
Every line of code must map back to something written here.
