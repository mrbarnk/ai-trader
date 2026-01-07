Perfect. Below is a **STRICT AI CHECKLIST**, converted **directly** from your Version 1 goal file.

This checklist is **binary**.
No interpretation.
No flexibility.
If any step fails → **NO TRADE**.

---

# VERSION 1 — STRICT AI TRADE CHECKLIST (LOCKED)

The AI must evaluate the following **in order**.
It is not allowed to skip steps.

---

## STEP 0 — TIMEFRAME LOCK

* **1M is optional** (config toggle)
* All structure checks use **completed candles only**

If any live/incomplete candle is used → **NO TRADE**
If 1M is enabled but data is missing → **NO TRADE**

---

## STEP 1 — ASSET & SESSION CHECK

* Asset is **GBPUSD (GU)** (accept broker variants: GBPUSD, GBPUSDm, GBPUSD.r) → YES / NO
* Broker timestamps are converted to **UTC** → YES / NO
* Current UTC time is within **London session (07:00–11:00)** or **early NY (12:00–14:00)** → YES / NO
* If in early NY, 4H bias must have been **established during London** → YES / NO
* Early NY is **continuation only**, not reversal hunting → YES / NO
* Spread filter (optional): spread must be **<= max pips** (config) → YES / NO

If **NO** to any → **NO TRADE**

---

## STEP 2 — 4H DIRECTION CHECK (BIAS)

Answer **one**:

* BUY ONLY
* SELL ONLY
* NO TRADE

Rules:

* Bias must be clear
* Bias must be based on structure
* Bias is checked **only on a new 4H candle close**
* If bias is unclear → NO TRADE
* Bias is clear only if ALL are true:

  * At least one confirmed BOS in one direction
  * Last valid swing high/low is respected
  * No opposite BOS after the last impulse
* Bias is invalidated if:

  * An opposite-direction BOS occurs
  * OR price closes beyond the last valid swing that defined the bias

If result = **NO TRADE** → stop here

---

## STEP 3 — 4H ACTIVE LEG IDENTIFICATION

* Identify the current 4H impulse leg that caused the last BOS
* Define:

  * Top 50% = Premium
  * Bottom 50% = Discount
* Leg boundaries:

  * Start = swing low (bullish) or swing high (bearish)
  * End = impulse high/low that caused BOS

This step **must succeed** before continuing.

If active leg cannot be identified → **NO TRADE**

---

## STEP 4 — 15M LOCATION CHECK (SETUP VALIDITY)

Structure definitions (15M and 5M):

* Swing high/low requires **at least 2 candles on each side**
* BOS = candle **closes beyond** prior swing high/low (wick-only breaks do not count)
* CHoCH = BOS **against the prior micro-trend**
* No candle close = no structure break

15M pullback definition:

* Price retraces **against the 4H bias**
* Retrace must be **at least 38.2%** of the active 4H leg

### If 4H bias = SELL:

* Did price **first close into premium** at **50% or 70%** (config)? → YES / NO
* Did 15M **CHoCH** occur **after** that premium close? → YES / NO
* Did 15M **CHoCH** occur in **top 50% of the 4H leg**? → YES / NO
* Structure must be within **0% – 100%** of the active 4H leg
* Valid zone = **50% – 100%**
* Strong zone = **70% – 100%**
* Mid-range (invalid) = **38.2% – 50%**

### If 4H bias = BUY:

* Did price **first close into discount** at **50% or 30%** (config)? → YES / NO
* Did 15M **CHoCH** occur **after** that discount close? → YES / NO
* Did 15M **CHoCH** occur in **bottom 50% of the 4H leg**? → YES / NO
* Structure must be within **0% – 100%** of the active 4H leg
* Valid zone = **0% – 50%**
* Strong zone = **0% – 30%**
* Mid-range (invalid) = **50% – 61.8%**

If **NO** → **NO TRADE**

---

## STEP 5 — 15M QUALITY BOOST (NOT REQUIRED, BUT RANKED)

Check and record:

* Did structure occur at **70%+ (SELL)** or **30% or lower (BUY)**? → YES / NO
* Did a **liquidity sweep** occur before the structure? → YES / NO

Liquidity sweep definition (GU-specific):

* Price wicks beyond a prior equal/obvious high/low
* Wick exceeds that level by **>= 3 pips**
* Candle closes back inside the prior range

These do **not** override Step 4, but increase confidence.

---

## STEP 6 — 5M TRIGGER CHECK (CHoCH)

* Did a **5M CHoCH** occur **after** the 15M CHoCH?
* Is the CHoCH **in the same direction as 4H bias**?
* Did it occur **inside the 15M pullback**?
* It must be within **0% – 100%** of the active 4H leg
* 5M CHoCH premium check is **optional** (config)
* Did it occur after a pullback (not in consolidation)?
* CHoCH candle range filter (optional): **>= min pips** (config)

If **any NO** → **NO TRADE**

### Optional 1M entry (config)

If enabled:

* Did a **1M CHoCH** occur **after** the 5M CHoCH?
* Is the CHoCH **in the same direction as 4H bias**?
* Did it occur **inside the 15M pullback**?
* It must be within **0% – 100%** of the active 4H leg
* 1M CHoCH premium check is **optional** (config)
* Did it occur after a pullback (not in consolidation)?
* CHoCH candle range filter (optional): **>= min pips** (config)

If **any NO** → **NO TRADE**

---

## STEP 7 — STOP LOSS VALIDITY CHECK

* Can SL be placed **beyond the entry CHoCH** (1M or 5M)?

  * SELL → SL above CHoCH high
  * BUY → SL below CHoCH low

If SL placement is unclear or invalid → **NO TRADE**
No widening stops. Ever.

---

## STEP 7B — RISK & POSITION SIZE (REQUIRED)

* Risk per trade is fixed (config) → YES / NO
* Account balance available (or override set) → YES / NO
* Position size is valid within min/max lot bounds → YES / NO

If **any NO** → **NO TRADE**

---

## STEP 8 — TAKE PROFIT PLAN CHECK

AI must select **one**:

### TP PLAN A (ONLY)

* TP1 = **50% of the active 4H leg**
* TP2 = **4H LL** (SELL) or **4H HH** (BUY)

If the plan is invalid relative to entry → **NO TRADE**

---

## STEP 9 — FINAL CONSISTENCY CHECK

Confirm:

* Direction aligns across all TFs
* Location is not mid-range
* No rule was bypassed

If consistent → **TRADE ALLOWED**
If inconsistent → **NO TRADE**

---

## STEP 10 — OUTPUT FORMAT (MANDATORY)

AI output **must** be:

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

No extra commentary.
No predictions.
No confidence language outside the checklist.
No extra fields.

---

## OVERRIDE RULE (ABSOLUTE)

If **any single rule fails**, the AI **must return NO TRADE**, even if all others pass.

---

## ONE-LINE SYSTEM LAW

**Structure is only tradable when it breaks at the correct part of the higher-timeframe leg during London session.**
