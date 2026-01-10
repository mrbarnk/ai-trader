from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Literal
from enum import Enum

from ...structure import BreakEvent, BreakDirection, find_breaks, find_swings


class OrderBlockType(Enum):
    """Different types of order blocks for better categorization"""
    STRUCTURE_BREAK = "structure_break"     # Original logic - last opposite candle before break
    FAILED_BREAKOUT = "failed_breakout"     # Failed attempt to break a level
    VOLUME_ZONE = "volume_zone"             # High volume rejection area
    RETEST_ZONE = "retest_zone"            # Area that was retested multiple times
    SWING_FAILURE = "swing_failure"         # Failed swing high/low formation


class OrderBlockQuality(Enum):
    """Quality rating for order blocks"""
    PREMIUM = "premium"      # Highest quality - multiple confirmations
    HIGH = "high"           # High quality - good confirmation  
    MEDIUM = "medium"       # Medium quality - basic confirmation
    LOW = "low"            # Lower quality - minimal confirmation


@dataclass
class EnhancedOrderBlock:
    """Enhanced Order Block with more detailed information"""
    high: float
    low: float
    created_time: datetime
    direction: Literal["BUY", "SELL"]  # Direction to trade FROM this OB
    target_price: float | None = None
    
    # Enhanced attributes
    ob_type: OrderBlockType = OrderBlockType.STRUCTURE_BREAK
    quality: OrderBlockQuality = OrderBlockQuality.MEDIUM
    strength_score: float = 0.0  # 0-100 scoring system
    touch_count: int = 0
    last_touch_time: datetime | None = None
    invalidation_price: float | None = None
    source_break_event: BreakEvent | None = None
    
    # State tracking
    traded: bool = False
    active: bool = True
    
    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def age_hours(self) -> float:
        """Age of order block in hours"""
        return (datetime.utcnow() - self.created_time).total_seconds() / 3600
    
    @property
    def is_fresh(self) -> bool:
        """Check if order block is still fresh (< 24 hours)"""
        return self.age_hours < 24
    
    @property
    def is_untested(self) -> bool:
        """Check if order block hasn't been touched yet"""
        return self.touch_count == 0


class EnhancedOrderBlockDetector:
    """Enhanced order block detection system"""
    
    def __init__(self):
        self.detected_blocks: list[EnhancedOrderBlock] = []
        
    def find_all_order_blocks(
        self,
        candles: list,
        events: list[BreakEvent],
        direction: Literal["BUY", "SELL"],
        active_leg,
        current_time: datetime,
        lookback_hours: int = 72  # Look back 72 hours for order blocks
    ) -> list[EnhancedOrderBlock]:
        """
        Enhanced order block detection - finds multiple quality order blocks
        
        Args:
            candles: Price candle data
            events: Structure break events 
            direction: Trading direction bias
            active_leg: Current 4H active leg
            current_time: Current time for filtering
            lookback_hours: How far back to look for order blocks
        """
        order_blocks = []
        cutoff_time = current_time - timedelta(hours=lookback_hours)
        desired_dir = "bear" if direction == "SELL" else "bull"
        
        # 1. STRUCTURE BREAK ORDER BLOCKS (Enhanced original logic)
        structure_obs = self._find_structure_break_obs(
            candles, events, desired_dir, cutoff_time, active_leg
        )
        order_blocks.extend(structure_obs)
        
        # 2. FAILED BREAKOUT ORDER BLOCKS  
        failed_breakout_obs = self._find_failed_breakout_obs(
            candles, events, desired_dir, cutoff_time, active_leg
        )
        order_blocks.extend(failed_breakout_obs)
        
        # 3. SWING FAILURE ORDER BLOCKS
        swing_failure_obs = self._find_swing_failure_obs(
            candles, events, desired_dir, cutoff_time, active_leg
        )
        order_blocks.extend(swing_failure_obs)
        
        # 4. RETEST ZONE ORDER BLOCKS
        retest_obs = self._find_retest_zone_obs(
            candles, events, desired_dir, cutoff_time, active_leg
        )
        order_blocks.extend(retest_obs)
        
        # 5. Filter and score order blocks
        filtered_obs = self._filter_and_score_order_blocks(
            order_blocks, active_leg, direction
        )
        
        # 6. Remove overlapping order blocks (keep highest quality)
        final_obs = self._remove_overlapping_blocks(filtered_obs)
        
        return final_obs
    
    def _find_structure_break_obs(
        self,
        candles: list,
        events: list[BreakEvent], 
        desired_dir: str,
        cutoff_time: datetime,
        active_leg
    ) -> list[EnhancedOrderBlock]:
        """Enhanced structure break order block detection"""
        order_blocks = []
        
        for event in events:
            if event.time_utc < cutoff_time:
                continue
            if event.direction != desired_dir:
                continue
            if event.event_type not in ("CHoCH", "BOS"):
                continue
                
            # Check if within active 4H leg
            if active_leg:
                structure_pos = active_leg.position(event.close_price)
                if structure_pos is None or structure_pos < 0 or structure_pos > 1:
                    continue
            
            # Find order block with enhanced logic
            ob_zone = self._find_enhanced_order_block(candles, event, desired_dir)
            if not ob_zone:
                continue
                
            ob_high, ob_low, quality_score = ob_zone
            
            # Determine quality based on various factors
            quality = self._determine_ob_quality(quality_score, event, candles)
            
            direction = "BUY" if desired_dir == "bull" else "SELL"
            
            order_block = EnhancedOrderBlock(
                high=ob_high,
                low=ob_low,
                created_time=event.time_utc,
                direction=direction,
                target_price=event.break_level,
                ob_type=OrderBlockType.STRUCTURE_BREAK,
                quality=quality,
                strength_score=quality_score,
                source_break_event=event,
                invalidation_price=self._calculate_invalidation_price(
                    ob_high, ob_low, direction
                )
            )
            
            order_blocks.append(order_block)
        
        return order_blocks
    
    def _find_enhanced_order_block(
        self,
        candles: list,
        break_event: BreakEvent,
        desired_dir: str
    ) -> tuple[float, float, float] | None:
        """
        Enhanced order block detection with quality scoring
        
        Returns: (ob_high, ob_low, quality_score) or None
        """
        if break_event.index < 5:  # Need enough history
            return None
        
        break_index = break_event.index
        search_start = max(0, break_index - 20)  # Expanded search window
        
        candidates = []
        
        # Look for multiple potential order blocks
        for i in range(break_index - 1, search_start - 1, -1):
            candle = candles[i]
            
            # Check if this is an opposite candle
            is_opposite = False
            if desired_dir == "bear":
                is_opposite = candle.close > candle.open  # Bullish candle
            else:
                is_opposite = candle.close < candle.open  # Bearish candle
            
            if not is_opposite:
                continue
                
            # Calculate quality score for this candle
            quality_score = self._score_order_block_candle(
                candle, candles, i, break_event, desired_dir
            )
            
            candidates.append({
                'high': candle.high,
                'low': candle.low,
                'index': i,
                'quality_score': quality_score,
                'candle': candle
            })
        
        if not candidates:
            return None
        
        # Sort by quality score and distance to break (prefer closer, higher quality)
        candidates.sort(key=lambda x: (
            x['quality_score'],  # Primary: quality
            -(break_index - x['index'])  # Secondary: proximity (closer = better)
        ), reverse=True)
        
        best = candidates[0]
        
        # Minimum quality threshold
        if best['quality_score'] < 30:  # Minimum 30/100 score
            return None
            
        return best['high'], best['low'], best['quality_score']
    
    def _score_order_block_candle(
        self,
        candle,
        candles: list, 
        candle_index: int,
        break_event: BreakEvent,
        desired_dir: str
    ) -> float:
        """
        Score an order block candidate (0-100)
        
        Factors:
        - Candle size/range (bigger = better institutions)
        - Body vs wick ratio (strong body = conviction)
        - Volume characteristics (if available)
        - Position relative to recent price action
        - Proximity to break event
        """
        score = 0.0
        
        # 1. Candle range score (0-25 points)
        candle_range = candle.high - candle.low
        if candle_range > 0:
            # Compare to recent average range
            recent_ranges = []
            for i in range(max(0, candle_index - 10), candle_index):
                if i < len(candles):
                    recent_ranges.append(candles[i].high - candles[i].low)
            
            if recent_ranges:
                avg_range = sum(recent_ranges) / len(recent_ranges)
                range_ratio = candle_range / avg_range if avg_range > 0 else 1
                score += min(25, range_ratio * 12.5)  # Cap at 25 points
        
        # 2. Body strength score (0-20 points)
        body_size = abs(candle.close - candle.open)
        if candle_range > 0:
            body_ratio = body_size / candle_range
            score += body_ratio * 20  # Strong body = institutional conviction
        
        # 3. Wick characteristics (0-15 points)
        if desired_dir == "bear":  # Looking for bullish OB
            upper_wick = candle.high - max(candle.open, candle.close)
            lower_wick = min(candle.open, candle.close) - candle.low
            # For bullish OB, small upper wick is better (no selling pressure at top)
            if candle_range > 0:
                wick_score = (1 - (upper_wick / candle_range)) * 15
                score += max(0, wick_score)
        else:  # Looking for bearish OB
            upper_wick = candle.high - max(candle.open, candle.close)
            lower_wick = min(candle.open, candle.close) - candle.low
            # For bearish OB, small lower wick is better (no buying pressure at bottom)
            if candle_range > 0:
                wick_score = (1 - (lower_wick / candle_range)) * 15
                score += max(0, wick_score)
        
        # 4. Proximity to break (0-15 points)
        distance_to_break = break_event.index - candle_index
        if distance_to_break > 0:
            # Closer to break = more relevant (but not immediate previous candle)
            if distance_to_break == 1:
                score += 15  # Perfect distance
            elif distance_to_break <= 3:
                score += 12  # Very close
            elif distance_to_break <= 5:
                score += 8   # Close
            else:
                score += max(0, 15 - distance_to_break)  # Decreasing relevance
        
        # 5. Price level significance (0-25 points)
        # Check if this candle represents a significant level
        level_score = self._score_price_level_significance(
            candle, candles, candle_index, desired_dir
        )
        score += level_score
        
        return min(100, score)  # Cap at 100
    
    def _score_price_level_significance(
        self,
        candle,
        candles: list,
        candle_index: int, 
        desired_dir: str
    ) -> float:
        """Score how significant this price level is (0-25 points)"""
        score = 0.0
        
        # Check for multiple touches at this level
        level_price = candle.close if desired_dir == "bear" else candle.open
        tolerance = (candle.high - candle.low) * 0.5  # 50% of candle range tolerance
        
        touches = 0
        lookback_start = max(0, candle_index - 50)  # Look back 50 candles
        lookback_end = min(len(candles), candle_index + 20)  # Look ahead 20 candles
        
        for i in range(lookback_start, lookback_end):
            if i == candle_index:
                continue
            test_candle = candles[i]
            
            # Check if this candle touched our level
            if (test_candle.low <= level_price + tolerance and 
                test_candle.high >= level_price - tolerance):
                touches += 1
        
        # More touches = more significant level
        if touches >= 5:
            score += 25
        elif touches >= 3:
            score += 20
        elif touches >= 2:
            score += 15
        elif touches >= 1:
            score += 10
        
        return score
    
    def _find_failed_breakout_obs(
        self,
        candles: list,
        events: list[BreakEvent],
        desired_dir: str,
        cutoff_time: datetime,
        active_leg
    ) -> list[EnhancedOrderBlock]:
        """Find order blocks from failed breakout attempts"""
        order_blocks = []
        
        # Look for opposite direction breaks that failed quickly
        opposite_dir = "bull" if desired_dir == "bear" else "bear"
        
        for event in events:
            if event.time_utc < cutoff_time:
                continue
            if event.direction != opposite_dir:
                continue
            if event.event_type not in ("CHoCH", "BOS"):
                continue
                
            # Check if this break was quickly invalidated (failed breakout)
            failed_quickly = self._check_failed_breakout(candles, event, desired_dir)
            if not failed_quickly:
                continue
                
            # The failed breakout zone becomes an order block
            if event.index >= len(candles):
                continue
                
            break_candle = candles[event.index]
            
            # Create order block from the failed breakout zone
            if desired_dir == "bear":  # We want SELL, so failed bull break = bearish OB
                ob_high = break_candle.high
                ob_low = min(break_candle.low, break_candle.open, break_candle.close)
                direction = "SELL"
            else:  # We want BUY, so failed bear break = bullish OB  
                ob_high = max(break_candle.high, break_candle.open, break_candle.close)
                ob_low = break_candle.low
                direction = "BUY"
            
            order_block = EnhancedOrderBlock(
                high=ob_high,
                low=ob_low,
                created_time=event.time_utc,
                direction=direction,
                target_price=event.defining_swing_price,
                ob_type=OrderBlockType.FAILED_BREAKOUT,
                quality=OrderBlockQuality.HIGH,  # Failed breakouts often high quality
                strength_score=75.0,  # Good score for failed breakouts
                source_break_event=event
            )
            
            order_blocks.append(order_block)
        
        return order_blocks
    
    def _check_failed_breakout(
        self,
        candles: list,
        break_event: BreakEvent,
        desired_dir: str
    ) -> bool:
        """Check if a breakout failed quickly (within 3-10 candles)"""
        if break_event.defining_swing_price is None:
            return False
            
        break_index = break_event.index
        check_end = min(len(candles), break_index + 10)  # Check next 10 candles
        
        # Must fail within reasonable time but not immediately
        failure_count = 0
        
        for i in range(break_index + 3, check_end):  # Skip first 2 candles
            candle = candles[i]
            
            # Check if price came back to invalidate the break
            if break_event.direction == "bull":
                # Bull break failed if price goes back below break level
                if candle.close < break_event.defining_swing_price:
                    failure_count += 1
            else:
                # Bear break failed if price goes back above break level  
                if candle.close > break_event.defining_swing_price:
                    failure_count += 1
        
        # Consider it failed if 2+ candles closed back inside
        return failure_count >= 2
    
    def _find_swing_failure_obs(
        self,
        candles: list,
        events: list[BreakEvent],
        desired_dir: str,
        cutoff_time: datetime,
        active_leg
    ) -> list[EnhancedOrderBlock]:
        """Find order blocks from swing failure patterns"""
        order_blocks = []
        
        # Look for failed swing formations
        for i, event in enumerate(events):
            if event.time_utc < cutoff_time:
                continue
            if event.direction != desired_dir:
                continue
                
            # Look for pattern where price tried to make new swing but failed
            swing_failed = self._check_swing_failure_pattern(
                candles, events, i, desired_dir
            )
            
            if swing_failed:
                failure_zone = self._identify_swing_failure_zone(
                    candles, event, desired_dir
                )
                
                if failure_zone:
                    ob_high, ob_low = failure_zone
                    direction = "BUY" if desired_dir == "bull" else "SELL"
                    
                    order_block = EnhancedOrderBlock(
                        high=ob_high,
                        low=ob_low,
                        created_time=event.time_utc,
                        direction=direction,
                        ob_type=OrderBlockType.SWING_FAILURE,
                        quality=OrderBlockQuality.HIGH,
                        strength_score=70.0
                    )
                    
                    order_blocks.append(order_block)
        
        return order_blocks
    
    def _check_swing_failure_pattern(
        self,
        candles: list,
        events: list[BreakEvent],
        event_index: int,
        desired_dir: str
    ) -> bool:
        """Check if this represents a failed swing formation"""
        # Implementation for swing failure detection
        # This would look for patterns like:
        # - Higher high that gets taken out immediately
        # - Lower low that gets reclaimed quickly
        # - Double top/bottom failures
        
        return False  # Placeholder - implement based on your specific criteria
    
    def _identify_swing_failure_zone(
        self,
        candles: list,
        event: BreakEvent,
        desired_dir: str
    ) -> tuple[float, float] | None:
        """Identify the price zone where the swing failure occurred"""
        # Implementation to identify the specific zone
        return None  # Placeholder
    
    def _find_retest_zone_obs(
        self,
        candles: list,
        events: list[BreakEvent],
        desired_dir: str,
        cutoff_time: datetime,
        active_leg
    ) -> list[EnhancedOrderBlock]:
        """Find order blocks from frequently retested zones"""
        order_blocks = []
        
        # This would identify zones that have been tested multiple times
        # and could act as strong support/resistance
        
        # Implementation would:
        # 1. Identify price levels with multiple touches
        # 2. Check for rejection patterns at these levels
        # 3. Create order blocks around strong rejection zones
        
        return order_blocks  # Placeholder
    
    def _determine_ob_quality(
        self,
        quality_score: float,
        event: BreakEvent,
        candles: list
    ) -> OrderBlockQuality:
        """Determine order block quality based on score and other factors"""
        if quality_score >= 80:
            return OrderBlockQuality.PREMIUM
        elif quality_score >= 65:
            return OrderBlockQuality.HIGH  
        elif quality_score >= 45:
            return OrderBlockQuality.MEDIUM
        else:
            return OrderBlockQuality.LOW
    
    def _calculate_invalidation_price(
        self,
        ob_high: float,
        ob_low: float,
        direction: str
    ) -> float:
        """Calculate price level that would invalidate this order block"""
        if direction == "BUY":
            # Bullish OB invalidated if price breaks below the low
            return ob_low
        else:
            # Bearish OB invalidated if price breaks above the high
            return ob_high
    
    def _filter_and_score_order_blocks(
        self,
        order_blocks: list[EnhancedOrderBlock],
        active_leg,
        direction: str
    ) -> list[EnhancedOrderBlock]:
        """Filter and score order blocks for final selection"""
        filtered = []
        
        for ob in order_blocks:
            # 1. Check if OB is within reasonable bounds of 4H leg
            if active_leg:
                ob_mid_pos = active_leg.position(ob.mid)
                if ob_mid_pos is None or ob_mid_pos < -0.1 or ob_mid_pos > 1.1:
                    continue
            
            # 2. Check minimum size requirements
            min_range_pips = 5  # Minimum 5 pip order block
            pip_size = 0.0001  # Adjust based on your config
            if ob.range < min_range_pips * pip_size:
                continue
            
            # 3. Age filter - prefer fresher order blocks but allow some older ones
            if ob.age_hours > 168:  # Older than 1 week
                continue
            
            # 4. Boost score for better positioned OBs within the leg
            if active_leg and direction == "SELL":
                # For SELL, prefer OBs in premium zone (61.8%+)
                ob_pos = active_leg.position(ob.mid)
                if ob_pos and ob_pos >= 0.618:
                    ob.strength_score *= 1.2
            elif active_leg and direction == "BUY":
                # For BUY, prefer OBs in discount zone (38.2%-)  
                ob_pos = active_leg.position(ob.mid)
                if ob_pos and ob_pos <= 0.382:
                    ob.strength_score *= 1.2
            
            filtered.append(ob)
        
        # Sort by quality and strength score
        filtered.sort(key=lambda x: (x.quality.value, x.strength_score), reverse=True)
        
        return filtered
    
    def _remove_overlapping_blocks(
        self,
        order_blocks: list[EnhancedOrderBlock]
    ) -> list[EnhancedOrderBlock]:
        """Remove overlapping order blocks, keeping the highest quality ones"""
        if not order_blocks:
            return []
        
        final_blocks = []
        
        for ob in order_blocks:
            # Check if this OB overlaps significantly with any existing final block
            overlaps = False
            
            for existing in final_blocks:
                overlap_amount = self._calculate_overlap(ob, existing)
                if overlap_amount > 0.5:  # More than 50% overlap
                    overlaps = True
                    break
            
            if not overlaps:
                final_blocks.append(ob)
        
        return final_blocks
    
    def _calculate_overlap(
        self,
        ob1: EnhancedOrderBlock,
        ob2: EnhancedOrderBlock
    ) -> float:
        """Calculate overlap percentage between two order blocks"""
        # Find overlapping range
        overlap_high = min(ob1.high, ob2.high)
        overlap_low = max(ob1.low, ob2.low)
        
        if overlap_high <= overlap_low:
            return 0.0  # No overlap
        
        overlap_range = overlap_high - overlap_low
        smaller_range = min(ob1.range, ob2.range)
        
        if smaller_range == 0:
            return 0.0
        
        return overlap_range / smaller_range


# Integration with existing SignalEngine class
def integrate_enhanced_ob_detection(signal_engine_class):
    """
    Integration function to add enhanced OB detection to your existing SignalEngine
    """
    
    def _find_all_order_blocks_enhanced(self, candles_15m, events_15m, direction, active_leg, reference_time):
        """Replace the existing OB logic with enhanced detection"""
        
        detector = EnhancedOrderBlockDetector()
        
        # Find all order blocks using enhanced detection
        enhanced_obs = detector.find_all_order_blocks(
            candles=candles_15m,
            events=events_15m,
            direction=direction,
            active_leg=active_leg,
            current_time=reference_time,
            lookback_hours=72  # 3 days lookback
        )
        
        # Convert to your existing OrderBlock format if needed
        converted_obs = []
        for eob in enhanced_obs:
            # Convert EnhancedOrderBlock to your OrderBlock format
            from your_existing_module import OrderBlock  # Replace with actual import
            
            ob = OrderBlock(
                high=eob.high,
                low=eob.low,
                created_time=eob.created_time,
                direction=eob.direction,
                target_price=eob.target_price,
                traded=eob.traded
            )
            converted_obs.append(ob)
        
        return converted_obs
    
    # Add the method to your SignalEngine class
    signal_engine_class._find_all_order_blocks_enhanced = _find_all_order_blocks_enhanced
    
    return signal_engine_class


# Usage example:
"""
# In your main engine file, replace the STEP 4 order block detection with:

# STEP 4 — ENHANCED 15M ORDER BLOCK IDENTIFICATION
detector = EnhancedOrderBlockDetector()

# Clear existing OBs and rebuild with enhanced detection
self.state.active_order_blocks.clear()

enhanced_obs = detector.find_all_order_blocks(
    candles=candles_15m,
    events=events_15m, 
    direction=direction,
    active_leg=active_leg,
    current_time=now_utc,
    lookback_hours=72
)

# Convert to your existing OrderBlock format and add to state
for eob in enhanced_obs:
    ob = OrderBlock(
        high=eob.high,
        low=eob.low,
        created_time=eob.created_time,
        direction=eob.direction,
        target_price=eob.target_price,
        traded=False
    )
    self.state.active_order_blocks.append(ob)

# Rest of your existing logic continues...
"""