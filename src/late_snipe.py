#!/usr/bin/env python3
"""
Late Snipe Engine — Buy near-certain outcomes in the final seconds of a round.

Strategy:
  With <30 seconds left in a 5-minute round, if the crypto price is significantly
  above/below the start price, the probability of a reversal is very low.
  
  We buy the winning side at 95-99¢ as a taker. If it resolves to $1.00,
  we pocket 1-5¢ per share.

Math:
  - BTC 30-second volatility ≈ 0.06% (at ~60% annualized vol)
  - If BTC is 0.2% above start with 30s left:
    P(stays above) = Φ(0.2/0.06) ≈ 99.9%
  - Buying at 99¢ needs 99% win rate → very profitable
  
  Key: only snipe when probability SIGNIFICANTLY exceeds the price.
  If P(win) = 96% and price = 95¢, that's barely breakeven.
  We require P(win) - price ≥ min_edge (default 2%).

Risks:
  - Flash crashes / fat tail events (CDF assumes normal, reality doesn't)
  - Competition from other bots at the ask
  - Tiny profit per trade (1-5¢ per share)
  
  Mitigated by:
  - Only sniping on BIG moves (0.15%+ from start)
  - Requiring significant edge above price
  - High frequency: every 5 min × 3 assets = 36 opportunities/hour
"""
import math
import time
from dataclasses import dataclass, field
from typing import Optional

from src.logger import get_logger
from src.orderbook import fetch_order_book

logger = get_logger("late_snipe")


@dataclass
class SnipeConfig:
    """Late snipe configuration."""
    enabled: bool = True
    max_seconds_remaining: int = 15       # Only snipe in final 15s
    min_seconds_remaining: int = 3        # Don't snipe with < 3s (might not fill)
    min_move_pct: float = 0.07            # ~$50 on BTC at $70k
    min_dollar_move: float = 50.0         # BTC must be $50+ from start price
    min_edge: float = 0.01                # P(win) must exceed price by this much
    max_buy_price: float = 0.99           # Never pay more than 99¢
    min_buy_price: float = 0.80           # Don't buy below 80¢
    order_size_dollars: float = 5.0       # $ per snipe
    max_concurrent: int = 2               # Max simultaneous snipes
    cooldown_per_window: float = 10.0     # Don't re-snipe same window within 10s


class LateSnipeEngine:
    """
    Buys near-certain outcomes in the final seconds of prediction rounds.
    
    Uses Binance real-time price to calculate P(outcome) via normal CDF,
    then buys at the ask if P(outcome) significantly exceeds the ask price.
    """

    def __init__(self, config, clob_manager, price_feed, window_tracker):
        self.config = config
        self.clob = clob_manager
        self.price_feed = price_feed
        self.window_tracker = window_tracker
        
        # Load snipe config from main config or use defaults
        self.snipe_config = SnipeConfig()
        if hasattr(config, 'snipe'):
            for attr in vars(self.snipe_config):
                if hasattr(config.snipe, attr):
                    setattr(self.snipe_config, attr, getattr(config.snipe, attr))
        
        # State
        self._sniped_windows: dict[str, float] = {}  # window_key → last_snipe_time
        self._active_snipes: dict[str, dict] = {}     # window_key → snipe info
        
        # Stats
        self.stats = {
            "opportunities": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "avg_entry_price": 0.0,
            "avg_probability": 0.0,
        }
        
        logger.info("🎯 Late snipe engine initialized (snipe window: %d-%ds, min move: %.2f%%)",
                     self.snipe_config.min_seconds_remaining,
                     self.snipe_config.max_seconds_remaining,
                     self.snipe_config.min_move_pct)

    async def check_opportunities(self):
        """Scan all active windows for late snipe opportunities."""
        if not self.snipe_config.enabled:
            return
            
        # Check concurrent limit
        active_count = len(self._active_snipes)
        if active_count >= self.snipe_config.max_concurrent:
            return
        
        now = time.time()
        
        # Resolve any expired snipes first
        self._resolve_expired()
        
        active_windows = self.window_tracker.get_active_windows()
        
        for window in active_windows:
            time_remaining = window.end_ts - now
            
            # Only consider windows in the snipe zone
            if time_remaining > self.snipe_config.max_seconds_remaining:
                continue
            if time_remaining < self.snipe_config.min_seconds_remaining:
                continue
                
            # Skip if already sniped this window recently
            last_snipe = self._sniped_windows.get(window.key, 0)
            if now - last_snipe < self.snipe_config.cooldown_per_window:
                continue
                
            # Skip if already have an active snipe on this window
            if window.key in self._active_snipes:
                continue
                
            # Need token IDs
            if not window.up_token_id or not window.down_token_id:
                continue
                
            # Need start price
            start_price = window.start_price
            if start_price <= 0:
                continue
            
            # Get current crypto price from Binance
            current_price = self.price_feed.get_price(window.asset)
            if current_price <= 0:
                continue
            
            # Calculate move from start
            pct_move = ((current_price - start_price) / start_price) * 100
            
            if abs(pct_move) < self.snipe_config.min_move_pct:
                continue

            # Dollar move filter — e.g. BTC must be $50+ from start
            dollar_move = abs(current_price - start_price)
            min_dollar = getattr(self.snipe_config, 'min_dollar_move', 0)
            if min_dollar > 0 and dollar_move < min_dollar:
                continue
            
            # Calculate probability using CDF
            probability = self._calc_probability(
                pct_move=abs(pct_move),
                time_remaining=time_remaining,
                asset=window.asset,
            )
            
            # Determine which side to buy
            if pct_move > 0:
                buy_side = "Up"
                token_id = window.up_token_id
            else:
                buy_side = "Down"
                token_id = window.down_token_id
            
            # Fetch order book to check ask price
            book = fetch_order_book(token_id)
            if not book or book.asks.best_price <= 0:
                continue
            
            ask_price = book.asks.best_price
            
            # Check price bounds
            if ask_price > self.snipe_config.max_buy_price:
                continue
            if ask_price < self.snipe_config.min_buy_price:
                continue
            
            # THE KEY CHECK: probability must significantly exceed price
            edge = probability - ask_price
            if edge < self.snipe_config.min_edge:
                logger.debug("Snipe skip %s %s: P=%.3f, ask=%.2f, edge=%.3f (need %.3f)",
                             window.asset, buy_side, probability, ask_price,
                             edge, self.snipe_config.min_edge)
                continue
            
            # Check depth — need enough shares at this price
            depth = book.asks.depth_at_price(ask_price + 0.02, side="ask")
            tokens_needed = self.snipe_config.order_size_dollars / ask_price
            if depth < tokens_needed * 0.5:
                logger.debug("Snipe skip %s: insufficient depth (%.0f at ask)", 
                             window.key, depth)
                continue
            
            # 🎯 SNIPE IT
            self.stats["opportunities"] += 1
            
            expected_profit = (1.0 - ask_price) * tokens_needed
            expected_loss = ask_price * tokens_needed
            ev = probability * expected_profit - (1 - probability) * expected_loss
            
            logger.info(
                "🎯 SNIPE: %s %s %s | %.1fs left | move=%+.3f%% | "
                "P(win)=%.1f%% | ask=%.0f¢ | edge=%.1f%% | EV=$%.3f",
                window.asset, window.timeframe, buy_side,
                time_remaining, pct_move,
                probability * 100, ask_price * 100,
                edge * 100, ev,
            )
            
            await self._execute_snipe(
                window=window,
                buy_side=buy_side,
                token_id=token_id,
                ask_price=ask_price,
                probability=probability,
                pct_move=pct_move,
                tokens=tokens_needed,
            )
    
    def _calc_probability(self, pct_move: float, time_remaining: float, asset: str) -> float:
        """
        Calculate P(price stays on current side) using normal CDF.
        
        With small time remaining, even moderate moves have very high probability
        of sticking because there's not enough time for a reversal.
        """
        # Get realized volatility
        state = self.price_feed.get_state(asset)
        vol = state.realized_volatility(60) if state else 0.0001
        vol = max(vol, 0.0001)
        
        # Annualize the per-tick vol
        # realized_volatility returns stdev of returns over lookback period
        # We need sigma for the remaining time
        vol_annualized = vol * math.sqrt(365.25 * 24 * 3600)
        vol_annualized = max(vol_annualized, 0.3)  # Floor
        
        # Sigma for remaining time
        t_yr = time_remaining / (365.25 * 24 * 3600)
        sigma = vol_annualized * math.sqrt(max(t_yr, 1e-12))
        
        # Move in decimal form
        move = pct_move / 100.0
        
        # Mean reversion is minimal with so little time left
        # (no adjustment needed — move is already realized)
        
        # CDF: P(stays above/below start)
        z = move / sigma
        
        try:
            from scipy.stats import norm
            prob = float(norm.cdf(z))
        except ImportError:
            # Tanh approximation of normal CDF
            prob = 0.5 * (1.0 + math.tanh(z * 0.7978845608))
        
        # Apply a small fat-tail discount (reality has more reversals than normal dist)
        # Discount by 1% to be conservative
        prob = prob * 0.99
        
        return max(0.5, min(0.99, prob))
    
    async def _execute_snipe(self, window, buy_side: str, token_id: str,
                              ask_price: float, probability: float,
                              pct_move: float, tokens: float):
        """Place the snipe buy order."""
        
        if self.config.paper_mode:
            order_id = f"snipe_paper_{window.key}"
            logger.info("📝 [PAPER] SNIPE BUY %s %s @ %.2f¢ (%.1f tokens, P=%.1f%%)",
                         buy_side, window.asset, ask_price * 100, tokens, probability * 100)
        elif self.config.dry_run:
            logger.info("🔍 [DRY RUN] Would SNIPE %s %s @ %.2f¢", 
                         buy_side, window.asset, ask_price * 100)
            return
        else:
            result = self.clob.place_order(
                token_id=token_id,
                side="BUY",
                price=ask_price,
                size=tokens,
            )
            if not result.get("success"):
                logger.error("❌ SNIPE failed: %s %s — %s", 
                             buy_side, window.asset, result.get("error"))
                return
            order_id = result["orderID"]
            logger.info("🎯 SNIPE PLACED: %s %s @ %.0f¢ → %s",
                         buy_side, window.asset, ask_price * 100, order_id)
        
        # Track it
        self._active_snipes[window.key] = {
            "order_id": order_id,
            "side": buy_side,
            "asset": window.asset,
            "token_id": token_id,
            "entry_price": ask_price,
            "tokens": tokens,
            "cost": ask_price * tokens,
            "probability": probability,
            "pct_move": pct_move,
            "window_end": window.end_ts,
            "start_price": window.start_price,
            "placed_at": time.time(),
        }
        self._sniped_windows[window.key] = time.time()
        self.stats["trades"] += 1
        
        # Update running averages
        n = self.stats["trades"]
        self.stats["avg_entry_price"] = (
            (self.stats["avg_entry_price"] * (n - 1) + ask_price) / n
        )
        self.stats["avg_probability"] = (
            (self.stats["avg_probability"] * (n - 1) + probability) / n
        )
    
    def _resolve_expired(self):
        """Resolve snipes whose windows have ended."""
        now = time.time()
        resolved = []
        
        for key, snipe in self._active_snipes.items():
            if now < snipe["window_end"] + 5:  # 5s grace period
                continue
            
            # Determine outcome from current crypto price vs start
            current_price = self.price_feed.get_price(snipe["asset"])
            if current_price <= 0:
                continue
            
            start_price = snipe["start_price"]
            won = False
            if snipe["side"] == "Up" and current_price >= start_price:
                won = True
            elif snipe["side"] == "Down" and current_price < start_price:
                won = True
            
            if won:
                # Token resolves to $1.00
                payout = snipe["tokens"] * 1.0
                # Taker fee on buy (~1%)
                buy_fee = snipe["cost"] * 0.01
                pnl = payout - snipe["cost"] - buy_fee
                self.stats["wins"] += 1
                logger.info("✅ SNIPE WIN %s %s: bought @ %.0f¢, P&L: +$%.3f",
                             snipe["asset"], snipe["side"],
                             snipe["entry_price"] * 100, pnl)
            else:
                # Token resolves to $0.00
                pnl = -snipe["cost"]
                self.stats["losses"] += 1
                logger.warning("❌ SNIPE LOSS %s %s: bought @ %.0f¢, P&L: -$%.3f (reversal!)",
                                snipe["asset"], snipe["side"],
                                snipe["entry_price"] * 100, abs(pnl))
            
            self.stats["total_pnl"] += pnl
            resolved.append(key)
        
        for key in resolved:
            del self._active_snipes[key]
    
    def get_stats(self) -> dict:
        """Get snipe stats for health reporting."""
        return {
            **self.stats,
            "active_snipes": len(self._active_snipes),
            "win_rate": (
                self.stats["wins"] / max(1, self.stats["wins"] + self.stats["losses"]) * 100
            ),
        }
