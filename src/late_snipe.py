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
import json
import math
import os
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
    min_buy_price: float = 0.90           # global fallback min price
    max_buy_price: float = 0.97
    maker_bid_offset: float = 0.01
    min_edge: float = 0.01
    entry_times: tuple = (15, 10, 6)      # legacy — unused
    min_shares: float = 5.0
    order_size_dollars: float = 0.0      # 0 = use min_shares (5 shares); >0 = dollar-based
    taker_seconds_remaining: int = 6
    snipe_window_seconds: int = 17
    cooldown_seconds: float = 4.0
    max_entries_per_window: int = 2       # HARD LIMIT: 2 entries per window maximum
    max_concurrent: int = 2               # HARD: 2 concurrent windows max
    min_seconds_remaining: int = 3
    # Per-asset min buy prices
    min_buy_price_btc: float = 0.90
    min_buy_price_eth: float = 0.90
    min_buy_price_eth_afterhours: float = 0.90
    # ETH move filter: only snipe if ETH moved >= this % from round open (0 = disabled)
    eth_pct_move_from_open: float = 0.0   # no move filter — 90c min is the signal
    # Per-asset move filters (0 = disabled; price floor handles filtering)
    min_dollar_move: float = 0.0
    dollar_move_btc: float = 0.0
    dollar_move_eth: float = 0.0
    dollar_move_sol: float = 0.0
    # Asset controls
    eth_24h: bool = True               # ETH runs 24/7
    btc_24h: bool = True               # BTC runs 24/7 (90c min + $50 move filter handles selectivity)
    disabled_assets: list = field(default_factory=lambda: ["SOL"])
    trading_start_hour_et: int = 6
    trading_end_hour_et: int = 20
    trading_end_minute_et: int = 30


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

        # Trades logger
        try:
            from src.trades_logger import TradesLogger
            self.trades_logger = TradesLogger(clob_manager)
        except Exception as e:
            logger.warning("Trades logger disabled: %s", e)
            self.trades_logger = None
        
        # State
        self._sniped_windows: dict[str, float] = {}  # window_key → last_snipe_time
        self._active_snipes: dict[str, dict] = {}     # window_key → snipe info
        self._bot_start_time = time.time()
        self._snipes_persist_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "active_snipes.json"
        )
        self._load_persisted_snipes()
        
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
        
        logger.info("🎯 Late snipe engine initialized | entries at %s s | min=%.0f¢ | %.0f shares/entry",
                     self.snipe_config.entry_times,
                     self.snipe_config.min_buy_price * 100,
                     self.snipe_config.min_shares)

    def _check_conditions(self, window, time_remaining: float):
        """
        Check all snipe entry conditions for a window.
        Strategy: check both token order books, buy whichever side is >= min_buy_price.
        Token price IS the signal — no start_price direction guessing needed.
        Returns (buy_side, token_id, ask_price, probability, pct_move, tokens) or None.
        """
        key = f"{window.asset} {window.timeframe}"

        if not window.up_token_id or not window.down_token_id:
            logger.debug("Snipe skip %s: no token IDs", key)
            return None

        # Per-asset dollar move filter
        asset_upper = window.asset.upper()
        disabled_assets = getattr(self.snipe_config, 'disabled_assets', [])
        if asset_upper in [a.upper() for a in disabled_assets]:
            logger.debug("Snipe skip %s: asset disabled", key)
            return None

        # Time of day filter
        in_golden   = getattr(self, '_in_golden_hours', True)
        in_overnight = getattr(self, '_in_overnight_hours', False)

        eth_24h = getattr(self.snipe_config, 'eth_24h', True)
        btc_24h = getattr(self.snipe_config, 'btc_24h', True)
        if not in_golden:
            if asset_upper == "BTC" and not btc_24h:
                logger.debug("Snipe skip %s: BTC outside golden hours", key)
                return None
            if asset_upper == "ETH" and not eth_24h and not in_overnight:
                logger.debug("Snipe skip %s: ETH outside golden/overnight hours", key)
                return None
            # Both 24/7: 90c min + move filters handle selectivity

        # Balance check — skip if insufficient USDC (ignore negative = API error)
        try:
            balance = self.clob.get_usdc_balance()
            min_balance = getattr(self.config.trading, 'min_usdc_balance', 3.0)
            if 0 <= balance < min_balance:
                logger.warning("Snipe skip %s: low balance $%.2f", key, balance)
                return None
        except Exception:
            pass  # Don't block on balance check failure

        move_key = f"dollar_move_{asset_upper.lower()}"
        min_dollar = getattr(self.snipe_config, move_key,
                     getattr(self.snipe_config, 'min_dollar_move', 0))

        if min_dollar > 0:
            start_price = window.start_price
            if start_price <= 0:
                logger.debug("Snipe skip %s: start_price not yet available (kline pending)", key)
                return None
            current_price = self.price_feed.get_price(window.asset)
            if current_price <= 0:
                logger.debug("Snipe skip %s: no Binance price", key)
                return None
            dollar_move = abs(current_price - start_price)
            if dollar_move < min_dollar:
                logger.debug("Snipe skip %s: $%.0f move < $%.0f min", key, dollar_move, min_dollar)
                return None

        # ETH 0.03% move-from-open filter — only snipe if price confirmed moving
        eth_pct_filter = getattr(self.snipe_config, 'eth_pct_move_from_open', 0.0)
        if asset_upper == "ETH" and eth_pct_filter > 0:
            start_price = window.start_price
            if start_price <= 0:
                logger.debug("Snipe skip %s: ETH start_price not available", key)
                return None
            current_price = self.price_feed.get_price(window.asset)
            if current_price <= 0:
                logger.debug("Snipe skip %s: ETH no Binance price", key)
                return None
            pct_from_open = abs(current_price - start_price) / start_price * 100
            if pct_from_open < eth_pct_filter:
                logger.debug("Snipe skip ETH %s: only %.4f%% from open (need %.4f%%)",
                             key, pct_from_open, eth_pct_filter)
                return None
            logger.debug("ETH move filter pass %s: %.4f%% from open", key, pct_from_open)

        # Check both sides — buy whichever is at min_buy_price or higher
        candidates = [
            ("Up", window.up_token_id),
            ("Down", window.down_token_id),
        ]

        # Per-asset min buy price (higher threshold after golden hours for ETH)
        price_key = f"min_buy_price_{asset_upper.lower()}"
        min_price = getattr(self.snipe_config, price_key,
                    getattr(self.snipe_config, 'min_buy_price', 0.79))
        if not in_golden and asset_upper == "ETH":
            after_hours_min = getattr(self.snipe_config, 'min_buy_price_eth_afterhours', 0.88)
            min_price = max(min_price, after_hours_min)

        best = None
        for side, token_id in candidates:
            try:
                book = fetch_order_book(token_id)
                if not book or book.asks.best_price <= 0:
                    continue
                ask_price = book.asks.best_price
                if ask_price < min_price:
                    logger.debug("Snipe skip %s %s: ask %.0f¢ < min %.0f¢",
                                 key, side, ask_price * 100, min_price * 100)
                    continue
                if ask_price > self.snipe_config.max_buy_price:
                    continue
                # Pick the highest-priced side (most certain outcome)
                if best is None or ask_price > best[2]:
                    best = (side, token_id, ask_price)
            except Exception as e:
                logger.debug("Snipe book error %s %s: %s", key, side, e)

        if best is None:
            logger.debug("Snipe skip %s: no side >= %.0f¢", key, self.snipe_config.min_buy_price * 100)
            return None

        buy_side, token_id, ask_price = best
        min_shares = getattr(self.snipe_config, 'min_shares', 5.0)

        # Check depth
        book = fetch_order_book(token_id)
        depth = book.asks.depth_at_price(ask_price + 0.02, side="ask") if book else 0
        if depth < min_shares * 0.5:
            logger.info("Snipe skip %s %s: thin book %.1f shares", key, buy_side, depth)
            return None

        # Probability = ask price itself (market knows best at this point)
        probability = min(ask_price + 0.02, 0.99)
        pct_move = 0.0  # not used for direction anymore

        logger.info("Snipe candidate %s %s: ask=%.0f¢ depth=%.1f P=%.0f%%",
                    key, buy_side, ask_price * 100, depth, probability * 100)

        return buy_side, token_id, ask_price, probability, pct_move, min_shares

    async def check_opportunities(self):
        """
        Incremental limit-buy snipe strategy.
        All orders are limit bids (maker). No taker orders ever.

        Entry schedule (configurable via entry_times):
          - At 15s remaining: first limit bid at ask-1¢ ($3)
          - At 10s remaining: second limit bid if still 88¢+ ($3)
          - At  6s remaining: third limit bid if still 88¢+ ($3)

        Each entry re-checks conditions independently.
        If price drops below min_buy_price, that entry is skipped.
        Max total per window = len(entry_times) × entry_size_dollars.
        """
        if not self.snipe_config.enabled:
            return

        # Time-of-day filter (ET timezone)
        # ETH trades 24/7 but with higher min price after hours
        # BTC only trades during 6AM-8:30PM ET
        try:
            from datetime import datetime, timezone, timedelta
            ET = timezone(timedelta(hours=-4))  # EDT (UTC-4); adjust to -5 in winter
            et_now = datetime.now(ET)
            start_h = getattr(self.snipe_config, 'trading_start_hour_et', 6)
            end_h   = getattr(self.snipe_config, 'trading_end_hour_et', 20)
            end_m   = getattr(self.snipe_config, 'trading_end_minute_et', 30)
            et_minutes = et_now.hour * 60 + et_now.minute
            in_golden = et_minutes >= start_h * 60 and et_minutes < end_h * 60 + end_m
            self._in_golden_hours = in_golden  # used in _check_conditions
            # ETH overnight window: 2AM-5AM ET (100% win rate, after-hours min applies)
            eth_ov_start = getattr(self.snipe_config, 'eth_overnight_start_et', 2)
            eth_ov_end   = getattr(self.snipe_config, 'eth_overnight_end_et', 5)
            in_overnight = et_now.hour >= eth_ov_start and et_now.hour < eth_ov_end
            self._in_overnight_hours = in_overnight
            # BTC blocked outside golden hours entirely
            # ETH allowed in golden hours + overnight window only
        except Exception:
            self._in_golden_hours = True  # safe default

        now = time.time()
        self._resolve_expired()

        # Check pending trade outcomes
        if self.trades_logger:
            try:
                self.trades_logger.check_pending()
            except Exception as e:
                logger.debug("Trades logger check error: %s", e)

        # Cancel stale maker orders where token dropped below min price
        self._cancel_stale_makers()

        active_windows = self.window_tracker.get_active_windows()
        snipe_window = self.snipe_config.snipe_window_seconds
        cooldown = self.snipe_config.cooldown_seconds
        max_entries = self.snipe_config.max_entries_per_window
        taker_thresh = self.snipe_config.taker_seconds_remaining

        for window in active_windows:
            time_remaining = window.end_ts - now

            # Outside snipe window entirely
            if time_remaining < self.snipe_config.min_seconds_remaining:
                continue
            if time_remaining > snipe_window:
                continue

            existing = self._active_snipes.get(window.key)

            if existing:
                # Already entered — check cooldown and max entries
                entries_placed = existing.get("entries_placed", 0)
                if entries_placed >= max_entries:
                    continue  # Hit max entries for this window
                last_entry_time = existing.get("last_entry_time", 0)
                if now - last_entry_time < cooldown:
                    continue  # Too soon since last entry
                # Must stay on same side
            else:
                # First entry — check concurrent limit
                if len(self._active_snipes) >= self.snipe_config.max_concurrent:
                    continue

            # Check conditions
            result = self._check_conditions(window, time_remaining)
            if not result:
                continue

            buy_side, token_id, ask_price, probability, pct_move, _ = result

            # If already in position, must be same side
            if existing and buy_side != existing.get("side"):
                logger.info("⏭ Incremental skip %s: direction flipped %s→%s",
                            window.key, existing["side"], buy_side)
                continue

            # Dollar-based sizing: target $7.50/entry; floor at min_shares (5)
            _order_usd = getattr(self.snipe_config, 'order_size_dollars', 0.0)
            if _order_usd > 0:
                tokens = max(_order_usd / ask_price, self.snipe_config.min_shares)
            else:
                tokens = max(self.snipe_config.min_shares, 5.0)
            entry_num = (existing.get("entries_placed", 0) if existing else 0) + 1
            is_taker = time_remaining <= taker_thresh

            # Taker at ≤6s (guaranteed fill), maker earlier
            if is_taker:
                order_price = ask_price
            else:
                order_price = round(max(ask_price - self.snipe_config.maker_bid_offset, 0.01), 2)

            # Final price guard: re-fetch ask to catch market moves since _check_conditions
            # A BUY limit fills at ask price even if ask dropped below your limit
            try:
                fresh_book = fetch_order_book(token_id)
                if fresh_book and fresh_book.asks.best_price > 0:
                    fresh_ask = fresh_book.asks.best_price
                    price_key = f"min_buy_price_{window.asset.upper().lower()}"
                    # Use same min price logic as _check_conditions
                    _in_golden = getattr(self, '_in_golden_hours', True)
                    _min = getattr(self.snipe_config, price_key,
                                   getattr(self.snipe_config, 'min_buy_price', 0.79))
                    if not _in_golden and window.asset.upper() == "ETH":
                        _min = max(_min, getattr(self.snipe_config, 'min_buy_price_eth_afterhours', 0.88))
                    if fresh_ask < _min:
                        logger.info("⚠️  Price guard: %s %s ask dropped %.0f¢ < min %.0f¢, skipping",
                                    window.asset, buy_side, fresh_ask * 100, _min * 100)
                        continue
                    # Use fresh price for order
                    if is_taker:
                        order_price = fresh_ask
                    else:
                        order_price = round(max(fresh_ask - self.snipe_config.maker_bid_offset, 0.01), 2)
                    ask_price = fresh_ask
            except Exception as e:
                logger.debug("Price guard fetch failed: %s", e)

            cost = order_price * tokens
            order_type = "TAKER" if is_taker else "MAKER"

            logger.info(
                "🎯 SNIPE #%d [%s]: %s %s %s | %.1fs left | "
                "ask=%.0f¢ order=%.0f¢ | %.0f shares ($%.2f)",
                entry_num, order_type, window.asset, window.timeframe, buy_side,
                time_remaining, ask_price * 100, order_price * 100,
                tokens, cost,
            )

            self.stats["opportunities"] += 1
            await self._execute_snipe(
                window=window,
                buy_side=buy_side,
                token_id=token_id,
                bid_price=order_price,
                ask_price=ask_price,
                probability=probability,
                tokens=tokens,
                entry_num=entry_num,
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
                              bid_price: float, ask_price: float,
                              probability: float, tokens: float,
                              entry_num: int = 1):
        """Place an incremental limit bid for a snipe entry."""
        if self.config.paper_mode:
            order_id = f"snipe_paper_{window.key}_e{entry_num}"
            logger.info("📝 [PAPER] SNIPE #%d BUY %s %s @ %.0f¢ (%.2f tokens)",
                        entry_num, buy_side, window.asset, bid_price * 100, tokens)
        elif self.config.dry_run:
            logger.info("🔍 [DRY RUN] SNIPE #%d %s %s @ %.0f¢",
                        entry_num, buy_side, window.asset, bid_price * 100)
            return
        else:
            result = self.clob.place_order(
                token_id=token_id,
                side="BUY",
                price=bid_price,
                size=tokens,
            )
            if not result.get("success"):
                logger.error("❌ Snipe #%d failed: %s %s — %s",
                             entry_num, buy_side, window.asset, result.get("error"))
                # Count failed attempt to prevent hammering same window
                if window.key not in self._active_snipes:
                    self._active_snipes[window.key] = {
                        "side": buy_side, "asset": window.asset,
                        "token_id": token_id, "window_end": window.end_ts,
                        "entries_placed": 1, "last_entry_time": time.time(),
                        "orders": [], "total_tokens": 0, "total_cost": 0,
                        "avg_price": 0, "probability": probability, "placed_at": time.time(),
                    }
                else:
                    self._active_snipes[window.key]["entries_placed"] += 1
                    self._active_snipes[window.key]["last_entry_time"] = time.time()
                return
            order_id = result["orderID"]
            logger.info("✅ SNIPE #%d PLACED: %s %s @ %.0f¢ (%.2f tokens) → %s",
                        entry_num, buy_side, window.asset, bid_price * 100, tokens, order_id)

            # Record for W/L tracking
            if self.trades_logger:
                self.trades_logger.record_order(
                    order_id=order_id,
                    asset=window.asset,
                    direction=buy_side,
                    entry_price=bid_price,
                    shares=tokens,
                    window_end=window.end_ts,
                    token_id=token_id,
                    condition_id=getattr(window, 'condition_id', ''),
                )

        now = time.time()
        cost = bid_price * tokens

        if window.key not in self._active_snipes:
            self._active_snipes[window.key] = {
                "side": buy_side,
                "asset": window.asset,
                "token_id": token_id,
                "window_end": window.end_ts,
                "entries_placed": 1,
                "last_entry_time": now,
                "orders": [{"order_id": order_id, "price": bid_price, "tokens": tokens, "cost": cost}],
                "total_tokens": tokens,
                "total_cost": cost,
                "avg_price": bid_price,
                "probability": probability,
                "placed_at": now,
            }
        else:
            snipe = self._active_snipes[window.key]
            snipe["entries_placed"] += 1
            snipe["last_entry_time"] = now
            snipe["orders"].append({"order_id": order_id, "price": bid_price, "tokens": tokens, "cost": cost})
            snipe["total_tokens"] += tokens
            snipe["total_cost"] += cost
            snipe["avg_price"] = snipe["total_cost"] / snipe["total_tokens"]
            logger.info("📊 Window %s: %d entries, avg=%.0f¢, total=$%.2f",
                        window.key, snipe["entries_placed"], snipe["avg_price"] * 100, snipe["total_cost"])

        # Persist to disk immediately so restarts see correct entry count
        self._persist_snipes()

        self._sniped_windows[window.key] = now
        self.stats["trades"] += 1

    def _load_persisted_snipes(self):
        """Load active snipes from disk on startup — restores entry counts across restarts."""
        try:
            if not os.path.exists(self._snipes_persist_path):
                return
            with open(self._snipes_persist_path) as f:
                data = json.load(f)
            now = time.time()
            loaded = 0
            for key, snipe in data.items():
                # Skip windows that have already ended
                if snipe.get("window_end", 0) < now:
                    continue
                self._active_snipes[key] = snipe
                loaded += 1
            if loaded:
                logger.info("Restored %d active snipe window(s) from disk", loaded)
                for key, snipe in self._active_snipes.items():
                    logger.info("  → %s: %d entries placed, side=%s",
                                key, snipe.get("entries_placed", 0), snipe.get("side", "?"))
        except Exception as e:
            logger.warning("Could not load persisted snipes: %s", e)

    def _persist_snipes(self):
        """Save current active snipes to disk."""
        try:
            os.makedirs(os.path.dirname(self._snipes_persist_path), exist_ok=True)
            with open(self._snipes_persist_path, "w") as f:
                json.dump(self._active_snipes, f, indent=2)
        except Exception as e:
            logger.debug("Could not persist snipes: %s", e)

    def _cancel_stale_makers(self):
        """
        Cancel open maker orders if the token price has dropped below min_buy_price.
        Only fires while the round is still live (>5s remaining).
        Prevents filling at a bad price if the market reversed after order placement.
        """
        now = time.time()
        for window_key, snipe in list(self._active_snipes.items()):
            window_end = snipe.get("window_end", 0)
            # Only cancel while round is still live with >5s remaining
            if window_end < now or (window_end - now) < 5:
                continue
            orders = snipe.get("orders", [])
            token_id = snipe.get("token_id", "")
            asset = snipe.get("asset", "")
            if not token_id:
                continue
            try:
                book = fetch_order_book(token_id)
                if not book:
                    continue
                current_ask = book.asks.best_price
                # Skip if ask is 0 or near 0 (round already resolved)
                if current_ask < 0.05:
                    continue
                asset_upper = asset.upper()
                price_key = f"min_buy_price_{asset_upper.lower()}"
                min_price = getattr(self.snipe_config, price_key,
                            getattr(self.snipe_config, 'min_buy_price', 0.79))
                # If token dropped significantly below min, cancel open makers
                if current_ask < min_price - 0.05:
                    for order in orders:
                        oid = order.get("order_id", "")
                        if oid and not oid.startswith("snipe_paper"):
                            try:
                                self.clob.cancel_order(oid)
                                logger.info("🚫 Cancelled stale maker %s: %s ask=%.0f¢ < min=%.0f¢",
                                            oid[:8], asset, current_ask * 100, min_price * 100)
                            except Exception:
                                pass
            except Exception:
                pass

    def _resolve_expired(self):
        """
        Resolve snipes whose windows have ended.
        P&L calculated from Polymarket activity API for accuracy.
        Falls back to token price estimate if API unavailable.
        """
        now = time.time()
        resolved = []

        for key, snipe in self._active_snipes.items():
            if now < snipe["window_end"] + 8:  # 8s grace for settlement
                continue

            total_tokens = snipe["total_tokens"]
            total_cost = snipe["total_cost"]
            avg_price = snipe["avg_price"]

            # Estimate outcome: check if token price is near $1 or $0
            # Use Polymarket market price if available via window
            # For now: use current market data
            won = None
            try:
                import requests
                slug = key.replace("-5m-", f"-updown-5m-").replace("BTC", "btc").replace("ETH", "eth").replace("SOL", "sol")
                # Simple heuristic: if avg_price was >= 0.85, high chance of win
                # Real outcome determined by Polymarket settlement
                # We log "PENDING" and let the watcher script report actual P&L
                won = None  # Can't determine without settlement data
            except Exception:
                pass

            if won is True:
                pnl = total_tokens * 1.0 - total_cost
                self.stats["wins"] += 1
                logger.info("✅ SNIPE WIN %s %s: %d entries, avg=%.0f¢, P&L: +$%.3f",
                            snipe["asset"], snipe["side"],
                            snipe["entries_placed"], avg_price * 100, pnl)
            elif won is False:
                pnl = -total_cost
                self.stats["losses"] += 1
                logger.warning("❌ SNIPE LOSS %s %s: %d entries, avg=%.0f¢, P&L: -$%.3f",
                               snipe["asset"], snipe["side"],
                               snipe["entries_placed"], avg_price * 100, abs(pnl))
            else:
                # Can't determine — log as pending, actual P&L from Polymarket
                pnl = 0
                logger.info("⏳ SNIPE SETTLED %s %s: %d entries, avg=%.0f¢, $%.2f spent — check Polymarket for outcome",
                            snipe["asset"], snipe["side"],
                            snipe["entries_placed"], avg_price * 100, total_cost)

            self.stats["total_pnl"] += pnl
            resolved.append(key)

        for key in resolved:
            del self._active_snipes[key]
        if resolved:
            self._persist_snipes()

    def get_stats(self) -> dict:
        """Get snipe stats for health reporting."""
        return {
            **self.stats,
            "active_snipes": len(self._active_snipes),
            "win_rate": (
                self.stats["wins"] / max(1, self.stats["wins"] + self.stats["losses"]) * 100
            ),
        }
