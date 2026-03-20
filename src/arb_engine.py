#!/usr/bin/env python3
"""
Arb Engine — Latency Arbitrage on Polymarket Crypto Prediction Markets.

Detects when Binance crypto prices move sharply and buys the underpriced
side on Polymarket before the market adjusts.

Strategy:
  1. Monitor Binance price velocity (rolling 5s, 10s, 30s windows)
  2. When price moves >0.2% in 5 seconds, check Polymarket pricing
  3. Calculate fair value based on move magnitude + time remaining
  4. If Polymarket price < fair_value - edge_threshold, buy as taker
  5. Immediately post maker sell at entry + markup to capture quick profit
  6. If sell doesn't fill within timeout, hold to expiry as fallback
"""
import time
from dataclasses import dataclass, field
from typing import Optional

from src.logger import get_logger
from src.orderbook import fetch_order_book

logger = get_logger("arb_engine")


@dataclass
class ArbPosition:
    """Tracks a single arb position (one side of one window)."""
    position_id: int
    window_key: str
    asset: str
    timeframe: str
    side: str                     # "Up" or "Down"
    token_id: str
    entry_price: float            # What we paid per token
    size: float                   # Dollar size of the position
    fair_value: float             # Our estimated fair value at entry
    edge: float                   # fair_value - entry_price
    crypto_price_at_entry: float  # Binance price when we entered
    crypto_pct_move: float        # % move that triggered the arb
    window_start_ts: int
    window_end_ts: int
    window_start_price: float     # Crypto price at window open
    clob_order_id: str = ""
    is_filled: bool = False
    filled_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    pnl: float = 0.0
    outcome: str = ""             # "win", "loss", "sold", or ""
    # Quick-profit sell fields
    sell_price: float = 0.0       # Target sell price
    sell_order_id: str = ""       # CLOB order ID for the sell
    sell_posted_at: float = 0.0   # When we posted the sell
    sell_filled: bool = False     # Whether sell has been filled
    sell_timeout: float = 60.0    # Seconds to wait for sell before holding to expiry

    @property
    def time_remaining(self) -> float:
        return max(0, self.window_end_ts - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.window_end_ts

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "window_key": self.window_key,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "side": self.side,
            "entry_price": round(self.entry_price, 4),
            "size": round(self.size, 2),
            "fair_value": round(self.fair_value, 4),
            "edge": round(self.edge, 4),
            "crypto_pct_move": round(self.crypto_pct_move, 4),
            "time_remaining": round(self.time_remaining, 1),
            "is_filled": self.is_filled,
            "resolved": self.resolved,
            "pnl": round(self.pnl, 4),
            "outcome": self.outcome,
            "created_at": self.created_at,
            "sell_price": round(self.sell_price, 4),
            "sell_posted": self.sell_order_id != "",
            "sell_filled": self.sell_filled,
        }


@dataclass
class ArbConfig:
    """Arb-specific configuration (extracted from main config)."""
    enabled: bool = True
    order_size: float = 5.0
    max_exposure: float = 15.0
    min_edge: float = 0.05
    cooldown_seconds: float = 30.0
    move_threshold_pct: float = 0.2
    time_weight_early: float = 0.3
    time_weight_late: float = 0.7
    sell_discount_from_fair: float = 0.02  # Sell at fair_value - this (e.g. fair 0.65, sell 0.63)
    sell_timeout_seconds: float = 60.0     # Hold to expiry if sell not filled in this time


class ArbEngine:
    """
    Latency arbitrage engine for Polymarket crypto prediction markets.

    Monitors Binance price velocity and buys underpriced Polymarket outcomes
    before the market catches up. Posts a quick-profit sell immediately after
    buying; falls back to hold-to-expiry if sell doesn't fill.

    Args:
        config: Main bot Config object
        clob_manager: ClobManager for order placement (None in paper mode)
        price_feed: PriceFeed for Binance data
        window_tracker: WindowTracker for market discovery
    """

    def __init__(self, config, clob_manager, price_feed, window_tracker):
        self.config = config
        self.clob = clob_manager
        self.price_feed = price_feed
        self.window_tracker = window_tracker

        # Parse arb config from main config's raw yaml or use defaults
        self.arb_config = self._load_arb_config(config)

        # State
        self.positions: dict[str, ArbPosition] = {}  # window_key → position
        self._next_position_id = 1
        self._last_trade_time: float = 0.0
        self._traded_windows: set = set()  # window keys we've already arb'd

        # Stats (tracked separately from spread capture)
        self.stats = {
            "opportunities_detected": 0,
            "trades_placed": 0,
            "trades_filled": 0,
            "sells_posted": 0,
            "sells_filled": 0,
            "wins": 0,
            "losses": 0,
            "total_pnl": 0.0,
            "total_edge_captured": 0.0,
            "avg_edge": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
        }

        if self.arb_config.enabled:
            logger.info("=" * 50)
            logger.info("  ⚡ ARB ENGINE INITIALIZED")
            logger.info("  Order size: $%.2f", self.arb_config.order_size)
            logger.info("  Max exposure: $%.2f", self.arb_config.max_exposure)
            logger.info("  Min edge: $%.2f", self.arb_config.min_edge)
            logger.info("  Move threshold: %.2f%%", self.arb_config.move_threshold_pct)
            logger.info("  Cooldown: %.0fs", self.arb_config.cooldown_seconds)
            logger.info("  Sell price: fair_value - $%.2f", self.arb_config.sell_discount_from_fair)
            logger.info("  Sell timeout: %.0fs (then hold to expiry)",
                        self.arb_config.sell_timeout_seconds)
            logger.info("=" * 50)
        else:
            logger.info("⚡ Arb engine disabled in config")

    def _load_arb_config(self, config) -> ArbConfig:
        """Extract arb config from main config (supports raw yaml data)."""
        arb_cfg = ArbConfig()

        # Check if config has an arb sub-object (from _apply_section)
        if hasattr(config, 'arb'):
            arb_obj = config.arb
            for attr in [
                "enabled", "order_size", "max_exposure", "min_edge",
                "cooldown_seconds", "move_threshold_pct",
                "time_weight_early", "time_weight_late",
                "sell_discount_from_fair", "sell_timeout_seconds",
            ]:
                if hasattr(arb_obj, attr):
                    val = getattr(arb_obj, attr)
                    if val is not None:
                        expected_type = type(getattr(arb_cfg, attr))
                        try:
                            setattr(arb_cfg, attr, expected_type(val))
                        except (ValueError, TypeError):
                            pass

        return arb_cfg

    async def check_opportunities(self):
        """
        Main entry point — called every tick from the trading loop.

        1. Manage active sell orders (post sells, check fills, handle timeouts)
        2. Resolve expired positions
        3. Check price velocity on Binance
        4. Find windows where Polymarket hasn't caught up
        5. Execute if edge is sufficient
        """
        if not self.arb_config.enabled:
            return

        # Manage active positions (post sells, check timeouts)
        self._manage_active_positions()

        # Resolve expired positions
        self._resolve_expired()

        # Check cooldown
        now = time.time()
        if now - self._last_trade_time < self.arb_config.cooldown_seconds:
            return

        # Check exposure limit
        current_exposure = self._get_total_exposure()
        if current_exposure >= self.arb_config.max_exposure:
            return

        # Scan each asset for price velocity signals
        for asset in self.config.trading.allowed_assets:
            state = self.price_feed.get_state(asset)
            if not state or state.price <= 0:
                continue

            # Check price velocity over multiple windows
            signal = self._check_price_velocity(state)
            if not signal:
                continue

            # Found a significant move — look for arb opportunities
            await self._scan_windows_for_arb(asset, state, signal)

    def _check_price_velocity(self, state) -> Optional[dict]:
        """
        Check if price has moved significantly in recent seconds.

        Looks at 5s, 10s, and 30s rolling windows. Returns the strongest
        signal if any exceed the threshold.

        Returns:
            Dict with move details, or None if no significant move.
        """
        now = time.time()
        history = state.price_history
        if len(history) < 2:
            return None

        current_price = state.price
        best_signal = None

        for lookback_seconds in [5, 10, 30]:
            cutoff = now - lookback_seconds
            # Find the price closest to the cutoff time
            old_price = None
            for ts, price in history:
                if ts >= cutoff:
                    old_price = price
                    break

            if old_price is None or old_price <= 0:
                continue

            pct_move = ((current_price - old_price) / old_price) * 100

            if abs(pct_move) >= self.arb_config.move_threshold_pct:
                # Check if this is the strongest signal
                if best_signal is None or abs(pct_move) > abs(best_signal["pct_move"]):
                    best_signal = {
                        "pct_move": pct_move,
                        "direction": "up" if pct_move > 0 else "down",
                        "lookback_seconds": lookback_seconds,
                        "old_price": old_price,
                        "current_price": current_price,
                    }

        return best_signal

    async def _scan_windows_for_arb(self, asset: str, price_state, signal: dict):
        """
        For a given asset with a price velocity signal, check active windows
        for mispriced Polymarket outcomes.
        """
        active_windows = self.window_tracker.get_active_windows()

        for window in active_windows:
            # Only look at matching asset
            if window.asset != asset:
                continue

            # Only 5m windows (most liquid, fastest to resolve)
            if window.timeframe not in self.config.trading.allowed_timeframes:
                continue

            # Skip if we already have an arb position on this window
            if window.key in self._traded_windows:
                continue

            # Skip if not enough time remaining (need at least 30s)
            if window.time_remaining < 30:
                continue

            # Skip if no token IDs
            if not window.up_token_id or not window.down_token_id:
                continue

            # Need the window's start price to calculate fair value
            start_price = window.start_price
            if start_price <= 0:
                # Try to use the price feed's reference if available
                start_price = price_state.reference_price
            if start_price <= 0:
                continue

            # Calculate % move from window start
            pct_from_start = ((price_state.price - start_price) / start_price) * 100

            # Determine which side to buy
            if pct_from_start > 0:
                buy_side = "Up"
                token_id = window.up_token_id
            else:
                buy_side = "Down"
                token_id = window.down_token_id

            # Calculate fair value
            fair_value = self._calculate_fair_value(
                pct_move=abs(pct_from_start),
                time_remaining=window.time_remaining,
                total_window=window.end_ts - window.start_ts,
                volatility=price_state.realized_volatility(60),
            )

            # Fetch the order book to check actual market price
            book = fetch_order_book(token_id)
            if not book:
                continue

            # We're buying as taker — look at best ask (what we pay)
            market_price = book.asks.best_price
            if market_price <= 0:
                # Fall back to bid + spread estimate
                if book.bids.best_price > 0:
                    market_price = book.bids.best_price + 0.01
                else:
                    continue

            # Check if there's enough edge
            edge = fair_value - market_price
            if edge < self.arb_config.min_edge:
                logger.debug(
                    "Arb skip %s %s %s: fair=%.3f market=%.3f edge=%.3f (need %.3f)",
                    asset, window.timeframe, buy_side,
                    fair_value, market_price, edge, self.arb_config.min_edge,
                )
                continue

            # Check book depth — make sure there's enough liquidity
            available_at_ask = book.asks.depth_at_price(market_price + 0.02, side="ask")
            tokens_needed = self.arb_config.order_size / market_price
            if available_at_ask < tokens_needed * 0.5:
                logger.debug("Arb skip %s: insufficient ask depth (%.1f shares at %.2f)",
                             window.key, available_at_ask, market_price)
                continue

            # Re-check exposure limit
            current_exposure = self._get_total_exposure()
            if current_exposure + self.arb_config.order_size > self.arb_config.max_exposure:
                logger.debug("Arb skip: would exceed max exposure ($%.2f + $%.2f > $%.2f)",
                             current_exposure, self.arb_config.order_size,
                             self.arb_config.max_exposure)
                return  # Stop scanning entirely

            # 🎯 Execute the arb!
            self.stats["opportunities_detected"] += 1
            logger.info(
                "⚡ ARB SIGNAL: %s %s %s | move=%.3f%% from start | "
                "fair=%.3f | market=%.3f | edge=$%.3f | time_left=%.0fs",
                asset, window.timeframe, buy_side,
                pct_from_start, fair_value, market_price, edge,
                window.time_remaining,
            )

            await self._execute_arb(
                window=window,
                asset=asset,
                buy_side=buy_side,
                token_id=token_id,
                market_price=market_price,
                fair_value=fair_value,
                edge=edge,
                crypto_price=price_state.price,
                crypto_pct_move=pct_from_start,
                start_price=start_price,
            )

            # Only one arb per tick to avoid overtrading
            return

    def _calculate_fair_value(self, pct_move: float, time_remaining: float,
                               total_window: float, volatility: float) -> float:
        """
        Calculate fair value of the winning side given price move and time remaining.

        Model:
        - Base value = 0.50 (50/50 at window start)
        - Adjustment = (pct_move / volatility_estimate) * time_weight
        - Early in window: conservative (move might reverse)
        - Late in window: aggressive (move likely to stick)

        Args:
            pct_move: Absolute % move from window start price
            time_remaining: Seconds remaining in window
            total_window: Total window duration in seconds
            volatility: Realized volatility estimate (std of returns)

        Returns:
            Fair value probability [0.5, 0.99]
        """
        # Time weight: how confident are we the move will stick?
        fraction_remaining = time_remaining / total_window if total_window > 0 else 0.5

        if fraction_remaining > 0.6:  # >3 min left in 5-min window
            time_weight = self.arb_config.time_weight_early
        elif fraction_remaining < 0.4:  # <2 min left
            time_weight = self.arb_config.time_weight_late
        else:
            # Linear interpolation between early and late
            t = (0.6 - fraction_remaining) / 0.2
            time_weight = (self.arb_config.time_weight_early +
                           t * (self.arb_config.time_weight_late -
                                self.arb_config.time_weight_early))

        # Volatility estimate: use realized vol, with a floor
        vol_estimate = max(volatility, 0.0001)  # Floor to avoid division by zero

        # Normalize the move by volatility
        # Higher move relative to vol = more confident = higher fair value
        normalized_move = pct_move / (vol_estimate * 100)  # vol is in decimal form

        # Cap the normalized move to avoid extreme values
        normalized_move = min(normalized_move, 5.0)

        # Fair value: 0.50 + adjustment
        adjustment = normalized_move * time_weight * 0.15  # Scale factor
        fair_value = 0.50 + adjustment

        # Additional boost for very late windows with strong moves
        if fraction_remaining < 0.2 and pct_move > 0.3:
            # Less than 1 min left with >0.3% move — very likely to stick
            fair_value = max(fair_value, 0.65)

        if fraction_remaining < 0.1 and pct_move > 0.2:
            # Less than 30s left — almost certain
            fair_value = max(fair_value, 0.70)

        # Clamp to [0.50, 0.95] — never be 100% sure
        fair_value = max(0.50, min(0.95, fair_value))

        return fair_value

    async def _execute_arb(self, window, asset: str, buy_side: str,
                            token_id: str, market_price: float,
                            fair_value: float, edge: float,
                            crypto_price: float, crypto_pct_move: float,
                            start_price: float):
        """Place a taker buy order for the arb, then post maker sell for quick profit."""

        # Calculate size in tokens
        tokens = self.arb_config.order_size / market_price

        # Create position record
        position = ArbPosition(
            position_id=self._next_position_id,
            window_key=window.key,
            asset=asset,
            timeframe=window.timeframe,
            side=buy_side,
            token_id=token_id,
            entry_price=market_price,
            size=self.arb_config.order_size,
            fair_value=fair_value,
            edge=edge,
            crypto_price_at_entry=crypto_price,
            crypto_pct_move=crypto_pct_move,
            window_start_ts=window.start_ts,
            window_end_ts=window.end_ts,
            window_start_price=start_price,
            sell_timeout=self.arb_config.sell_timeout_seconds,
        )
        self._next_position_id += 1

        # Place order
        if self.config.paper_mode:
            position.clob_order_id = f"arb_paper_{position.position_id}"
            position.is_filled = True
            position.filled_at = time.time()
            logger.info(
                "📝 [PAPER] ARB BUY %s %s @ %.3f ($%.2f) | edge=$%.3f | "
                "fair=%.3f | crypto_move=%.3f%%",
                buy_side, asset, market_price, self.arb_config.order_size,
                edge, fair_value, crypto_pct_move,
            )
        elif self.config.dry_run:
            logger.info(
                "🔍 [DRY RUN] Would ARB BUY %s %s @ %.3f ($%.2f) | edge=$%.3f",
                buy_side, asset, market_price, self.arb_config.order_size,
                edge,
            )
            # Don't record position in dry run
            return
        else:
            # Live order — buy as taker (use ask price to ensure fill)
            result = self.clob.place_order(
                token_id=token_id,
                side="BUY",
                price=market_price,
                size=tokens,
            )

            if result.get("success"):
                position.clob_order_id = result["orderID"]
                position.is_filled = True  # Taker order = immediate fill
                position.filled_at = time.time()
                logger.info(
                    "⚡ ARB BUY FILLED: %s %s @ %.3f ($%.2f) → %s",
                    buy_side, asset, market_price,
                    self.arb_config.order_size, result["orderID"],
                )
            else:
                logger.error(
                    "❌ ARB BUY FAILED: %s %s @ %.3f — %s",
                    buy_side, asset, market_price,
                    result.get("error", "unknown"),
                )
                return

        # Record position
        self.positions[window.key] = position
        self._traded_windows.add(window.key)
        self._last_trade_time = time.time()
        self.stats["trades_placed"] += 1
        if position.is_filled:
            self.stats["trades_filled"] += 1

    # ── Quick-profit sell management ──────────────────────────────────

    def _manage_active_positions(self):
        """Manage active arb positions: post sell orders and handle timeouts."""
        now = time.time()

        for key, pos in list(self.positions.items()):
            if pos.resolved or not pos.is_filled:
                continue

            # Already sold — nothing to do
            if pos.sell_filled:
                continue

            # Post sell order if we haven't yet
            if not pos.sell_order_id:
                self._post_sell_order(pos)
                continue

            # Check if sell has been filled
            if pos.sell_order_id and not pos.sell_filled:
                self._check_sell_fill(pos)

            # If sell hasn't filled and timeout exceeded, cancel and hold to expiry
            if (pos.sell_order_id and not pos.sell_filled and
                pos.sell_posted_at > 0 and
                now - pos.sell_posted_at > pos.sell_timeout):

                # Don't cancel if window is about to expire anyway (< 10s)
                if pos.time_remaining < 10:
                    continue

                logger.info(
                    "⏰ ARB #%d: sell timeout (%.0fs), cancelling sell → holding to expiry",
                    pos.position_id, now - pos.sell_posted_at,
                )
                try:
                    self.clob.cancel_order(pos.sell_order_id)
                except Exception as e:
                    logger.warning("Failed to cancel arb sell order: %s", e)
                pos.sell_order_id = ""  # Clear so we don't try to cancel again

    def _post_sell_order(self, pos: ArbPosition):
        """Post a maker sell order just below fair value to capture the mispricing edge."""
        # Sell at fair_value - small discount (to get filled quickly)
        sell_price = round(pos.fair_value - self.arb_config.sell_discount_from_fair, 2)
        # Floor: never sell below entry price (guaranteed loss)
        sell_price = max(sell_price, round(pos.entry_price + 0.01, 2))
        sell_price = min(sell_price, 0.99)  # Cap at 0.99

        tokens = pos.size / pos.entry_price  # How many tokens we bought
        expected_profit = (sell_price - pos.entry_price) * tokens

        if self.config.paper_mode:
            pos.sell_price = sell_price
            pos.sell_order_id = f"arb_sell_paper_{pos.position_id}"
            pos.sell_posted_at = time.time()
            self.stats["sells_posted"] += 1
            logger.info(
                "📝 [PAPER] ARB SELL posted: #%d %s %s @ %.3f "
                "(bought @ %.3f, fair=%.3f, expected profit $%.2f)",
                pos.position_id, pos.asset, pos.side,
                sell_price, pos.entry_price, pos.fair_value, expected_profit,
            )
            return

        result = self.clob.place_order(
            token_id=pos.token_id,
            side="SELL",
            price=sell_price,
            size=tokens,
        )

        if result.get("success"):
            pos.sell_price = sell_price
            pos.sell_order_id = result["orderID"]
            pos.sell_posted_at = time.time()
            self.stats["sells_posted"] += 1
            logger.info(
                "⚡ ARB SELL posted: #%d %s %s @ %.3f (bought @ %.3f, fair=%.3f, "
                "expected ~$%.2f) → %s",
                pos.position_id, pos.asset, pos.side,
                sell_price, pos.entry_price, pos.fair_value,
                expected_profit, result["orderID"],
            )
        else:
            logger.error(
                "❌ ARB SELL failed: #%d %s — %s (will retry next tick)",
                pos.position_id, pos.asset, result.get("error", "unknown"),
            )

    def _check_sell_fill(self, pos: ArbPosition):
        """Check if our sell order has been filled."""
        if not pos.sell_order_id or pos.sell_filled:
            return

        # In paper mode, simulate fill if market bid >= sell_price
        if self.config.paper_mode:
            book = fetch_order_book(pos.token_id)
            if book and book.bids.best_price >= pos.sell_price:
                self._complete_sell(pos)
            return

        try:
            order = self.clob.get_order_status(pos.sell_order_id)
            if order:
                matched = float(order.get("size_matched", 0))
                total = float(order.get("original_size", 1))
                if matched >= total * 0.95:  # 95%+ filled = consider it done
                    self._complete_sell(pos)
        except Exception as e:
            logger.debug("Error checking arb sell order: %s", e)

    def _complete_sell(self, pos: ArbPosition):
        """Mark a sell as complete and calculate P&L."""
        tokens = pos.size / pos.entry_price
        revenue = tokens * pos.sell_price
        # Taker fee on buy (~1% estimate), maker fee on sell (0% + small rebate)
        buy_fee = pos.size * 0.01
        pnl = revenue - pos.size - buy_fee

        pos.sell_filled = True
        pos.resolved = True
        pos.pnl = pnl
        pos.outcome = "sold"

        self.stats["sells_filled"] += 1
        self.stats["wins"] += 1
        self.stats["total_pnl"] += pnl
        self.stats["total_edge_captured"] += pos.edge
        self.stats["largest_win"] = max(self.stats["largest_win"], pnl)

        logger.info(
            "💰 ARB SOLD #%d: %s %s | bought @ %.3f → sold @ %.3f | "
            "P&L: $%.4f (%.0fs held)",
            pos.position_id, pos.asset, pos.side,
            pos.entry_price, pos.sell_price, pnl,
            time.time() - pos.filled_at,
        )

    # ── Resolution (fallback for unsold positions) ────────────────────

    def _resolve_expired(self):
        """Resolve arb positions whose windows have expired."""
        for key, pos in list(self.positions.items()):
            if pos.resolved or not pos.is_expired:
                continue

            if not pos.is_filled:
                # Never filled — just clean up
                pos.resolved = True
                pos.pnl = 0.0
                pos.outcome = "unfilled"
                continue

            # Already sold for quick profit — skip
            if pos.sell_filled:
                continue

            # Cancel any pending sell order before resolution
            if pos.sell_order_id and not pos.sell_filled:
                try:
                    self.clob.cancel_order(pos.sell_order_id)
                    logger.debug("Cancelled pending arb sell for expired #%d", pos.position_id)
                except Exception:
                    pass

            # Determine outcome based on final crypto price
            current_price = self.price_feed.get_price(pos.asset)
            if current_price <= 0:
                logger.warning("Cannot resolve arb #%d: no price data for %s",
                               pos.position_id, pos.asset)
                continue

            won = False
            if pos.side == "Up" and current_price >= pos.window_start_price:
                won = True
            elif pos.side == "Down" and current_price < pos.window_start_price:
                won = True

            if won:
                # Winner pays $1.00 per token
                tokens = pos.size / pos.entry_price
                payout = tokens * 1.0
                buy_fee = pos.size * 0.01  # Taker fee on original buy
                pnl = payout - pos.size - buy_fee
                pos.pnl = pnl
                pos.outcome = "win"
                self.stats["wins"] += 1
                self.stats["largest_win"] = max(self.stats["largest_win"], pnl)
                logger.info(
                    "✅ ARB WIN #%d (held to expiry): %s %s %s | entry=%.3f | "
                    "P&L=$%.4f",
                    pos.position_id, pos.asset, pos.side,
                    pos.timeframe, pos.entry_price, pnl,
                )
            else:
                # Loser pays $0.00
                pnl = -pos.size  # Lost entire position
                pos.pnl = pnl
                pos.outcome = "loss"
                self.stats["losses"] += 1
                self.stats["largest_loss"] = min(self.stats["largest_loss"], pnl)
                logger.info(
                    "❌ ARB LOSS #%d (held to expiry): %s %s %s | entry=%.3f | "
                    "P&L=$%.4f",
                    pos.position_id, pos.asset, pos.side,
                    pos.timeframe, pos.entry_price, pnl,
                )

            pos.resolved = True
            self.stats["total_pnl"] += pnl
            self.stats["total_edge_captured"] += pos.edge if won else 0

            # Update average edge
            filled = self.stats["trades_filled"]
            if filled > 0:
                self.stats["avg_edge"] = self.stats["total_edge_captured"] / filled

    # ── Helpers ───────────────────────────────────────────────────────

    def _get_total_exposure(self) -> float:
        """Get total dollar exposure across all active (unfilled or unresolved) arb positions."""
        return sum(
            pos.size for pos in self.positions.values()
            if pos.is_filled and not pos.resolved
        )

    def get_active_arb_positions(self) -> list[ArbPosition]:
        """Get all active (unresolved) arb positions for health/dashboard reporting."""
        return [
            pos for pos in self.positions.values()
            if not pos.resolved
        ]

    def get_stats(self) -> dict:
        """Get arb stats for health reporting."""
        return {
            **self.stats,
            "active_positions": len(self.get_active_arb_positions()),
            "total_exposure": round(self._get_total_exposure(), 2),
            "cooldown_remaining": max(0, round(
                self.arb_config.cooldown_seconds - (time.time() - self._last_trade_time), 1
            )),
        }
