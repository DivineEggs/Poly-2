#!/usr/bin/env python3
"""
Trader V2 — Two-Sided Market Making with Binance Latency Shield.

Main trading loop:
  1. Scan for active 5-min crypto up/down markets
  2. Fetch order books for both sides (Up/Down)
  3. Evaluate if spread is profitable (Up bid + Down bid < $1.00)
  4. Place BUY orders on both sides as maker (0% fee + rebate)
  5. Monitor Binance for price moves → cancel losing side (shield)
  6. Track fills and resolve at window expiry
"""
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config, load_config, set_config, DATA_DIR
from src.logger import setup_logging, get_logger
from src.kill_switch import KillSwitch
from src.state_manager import StateManager
from src.health import HealthMonitor, MetricsExporter
from src.price_feed import PriceFeed
from src.polymarket_feed import PolymarketFeed
from src.window_tracker import WindowTracker
from src.orderbook import fetch_order_book
from src.market_maker import MarketMakerEngine, PairStatus
from src.redeemer import AutoRedeemer
from src.arb_engine import ArbEngine
from src.late_snipe import LateSnipeEngine

logger = get_logger("trader")


class TraderV2:
    """
    Two-sided market maker for Polymarket crypto prediction markets.

    Supports:
      - paper_mode: Simulate fills (no real orders)
      - dry_run: Real API reads but log-only for orders
      - Live: Real CLOB orders with real money
    """

    def __init__(self, config: Config = None):
        self.config = config or load_config()
        set_config(self.config)
        self._root_logger = setup_logging(self.config)

        os.makedirs(DATA_DIR, exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)

        # Core components
        self.price_feed = PriceFeed()
        self.polymarket_feed = PolymarketFeed()
        self.window_tracker = WindowTracker(self.config)
        self.kill_switch = KillSwitch(self.config)
        self.state_manager = StateManager()
        self.health_monitor = HealthMonitor(self.config)
        self.metrics = MetricsExporter(self.config)

        # CLOB manager (only for live/dry-run)
        self.clob_manager = None
        if not self.config.paper_mode:
            try:
                from src.clob_client import ClobManager
                self.clob_manager = ClobManager(self.config)
            except Exception as e:
                logger.error("Failed to initialize CLOB client: %s", e)
                if not self.config.dry_run:
                    raise

        # Market maker engine
        self.mm_engine = MarketMakerEngine(
            clob_manager=self.clob_manager,
            config=self.config,
        )

        # Auto-redeemer (claims resolved positions → USDC)
        self.redeemer = AutoRedeemer(self.config)

        # Arb engine (latency arbitrage on price moves)
        self.arb_engine = ArbEngine(
            config=self.config,
            clob_manager=self.clob_manager,
            price_feed=self.price_feed,
            window_tracker=self.window_tracker,
        )

        # Late snipe engine (buy at 95-99¢ in final 30s when outcome is near-certain)
        self.snipe_engine = LateSnipeEngine(
            config=self.config,
            clob_manager=self.clob_manager,
            price_feed=self.price_feed,
            window_tracker=self.window_tracker,
        )

        # Shutdown handling
        self._shutting_down = False
        self._start_time = 0

        # Track which windows we've already created pairs for
        self._traded_windows = set()

        # Track repricing state (don't reprice the same pair repeatedly)
        self._repriced_pairs = {}  # pair_id → last_reprice_time

        # Track re-chase attempts per pair
        self._rechase_attempts = {}  # pair_id → count

        # Smart exit state: pairs being actively monitored for exit
        self._smart_exit_pairs = {}  # pair_id → {"side": str, "token_id": str, "entry_price": float,
                                     #             "baseline_crypto": float, "entered_at": float,
                                     #             "sell_order_id": str, "sell_price": float}

        # Balance state
        self._usdc_balance = -1.0  # -1 = not checked yet
        self._last_balance_check = 0

        logger.info("=" * 70)
        logger.info("  🎯 POLYMARKET MARKET MAKER V2")
        logger.info("  Mode: %s", "PAPER" if self.config.paper_mode else
                     ("DRY RUN" if self.config.dry_run else "🔴 LIVE"))
        logger.info("  Order size: $%.2f per side", self.config.trading.order_size)
        logger.info("  Max position: $%.2f", self.config.trading.max_position)
        logger.info("  Min spread profit: $%.2f", self.config.trading.min_spread_profit)
        logger.info("  Shield threshold: %.2f%%", self.config.shield.threshold * 100)
        logger.info("  Assets: %s", ", ".join(self.config.trading.allowed_assets))
        logger.info("  Arb engine: %s", "ENABLED" if hasattr(self.config, 'arb') and self.config.arb.enabled else "DISABLED")
        logger.info("  Spread capture: %s", "ENABLED" if self.config.trading.spread_capture_enabled else "DISABLED")
        logger.info("=" * 70)

    async def run(self, duration_seconds: int = 600):
        """Main trading loop."""
        self._start_time = time.time()
        self._setup_signal_handlers()

        # Start data feeds
        price_task = asyncio.create_task(self.price_feed.connect())
        # Enable Polymarket WebSocket for real-time order book + fill detection
        poly_task = asyncio.create_task(self.polymarket_feed.connect())

        # Wait for price data
        logger.info("Waiting for Binance price data...")
        for _ in range(100):
            await asyncio.sleep(0.1)
            if all(self.price_feed.get_price(s) > 0
                   for s in self.config.trading.allowed_assets):
                break
        else:
            logger.error("Timeout waiting for Binance prices")

        for sym in self.config.trading.allowed_assets:
            p = self.price_feed.get_price(sym)
            logger.info("  %s: $%.2f", sym, p)

        # Initial market scan
        logger.info("Scanning for active markets...")
        await self.window_tracker.scan()
        active = self.window_tracker.get_active_windows()
        logger.info("Found %d active windows", len(active))

        # Restore state from previous run
        self._restore_state()

        # Main loop
        last_scan = 0
        last_order_check = 0
        last_health = 0

        try:
            while not self._shutting_down:
                now = time.time()
                elapsed = now - self._start_time

                # Duration limit
                if duration_seconds and elapsed > duration_seconds:
                    logger.info("Duration limit reached (%.0fs)", elapsed)
                    break

                # Kill switch
                if self.kill_switch.check(self.mm_engine.stats["total_pnl"]):
                    logger.critical("Kill switch triggered: %s", self.kill_switch.reason)
                    self._cancel_all_open()
                    self.health_monitor.set_status("halted")
                    break

                # Periodic market scan + subscribe new tokens to Polymarket WS
                if now - last_scan > self.config.timing.scan_interval:
                    await self.window_tracker.scan()
                    # Subscribe new token IDs to Polymarket WS for real-time book data
                    for w in self.window_tracker.get_active_windows():
                        if w.up_token_id and w.down_token_id:
                            self.polymarket_feed.subscribe([w.up_token_id, w.down_token_id])
                    last_scan = now

                # Periodic balance check (every 60s)
                if now - self._last_balance_check > 60 and self.clob_manager:
                    bal = self.clob_manager.get_usdc_balance()
                    if bal >= 0:
                        self._usdc_balance = bal
                        if bal < self.config.trading.min_usdc_balance:
                            logger.warning("Low USDC balance: $%.2f (min: $%.2f)",
                                           bal, self.config.trading.min_usdc_balance)
                    self._last_balance_check = now

                # One-trade-at-a-time check
                active_count = len(self.mm_engine.get_active_pairs())
                active_arb = len(self.arb_engine.get_active_arb_positions())
                max_concurrent = getattr(self.config.trading, 'max_concurrent_pairs', 0)
                can_trade = max_concurrent <= 0 or (active_count + active_arb) < max_concurrent

                # Arb engine: check for latency arb opportunities (every tick)
                if can_trade:
                    await self.arb_engine.check_opportunities()

                # Evaluate and place new pairs (spread capture)
                if self.config.trading.spread_capture_enabled and can_trade:
                    await self._evaluate_markets()

                # Late snipe: buy near-certain outcomes in final 30s
                await self.snipe_engine.check_opportunities()

                # Shield check (every iteration — speed matters)
                self._run_shield_checks()

                # Check order fills
                if now - last_order_check > self.config.timing.order_check_interval:
                    await self._check_fills()
                    await self._check_early_exits()
                    last_order_check = now

                # Smart exit monitoring (runs every tick — speed matters)
                if self._smart_exit_pairs:
                    await self._run_smart_exits()

                # Resolve expired windows (only for both-filled pairs)
                self._resolve_expired()

                # Auto-redeem resolved positions (every 60s)
                self.redeemer.check_and_redeem()

                # Health update
                if now - last_health > self.config.timing.health_write_interval:
                    self._write_health()
                    last_health = now

                await asyncio.sleep(0.1)  # 100ms tick — fast enough for shield

        except Exception as e:
            logger.error("Main loop error: %s", e, exc_info=True)
        finally:
            await self._graceful_shutdown()
            price_task.cancel()
            if poly_task:
                self.polymarket_feed.stop()
                poly_task.cancel()
            try:
                await price_task
            except asyncio.CancelledError:
                pass
            if poly_task:
                try:
                    await poly_task
                except asyncio.CancelledError:
                    pass

    async def _evaluate_markets(self):
        """Check all active windows for profitable spread opportunities."""
        # Balance gate
        if 0 <= self._usdc_balance < self.config.trading.min_usdc_balance:
            return

        active_windows = self.window_tracker.get_active_windows()

        for window in active_windows:
            # Skip if we already have a pair for this window
            if window.key in self._traded_windows:
                continue

            # Skip if not enough time remaining
            if window.time_remaining < self.config.trading.cancel_time_remaining + 30:
                continue

            # Skip if not in allowed assets/timeframes
            if window.asset not in self.config.trading.allowed_assets:
                continue
            if window.timeframe not in self.config.trading.allowed_timeframes:
                continue

            # Check position limits
            current_exposure = self.mm_engine.get_total_exposure()
            if current_exposure >= self.config.trading.max_position:
                continue

            # One-trade-at-a-time check
            max_concurrent = getattr(self.config.trading, 'max_concurrent_pairs', 0)
            if max_concurrent > 0:
                active_count = len(self.mm_engine.get_active_pairs())
                active_arb = len(self.arb_engine.get_active_arb_positions())
                if (active_count + active_arb) >= max_concurrent:
                    return  # Stop evaluating entirely

            # Fetch order books for both sides (track latency)
            # Use Polymarket WS data if available, else fall back to REST
            t0 = time.time()
            up_ws_age = self.polymarket_feed.get_feed_age(window.up_token_id)
            down_ws_age = self.polymarket_feed.get_feed_age(window.down_token_id)

            if up_ws_age is not None and up_ws_age < 5:
                up_best_bid = self.polymarket_feed.get_best_bid(window.up_token_id)
                up_best_ask = self.polymarket_feed.get_best_ask(window.up_token_id)
                up_book = fetch_order_book(window.up_token_id)  # Still need depth
            else:
                up_book = fetch_order_book(window.up_token_id)
                up_best_bid = up_book.bids.best_price if up_book else 0
                up_best_ask = up_book.asks.best_price if up_book else 0

            if down_ws_age is not None and down_ws_age < 5:
                down_best_bid = self.polymarket_feed.get_best_bid(window.down_token_id)
                down_best_ask = self.polymarket_feed.get_best_ask(window.down_token_id)
                down_book = fetch_order_book(window.down_token_id)
            else:
                down_book = fetch_order_book(window.down_token_id)
                down_best_bid = down_book.bids.best_price if down_book else 0
                down_best_ask = down_book.asks.best_price if down_book else 0

            self._last_poly_latency_ms = (time.time() - t0) * 500

            if not up_book or not down_book:
                continue
            if up_best_bid <= 0 or down_best_bid <= 0:
                continue

            # ── IMPROVEMENT 3: Book depth analysis ──
            # Check that there's enough liquidity for our order to fill
            min_depth_shares = 5  # Need at least 5 shares of depth near our price
            up_depth = up_book.asks.depth_at_price(up_best_bid + 0.03, side="ask")
            down_depth = down_book.asks.depth_at_price(down_best_bid + 0.03, side="ask")
            if up_depth < min_depth_shares or down_depth < min_depth_shares:
                logger.debug("Skipping %s: thin book (Up depth=%.0f, Down depth=%.0f)",
                             window.key, up_depth, down_depth)
                continue

            # Evaluate spread
            evaluation = self.mm_engine.evaluate_spread(
                up_best_bid=up_best_bid,
                down_best_bid=down_best_bid,
                min_profit=self.config.trading.min_spread_profit,
            )

            if not evaluation["tradeable"]:
                logger.debug("Spread not profitable for %s: %s",
                             window.key, evaluation.get("reason", ""))
                continue

            # Get start price from Binance
            start_price = self.price_feed.get_price(window.asset)
            if start_price <= 0:
                continue

            # ── IMPROVEMENT 4: Dynamic sizing ──
            # Wider spreads get more shares, narrow spreads get minimum
            base_size = self.config.trading.order_size
            spread_cents = evaluation["profit"] * 100  # profit per token in cents
            if spread_cents >= 6:
                dynamic_size = base_size * 2.0    # Double size on wide spreads
            elif spread_cents >= 4:
                dynamic_size = base_size * 1.5    # 1.5x on decent spreads
            else:
                dynamic_size = base_size          # Base size on minimum spreads
            # Cap at max single exposure
            dynamic_size = min(dynamic_size, self.config.trading.max_single_exposure)

            # ── IMPROVEMENT 2: Smarter order sequencing ──
            # Post the HARDER fill first (less depth = harder)
            # If the hard side doesn't fill, cancel the easy side (no loss)
            # If the hard side fills, the easy side is more likely to fill
            up_is_harder = up_depth < down_depth

            # Create pair
            pair = self.mm_engine.create_pair(
                window_key=window.key,
                asset=window.asset,
                timeframe=window.timeframe,
                window_start_ts=window.start_ts,
                window_end_ts=window.end_ts,
                start_price=start_price,
                up_token_id=window.up_token_id,
                down_token_id=window.down_token_id,
                up_price=evaluation["up_price"],
                down_price=evaluation["down_price"],
                size=dynamic_size,
            )

            # Always track window to prevent duplicate pair creation
            self._traded_windows.add(window.key)

            # Subscribe to Polymarket WS for these tokens (faster fill detection)
            self.polymarket_feed.subscribe([window.up_token_id, window.down_token_id])

            # Place orders with smart sequencing
            success = await self._place_pair_smart(pair, up_is_harder)
            if success:
                logger.info("🚀 Pair placed for %s %s: Up@%.2f + Down@%.2f = $%.4f profit/token "
                            "(size=$%.1f, first=%s, up_depth=%.0f, down_depth=%.0f)",
                             window.asset, window.timeframe,
                             evaluation["up_price"], evaluation["down_price"],
                             evaluation["profit"], dynamic_size,
                             "Up" if up_is_harder else "Down",
                             up_depth, down_depth)

    async def _place_pair_smart(self, pair, up_is_harder: bool) -> bool:
        """
        Place pair orders with smart sequencing: harder fill first.
        
        If the hard side doesn't fill, we cancel and lose nothing.
        If it fills, the easy side is more likely to fill too.
        """
        if self.config.paper_mode:
            return await self.mm_engine.place_pair_orders(pair)

        if self.config.dry_run:
            return await self.mm_engine.place_pair_orders(pair)

        if not self.clob_manager:
            return False

        # Determine order: harder side first
        if up_is_harder:
            first_leg, second_leg = pair.up_leg, pair.down_leg
            first_label, second_label = "Up (harder)", "Down (easier)"
        else:
            first_leg, second_leg = pair.down_leg, pair.up_leg
            first_label, second_label = "Down (harder)", "Up (easier)"

        # Place first (harder) order
        logger.info("  Placing %s first: %s @ %.2f", first_label, first_leg.token_id[:12], first_leg.price)
        first_result = self.clob_manager.place_order(
            token_id=first_leg.token_id, side="BUY",
            price=first_leg.price, size=first_leg.size,
        )
        if not first_result.get("success"):
            logger.error("❌ First order (%s) failed: %s", first_label, first_result.get("error"))
            return False
        first_leg.clob_order_id = first_result["orderID"]

        # Place second (easier) order
        logger.info("  Placing %s second: %s @ %.2f", second_label, second_leg.token_id[:12], second_leg.price)
        second_result = self.clob_manager.place_order(
            token_id=second_leg.token_id, side="BUY",
            price=second_leg.price, size=second_leg.size,
        )
        if not second_result.get("success"):
            logger.error("❌ Second order (%s) failed, cancelling first", second_label)
            self.clob_manager.cancel_order(first_leg.clob_order_id)
            return False
        second_leg.clob_order_id = second_result["orderID"]

        pair.status = PairStatus.OPEN
        logger.info("✅ Pair #%d orders live (smart seq): %s=%s, %s=%s",
                     pair.pair_id, first_label, first_leg.clob_order_id[:12],
                     second_label, second_leg.clob_order_id[:12])
        return True

    def _run_shield_checks(self):
        """Check Binance prices and cancel losing sides."""
        if not self.config.shield.enabled:
            return

        for pair in self.mm_engine.get_active_pairs():
            if pair.status not in (PairStatus.OPEN,):
                continue

            current_price = self.price_feed.get_price(pair.asset)
            if current_price <= 0:
                continue

            cancel_side = self.mm_engine.shield_check(pair, current_price)
            if cancel_side:
                # Run cancel synchronously (speed critical)
                asyncio.ensure_future(self.mm_engine.execute_shield(pair, cancel_side))

    async def _check_fills(self):
        """Check order fill status. Uses Polymarket WS for speed, REST as fallback."""
        for pair in self.mm_engine.get_active_pairs():
            if pair.status == PairStatus.BOTH_FILLED:
                continue

            up_filled = pair.up_leg.is_filled
            down_filled = pair.down_leg.is_filled

            if self.config.paper_mode:
                now = time.time()
                if not up_filled and not pair.up_leg.is_cancelled and now - pair.created_at > 5:
                    up_filled = True
                if not down_filled and not pair.down_leg.is_cancelled and now - pair.created_at > 5:
                    down_filled = True
            else:
                # Check Up leg
                if not up_filled and not pair.up_leg.is_cancelled and pair.up_leg.clob_order_id:
                    # Fast path: check if Polymarket WS shows our bid was lifted
                    # (best bid dropped below our price = someone took our order)
                    ws_bid = self.polymarket_feed.get_best_bid(pair.up_leg.token_id)
                    ws_age = self.polymarket_feed.get_feed_age(pair.up_leg.token_id)
                    if ws_age is not None and ws_age < 5 and ws_bid > 0:
                        # If best bid is now below our price, our order may have filled
                        if ws_bid < pair.up_leg.price - 0.005:
                            # Confirm via REST
                            status = self.clob_manager.get_order_status(pair.up_leg.clob_order_id)
                            if float(status.get("size_matched", 0) or 0) > 0:
                                up_filled = True
                    else:
                        # Fallback: REST poll
                        status = self.clob_manager.get_order_status(pair.up_leg.clob_order_id)
                        if status.get("status") == "FILLED" or float(status.get("size_matched", 0) or 0) > 0:
                            up_filled = True

                # Check Down leg (same logic)
                if not down_filled and not pair.down_leg.is_cancelled and pair.down_leg.clob_order_id:
                    ws_bid = self.polymarket_feed.get_best_bid(pair.down_leg.token_id)
                    ws_age = self.polymarket_feed.get_feed_age(pair.down_leg.token_id)
                    if ws_age is not None and ws_age < 5 and ws_bid > 0:
                        if ws_bid < pair.down_leg.price - 0.005:
                            status = self.clob_manager.get_order_status(pair.down_leg.clob_order_id)
                            if float(status.get("size_matched", 0) or 0) > 0:
                                down_filled = True
                    else:
                        status = self.clob_manager.get_order_status(pair.down_leg.clob_order_id)
                        if status.get("status") == "FILLED" or float(status.get("size_matched", 0) or 0) > 0:
                            down_filled = True

            self.mm_engine.update_fill_status(pair, up_filled, down_filled)

    async def _check_early_exits(self):
        """Single-sided fills: re-chase the unfilled side, then stop-loss if that fails.
        
        Strategy:
        1. Re-chase: reprice unfilled side more aggressively (total still < $1.00)
        2. Up to max_rechase_attempts, each time bidding higher
        3. If re-chase exhausted → stop-loss: sell at market to limit losses
        4. NEVER hold single-side fills to resolution (that's a coin flip)
        """
        for pair in self.mm_engine.get_active_pairs():
            up_filled = pair.up_leg.is_filled
            down_filled = pair.down_leg.is_filled

            # Both filled = guaranteed profit, nothing to do
            if up_filled and down_filled:
                continue
            # Neither filled = nothing to exit
            if not up_filled and not down_filled:
                continue

            # Determine which side is unfilled
            if up_filled:
                unfilled_side = "Down"
                unfilled_leg = pair.down_leg
                filled_leg = pair.up_leg
            else:
                unfilled_side = "Up"
                unfilled_leg = pair.up_leg
                filled_leg = pair.down_leg

            now = time.time()
            time_until_expiry = pair.window_end_ts - now
            rechase_count = self._rechase_attempts.get(pair.pair_id, 0)
            max_rechase = getattr(self.config.trading, 'rechase_max_attempts', 3)
            rechase_wait = getattr(self.config.trading, 'rechase_wait_seconds', 5.0)

            # How long since the fill?
            fill_time = filled_leg.filled_at or pair.created_at
            time_since_fill = now - fill_time

            # If unfilled side was cancelled by shield, skip re-chase → go to SL
            if unfilled_leg.is_cancelled:
                # Stop-loss: sell the filled side at market
                if getattr(self.config.trading, 'stop_loss_enabled', True):
                    self._enter_smart_exit(pair, filled_leg)
                continue

            # RE-CHASE PHASE: try to fill the other side at a higher price
            if rechase_count < max_rechase and time_since_fill > rechase_wait:
                if time_until_expiry < 20:
                    # Not enough time for another re-chase → stop-loss
                    logger.info("⏰ Pair #%d: not enough time for re-chase (%ds left), stopping out",
                                pair.pair_id, int(time_until_expiry))
                    # Cancel unfilled order first
                    if unfilled_leg.clob_order_id and not unfilled_leg.is_cancelled:
                        if not self.config.paper_mode and self.clob_manager:
                            self.clob_manager.cancel_order(unfilled_leg.clob_order_id)
                        unfilled_leg.is_cancelled = True
                    self._enter_smart_exit(pair, filled_leg)
                    continue

                # Try repricing
                last_reprice = self._repriced_pairs.get(pair.pair_id, 0)
                if now - last_reprice > rechase_wait:
                    book = fetch_order_book(unfilled_leg.token_id)
                    if book and book.bids.best_price > 0:
                        reprice_info = self.mm_engine.reprice_unfilled_side(
                            pair, unfilled_side, book.bids.best_price)
                        if reprice_info:
                            success = await self.mm_engine.execute_reprice(pair, reprice_info)
                            if success:
                                self._repriced_pairs[pair.pair_id] = now
                                self._rechase_attempts[pair.pair_id] = rechase_count + 1
                                logger.info("🔄 Re-chase #%d/%d for pair #%d %s @ %.2f",
                                            rechase_count + 1, max_rechase,
                                            pair.pair_id, unfilled_side,
                                            reprice_info["new_price"])
                        else:
                            # Can't reprice further (at max price) → stop-loss
                            logger.info("🚫 Pair #%d: can't reprice further → stop-loss",
                                        pair.pair_id)
                            if unfilled_leg.clob_order_id and not unfilled_leg.is_cancelled:
                                if not self.config.paper_mode and self.clob_manager:
                                    self.clob_manager.cancel_order(unfilled_leg.clob_order_id)
                                unfilled_leg.is_cancelled = True
                            self._enter_smart_exit(pair, filled_leg)

            elif rechase_count >= max_rechase:
                # All re-chase attempts exhausted → stop-loss
                logger.info("🚫 Pair #%d: re-chase exhausted (%d attempts) → stop-loss",
                            pair.pair_id, rechase_count)
                if unfilled_leg.clob_order_id and not unfilled_leg.is_cancelled:
                    if not self.config.paper_mode and self.clob_manager:
                        self.clob_manager.cancel_order(unfilled_leg.clob_order_id)
                    unfilled_leg.is_cancelled = True
                self._enter_smart_exit(pair, filled_leg)

    def _enter_smart_exit(self, pair, filled_leg):
        """
        Enter smart exit mode: use Binance price as a leading indicator to decide
        whether to hold for profit or cut losses on a single-side fill.

        Logic:
        - We hold Up tokens and BTC is rising → Polymarket will reprice Up higher → hold/sell at profit
        - We hold Up tokens and BTC is dropping → Up tokens losing value → sell NOW
        - Same logic inverted for Down tokens
        
        Binance leads Polymarket by 5-30 seconds. That's our edge for exit timing.
        """
        crypto_price = self.price_feed.get_price(pair.asset)
        if crypto_price <= 0:
            crypto_price = pair.start_price

        self._smart_exit_pairs[pair.pair_id] = {
            "side": filled_leg.side,              # "Up" or "Down"
            "token_id": filled_leg.token_id,
            "entry_price": filled_leg.filled_price,
            "size": filled_leg.size,
            "baseline_crypto": crypto_price,       # Crypto price when we entered smart exit
            "entered_at": time.time(),
            "sell_order_id": "",
            "sell_price": 0.0,
            "pair": pair,
            "asset": pair.asset,
            "window_end": pair.window_end_ts,
            "best_seen_bid": 0.0,                  # Track best bid we've seen (for trailing)
        }

        logger.info("🧠 SMART EXIT pair #%d: monitoring %s %s (entry %.2f, crypto $%.2f)",
                     pair.pair_id, pair.asset, filled_leg.side,
                     filled_leg.filled_price, crypto_price)

    async def _run_smart_exits(self):
        """
        Monitor Binance prices for all smart-exit pairs and decide: sell for profit or cut loss.
        
        Decision matrix:
        ┌─────────────┬──────────────────────┬────────────────────────┐
        │             │ Crypto moving FOR us │ Crypto moving AGAINST  │
        ├─────────────┼──────────────────────┼────────────────────────┤
        │ Hold Up     │ BTC rising → HOLD    │ BTC dropping → SELL    │
        │ Hold Down   │ BTC dropping → HOLD  │ BTC rising → SELL      │
        └─────────────┴──────────────────────┴────────────────────────┘
        
        Additional rules:
        - If bid > entry + 0.01 → take profit (sell at bid)
        - If bid < entry - stop_loss_cents → cut loss (sell at bid)
        - If < 30s until window expires → emergency sell regardless
        - Trailing: if we've seen a higher bid and it drops back, sell
        """
        sl_cents = getattr(self.config.trading, 'stop_loss_cents', 0.03)
        completed = []

        for pair_id, state in self._smart_exit_pairs.items():
            pair = state["pair"]
            now = time.time()
            time_left = state["window_end"] - now

            # Emergency: window about to expire → sell immediately
            if time_left < 30:
                logger.info("⏰ SMART EXIT #%d: time running out (%ds) → emergency sell",
                            pair_id, int(time_left))
                await self._smart_exit_sell(pair_id, state, reason="time_emergency")
                completed.append(pair_id)
                continue

            # Get current crypto price from Binance (real-time via WebSocket)
            crypto_now = self.price_feed.get_price(state["asset"])
            if crypto_now <= 0:
                continue

            # Get current order book for our token
            book = fetch_order_book(state["token_id"])
            if not book or book.bids.best_price <= 0:
                continue

            current_bid = book.bids.best_price
            entry = state["entry_price"]

            # Track best bid seen (for trailing logic)
            if current_bid > state["best_seen_bid"]:
                state["best_seen_bid"] = current_bid

            # Calculate crypto move since entering smart exit
            baseline = state["baseline_crypto"]
            crypto_move_pct = ((crypto_now - baseline) / baseline) * 100 if baseline > 0 else 0

            # Is the move in our favor?
            if state["side"] == "Up":
                move_favorable = crypto_move_pct > 0.02   # BTC rising = good for Up
                move_against = crypto_move_pct < -0.02    # BTC dropping = bad for Up
            else:
                move_favorable = crypto_move_pct < -0.02  # BTC dropping = good for Down
                move_against = crypto_move_pct > 0.02     # BTC rising = bad for Down

            # DECISION 1: Take profit — bid is above our entry
            if current_bid >= entry + 0.01:
                profit_per_token = current_bid - entry
                logger.info("💰 SMART EXIT #%d: TAKE PROFIT — bid %.2f > entry %.2f (+%.0fc, crypto %+.3f%%)",
                            pair_id, current_bid, entry, profit_per_token * 100, crypto_move_pct)
                await self._smart_exit_sell(pair_id, state, price=current_bid, reason="take_profit")
                completed.append(pair_id)
                continue

            # DECISION 2: Crypto moving against us hard → cut loss NOW
            if move_against and current_bid < entry:
                loss_per_token = entry - current_bid
                logger.info("🛑 SMART EXIT #%d: CUT LOSS — crypto %+.3f%% against, bid %.2f (loss %.0fc/token)",
                            pair_id, crypto_move_pct, current_bid, loss_per_token * 100)
                await self._smart_exit_sell(pair_id, state, price=current_bid, reason="crypto_against")
                completed.append(pair_id)
                continue

            # DECISION 3: Hit max stop loss → cut regardless
            if current_bid <= entry - sl_cents:
                logger.info("🛑 SMART EXIT #%d: MAX STOP LOSS — bid %.2f, entry %.2f (-%0.fc)",
                            pair_id, current_bid, entry, sl_cents * 100)
                await self._smart_exit_sell(pair_id, state, price=current_bid, reason="max_stop_loss")
                completed.append(pair_id)
                continue

            # DECISION 4: Trailing — we saw a higher bid but it's pulling back
            if state["best_seen_bid"] > entry and current_bid < state["best_seen_bid"] - 0.02:
                logger.info("📉 SMART EXIT #%d: TRAILING — best seen %.2f, now %.2f (pullback %.0fc)",
                            pair_id, state["best_seen_bid"], current_bid,
                            (state["best_seen_bid"] - current_bid) * 100)
                await self._smart_exit_sell(pair_id, state, price=current_bid, reason="trailing_stop")
                completed.append(pair_id)
                continue

            # DECISION 5: Crypto is moving in our favor → hold and wait
            if move_favorable:
                # Don't log every tick, only every ~10s
                if int(now) % 10 < 1:
                    logger.debug("🧠 SMART EXIT #%d: holding — crypto %+.3f%% in our favor, bid %.2f",
                                 pair_id, crypto_move_pct, current_bid)

            # DECISION 6: Stale — been in smart exit too long (> 60s with no clear signal)
            if now - state["entered_at"] > 60 and not move_favorable:
                logger.info("⏰ SMART EXIT #%d: stale (%.0fs), exiting at market bid %.2f",
                            pair_id, now - state["entered_at"], current_bid)
                await self._smart_exit_sell(pair_id, state, price=current_bid, reason="stale_timeout")
                completed.append(pair_id)
                continue

        # Clean up completed exits
        for pid in completed:
            del self._smart_exit_pairs[pid]

    async def _smart_exit_sell(self, pair_id, state, price=None, reason=""):
        """Execute the sell for a smart exit."""
        pair = state["pair"]

        if price is None:
            book = fetch_order_book(state["token_id"])
            if book and book.bids.best_price > 0:
                price = book.bids.best_price
            else:
                price = round(max(0.01, state["entry_price"] - 0.03), 2)

        price = round(max(0.01, price), 2)
        pnl_estimate = (price - state["entry_price"]) * state["size"]

        exit_info = {
            "action": "sell",
            "side": state["side"],
            "token_id": state["token_id"],
            "size": state["size"],
            "price": price,
            "reason": f"smart_exit_{reason}",
        }

        logger.info("🧠 SMART EXIT SELL #%d: %s @ %.2f (entry %.2f) | est P&L: $%.3f | reason: %s",
                     pair_id, state["side"], price, state["entry_price"], pnl_estimate, reason)

        await self.mm_engine.execute_early_exit(pair, exit_info)

    def _resolve_expired(self):
        """Resolve pairs whose windows have ended. Only holds both-filled pairs to resolution."""
        for pair in list(self.mm_engine.pairs.values()):
            if pair.status == PairStatus.RESOLVED or pair.status == PairStatus.EXPIRED:
                continue

            if not pair.is_expired:
                continue

            # Cancel any remaining open orders
            for leg in [pair.up_leg, pair.down_leg]:
                if not leg.is_filled and not leg.is_cancelled and leg.clob_order_id:
                    if not self.config.paper_mode:
                        self.clob_manager.cancel_order(leg.clob_order_id)
                    leg.is_cancelled = True
                    leg.cancel_reason = "expired"

            # Only hold to resolution if both sides filled (guaranteed profit)
            if pair.up_leg.is_filled and pair.down_leg.is_filled:
                current_price = self.price_feed.get_price(pair.asset)
                if current_price > 0 and pair.start_price > 0:
                    outcome = "up" if current_price >= pair.start_price else "down"
                    self.mm_engine.resolve_pair(pair, outcome)
                else:
                    logger.warning("Cannot resolve pair #%d: no price data", pair.pair_id)
            elif pair.up_leg.is_filled or pair.down_leg.is_filled:
                # Single-sided fill expired without being exited — emergency market sell
                # This should be rare since _check_early_exits handles it
                filled_leg = pair.up_leg if pair.up_leg.is_filled else pair.down_leg
                logger.warning("⚠️ Pair #%d expired with single fill — emergency exit!",
                               pair.pair_id)
                # Force a market sell at whatever price we can get
                exit_info = {
                    "action": "sell",
                    "side": filled_leg.side,
                    "token_id": filled_leg.token_id,
                    "size": filled_leg.size,
                    "price": round(max(0.01, filled_leg.filled_price - 0.05), 2),
                    "reason": "expired_emergency_exit",
                }
                # Can't await in sync method, so use ensure_future
                asyncio.ensure_future(self.mm_engine.execute_early_exit(pair, exit_info))
            else:
                pair.status = PairStatus.EXPIRED
                pair.resolved_at = time.time()

    def _cancel_all_open(self):
        """Cancel all open orders (kill switch / shutdown)."""
        for pair in self.mm_engine.get_active_pairs():
            self.mm_engine.cancel_pair(pair, reason="shutdown")

        if self.clob_manager and not self.config.paper_mode:
            self.clob_manager.cancel_all()

    def _write_health(self):
        """Write health data for dashboard."""
        stats = self.mm_engine.stats
        active_pairs = self.mm_engine.get_active_pairs()

        health_data = {
            "status": "running",
            "mode": "paper" if self.config.paper_mode else ("dry_run" if self.config.dry_run else "live"),
            "strategy": "market_maker_v2",
            "uptime_seconds": int(time.time() - self._start_time),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pairs_created": stats["pairs_created"],
            "both_filled": stats["both_filled"],
            "shield_saves": stats["shield_saves"],
            "shield_failures": stats["shield_failures"],
            "early_exits": stats["early_exits"],
            "take_profits": stats["take_profits"],
            "stop_losses": stats["stop_losses"],
            "time_cutoffs": stats["time_cutoffs"],
            "total_pnl": round(stats["total_pnl"], 4),
            "wins": stats["wins"],
            "losses": stats["losses"],
            "active_pairs": len(active_pairs),
            "total_exposure": round(self.mm_engine.get_total_exposure(), 2),
            "active_pair_details": [p.to_dict() for p in active_pairs],
            "kill_switch": self.kill_switch.is_triggered,
            "binance_connected": getattr(self.price_feed, 'is_connected', True),
            "polymarket_connected": True,  # Using REST, always available
            "binance_ws_latency_ms": getattr(self.price_feed, 'last_latency_ms', 0),
            "arb": self.arb_engine.get_stats(),
            "arb_positions": [p.to_dict() for p in self.arb_engine.get_active_arb_positions()],
            "snipe": self.snipe_engine.get_stats(),
            "polymarket_rest_latency_ms": getattr(self, '_last_poly_latency_ms', 0),
        }

        health_path = os.path.join(DATA_DIR, "health.json")
        try:
            with open(health_path, "w") as f:
                json.dump(health_data, f, indent=2)
        except Exception as e:
            logger.error("Failed to write health data: %s", e)

    def _restore_state(self):
        """Restore state from previous run (for crash recovery)."""
        state_path = os.path.join(DATA_DIR, "mm_state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
                self.mm_engine.stats = state.get("stats", self.mm_engine.stats)
                self.mm_engine._next_pair_id = state.get("next_pair_id", 1)
                logger.info("Restored state: P&L=$%.4f, %dW/%dL",
                             self.mm_engine.stats["total_pnl"],
                             self.mm_engine.stats["wins"],
                             self.mm_engine.stats["losses"])
            except Exception as e:
                logger.warning("Failed to restore state: %s", e)

    def _save_state(self):
        """Save state for crash recovery."""
        state = {
            "stats": self.mm_engine.stats,
            "next_pair_id": self.mm_engine._next_pair_id,
            "pairs": {k: p.to_dict() for k, p in self.mm_engine.pairs.items()},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        state_path = os.path.join(DATA_DIR, "mm_state.json")
        try:
            tmp_path = state_path + ".tmp"
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, state_path)
        except Exception as e:
            logger.error("Failed to save state: %s", e)

    async def _graceful_shutdown(self):
        """Clean shutdown: cancel orders, save state, log summary."""
        logger.info("Starting graceful shutdown...")
        self.health_monitor.set_status("shutting_down")

        # Resolve any expired pairs
        self._resolve_expired()

        # Cancel all remaining open orders
        self._cancel_all_open()

        # Save state
        self._save_state()

        # Final summary
        stats = self.mm_engine.stats
        logger.info("=" * 70)
        logger.info("  MARKET MAKER V2 — SESSION SUMMARY")
        logger.info("  Duration: %.0f seconds", time.time() - self._start_time)
        logger.info("  Pairs created: %d", stats["pairs_created"])
        logger.info("  Both sides filled: %d", stats["both_filled"])
        logger.info("  Shield saves: %d | Shield failures: %d",
                     stats["shield_saves"], stats["shield_failures"])
        logger.info("  Wins: %d | Losses: %d", stats["wins"], stats["losses"])
        logger.info("  Total P&L: $%.4f", stats["total_pnl"])
        arb_stats = self.arb_engine.get_stats()
        logger.info("  --- ARB ---")
        logger.info("  Opportunities: %d | Trades: %d | Filled: %d",
                     arb_stats["opportunities_detected"],
                     arb_stats["trades_placed"],
                     arb_stats["trades_filled"])
        logger.info("  Arb W/L: %d/%d | Arb P&L: $%.4f",
                     arb_stats["wins"], arb_stats["losses"],
                     arb_stats["total_pnl"])
        snipe_stats = self.snipe_engine.get_stats()
        logger.info("  --- SNIPE ---")
        logger.info("  Snipes: %d | W/L: %d/%d | P&L: $%.4f",
                     snipe_stats["trades"],
                     snipe_stats["wins"], snipe_stats["losses"],
                     snipe_stats["total_pnl"])
        logger.info("=" * 70)

    def _setup_signal_handlers(self):
        """Handle SIGTERM/SIGINT for graceful shutdown."""
        def handler(sig, frame):
            if not self._shutting_down:
                self._shutting_down = True
                logger.info("Received signal %s, shutting down...", sig)

        signal.signal(signal.SIGTERM, handler)
        signal.signal(signal.SIGINT, handler)


async def main():
    config = load_config()
    trader = TraderV2(config)
    await trader.run(duration_seconds=0)  # Run continuously (0 = no limit)


if __name__ == "__main__":
    asyncio.run(main())
