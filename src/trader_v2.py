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

        # Shutdown handling
        self._shutting_down = False
        self._start_time = 0

        # Track which windows we've already created pairs for
        self._traded_windows = set()

        # Track repricing state (don't reprice the same pair repeatedly)
        self._repriced_pairs = {}  # pair_id → last_reprice_time

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

        # Start data feeds (Polymarket WS disabled — using REST for order books)
        price_task = asyncio.create_task(self.price_feed.connect())
        poly_task = None  # Disabled: WS keeps dropping (code 1006)

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

                # Periodic market scan
                if now - last_scan > self.config.timing.scan_interval:
                    await self.window_tracker.scan()
                    last_scan = now

                # Arb engine: check for latency arb opportunities (every tick)
                await self.arb_engine.check_opportunities()

                # Evaluate and place new pairs (spread capture)
                if self.config.trading.spread_capture_enabled:
                    await self._evaluate_markets()

                # Shield check (every iteration — speed matters)
                self._run_shield_checks()

                # Check order fills
                if now - last_order_check > self.config.timing.order_check_interval:
                    await self._check_fills()
                    await self._check_early_exits()
                    last_order_check = now

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
            try:
                await price_task
            except asyncio.CancelledError:
                pass

    async def _evaluate_markets(self):
        """Check all active windows for profitable spread opportunities."""
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

            # Fetch order books for both sides (track latency)
            t0 = time.time()
            up_book = fetch_order_book(window.up_token_id)
            down_book = fetch_order_book(window.down_token_id)
            self._last_poly_latency_ms = (time.time() - t0) * 500  # avg per call

            if not up_book or not down_book:
                continue

            up_best_bid = up_book.bids.best_price
            down_best_bid = down_book.bids.best_price

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

            # Create and place the pair
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
                size=self.config.trading.order_size,
            )

            # Always track window to prevent duplicate pair creation
            self._traded_windows.add(window.key)
            success = await self.mm_engine.place_pair_orders(pair)
            if success:
                logger.info("🚀 Pair placed for %s %s: Up@%.2f + Down@%.2f = $%.4f profit/token",
                             window.asset, window.timeframe,
                             evaluation["up_price"], evaluation["down_price"],
                             evaluation["profit"])

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
        """Poll CLOB for order fill status."""
        for pair in self.mm_engine.get_active_pairs():
            if pair.status == PairStatus.BOTH_FILLED:
                continue

            up_filled = pair.up_leg.is_filled
            down_filled = pair.down_leg.is_filled

            if self.config.paper_mode:
                # Paper mode: simulate fills based on time
                # In paper mode, assume fills happen if order is open for > 5 seconds
                now = time.time()
                if not up_filled and not pair.up_leg.is_cancelled and now - pair.created_at > 5:
                    up_filled = True
                if not down_filled and not pair.down_leg.is_cancelled and now - pair.created_at > 5:
                    down_filled = True
            else:
                # Live mode: check CLOB order status
                if not up_filled and not pair.up_leg.is_cancelled and pair.up_leg.clob_order_id:
                    status = self.clob_manager.get_order_status(pair.up_leg.clob_order_id)
                    if status.get("status") == "FILLED" or float(status.get("size_matched", 0) or 0) > 0:
                        up_filled = True

                if not down_filled and not pair.down_leg.is_cancelled and pair.down_leg.clob_order_id:
                    status = self.clob_manager.get_order_status(pair.down_leg.clob_order_id)
                    if status.get("status") == "FILLED" or float(status.get("size_matched", 0) or 0) > 0:
                        down_filled = True

            self.mm_engine.update_fill_status(pair, up_filled, down_filled)

    async def _check_early_exits(self):
        """Single-sided fills: reprice the unfilled side to get both filled.
        
        NO stop-loss, NO panic selling. Only strategy:
        1. Reprice unfilled side more aggressively (total still < $1.00)
        2. If window expires with one side, just let it resolve
        """
        for pair in self.mm_engine.get_active_pairs():
            up_filled = pair.up_leg.is_filled
            down_filled = pair.down_leg.is_filled

            if (up_filled and down_filled) or (not up_filled and not down_filled):
                continue

            # Determine which side is unfilled
            if up_filled:
                unfilled_side = "Down"
                unfilled_leg = pair.down_leg
            else:
                unfilled_side = "Up"
                unfilled_leg = pair.up_leg

            # Skip if unfilled side was cancelled (e.g. by shield)
            if unfilled_leg.is_cancelled:
                continue

            # Reprice every 10 seconds to chase the market
            now = time.time()
            last_reprice = self._repriced_pairs.get(pair.pair_id, 0)
            time_until_expiry = pair.window_end_ts - now

            # Stop repricing with <15s left (order won't fill in time)
            if time_until_expiry < 15:
                continue

            if now - last_reprice > 10:
                book = fetch_order_book(unfilled_leg.token_id)
                if book and book.bids.best_price > 0:
                    reprice_info = self.mm_engine.reprice_unfilled_side(
                        pair, unfilled_side, book.bids.best_price)
                    if reprice_info:
                        success = await self.mm_engine.execute_reprice(pair, reprice_info)
                        if success:
                            self._repriced_pairs[pair.pair_id] = now

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
                # Single-sided fill that wasn't exited early — force exit now
                # This shouldn't happen often since _check_early_exits runs frequently
                logger.warning("Pair #%d expired with single fill — should have been exited early",
                               pair.pair_id)
                current_price = self.price_feed.get_price(pair.asset)
                if current_price > 0 and pair.start_price > 0:
                    outcome = "up" if current_price >= pair.start_price else "down"
                    self.mm_engine.resolve_pair(pair, outcome)
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
