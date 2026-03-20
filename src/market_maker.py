#!/usr/bin/env python3
"""
Market Maker Engine — Two-sided market making with latency shield.

Core strategy:
  1. Post BUY orders on both Up AND Down sides of a market
  2. If both fill: guaranteed profit (cost < $1.00, one side pays $1.00)
  3. Use Binance real-time price as early warning to cancel the losing side
     before it gets filled against us

The Binance feed is DEFENSE (avoid bad fills), not offense (don't bet on direction).
"""
import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.logger import get_logger
from src.metrics_collector import log_pair_result, update_hourly_stats

logger = get_logger("market_maker")


class PairStatus(Enum):
    """Status of an order pair (Up + Down for one market window)."""
    PENDING = "pending"           # Orders not yet placed
    OPEN = "open"                 # Both orders live on CLOB
    PARTIAL_UP = "partial_up"     # Only Up order filled
    PARTIAL_DOWN = "partial_down" # Only Down order filled
    BOTH_FILLED = "both_filled"   # Both sides filled (guaranteed profit)
    SHIELD_CANCELLED = "shield_cancelled"  # Losing side cancelled by shield
    RESOLVED = "resolved"         # Window ended, P&L calculated
    EXPIRED = "expired"           # Window ended with no fills
    CANCELLED = "cancelled"       # Manually cancelled


@dataclass
class OrderLeg:
    """One side of a market making pair."""
    side: str                     # "Up" or "Down"
    token_id: str
    price: float                  # Our bid price
    size: float                   # Number of tokens
    clob_order_id: str = ""
    is_filled: bool = False
    filled_price: float = 0.0
    filled_at: float = 0.0
    is_cancelled: bool = False
    cancel_reason: str = ""


@dataclass
class MarketPair:
    """A two-sided order pair for one market window."""
    pair_id: int
    window_key: str
    asset: str
    timeframe: str
    window_start_ts: int
    window_end_ts: int
    start_price: float            # Crypto price at window start
    up_leg: OrderLeg
    down_leg: OrderLeg
    status: PairStatus = PairStatus.PENDING
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0
    pnl: float = 0.0
    total_cost: float = 0.0       # What we paid for both sides
    shield_triggered: bool = False
    shield_latency_ms: float = 0.0

    @property
    def spread_profit(self) -> float:
        """Guaranteed profit if both sides fill."""
        return 1.0 - (self.up_leg.price + self.down_leg.price)

    @property
    def time_remaining(self) -> float:
        return max(0, self.window_end_ts - time.time())

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.window_end_ts

    def to_dict(self) -> dict:
        return {
            "pair_id": self.pair_id,
            "window_key": self.window_key,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "status": self.status.value,
            "start_price": self.start_price,
            "up_price": self.up_leg.price,
            "down_price": self.down_leg.price,
            "up_filled": self.up_leg.is_filled,
            "down_filled": self.down_leg.is_filled,
            "spread_profit": round(self.spread_profit, 4),
            "total_cost": round(self.total_cost, 4),
            "pnl": round(self.pnl, 4),
            "shield_triggered": self.shield_triggered,
            "shield_latency_ms": round(self.shield_latency_ms, 1),
            "time_remaining": round(self.time_remaining, 1),
            "created_at": self.created_at,
        }


class MarketMakerEngine:
    """
    Two-sided market making engine with Binance latency shield.

    Args:
        clob_manager: CLOB client for order placement/cancellation
        config: Bot configuration
    """

    def __init__(self, clob_manager, config):
        self.clob = clob_manager
        self.config = config
        self.pairs: dict[str, MarketPair] = {}  # window_key → MarketPair
        self._next_pair_id = 1
        self._shield_threshold = getattr(config, 'shield_threshold', 0.001)  # 0.1% default

        # Early exit parameters
        self._stop_loss_ticks = 0.03       # Exit if token drops $0.03 below fill price
        self._take_profit_ticks = 0.03     # Exit if token rises $0.03 above fill price
        self._time_cutoff_seconds = 120    # Exit after 2 minutes if single-sided
        self._price_improvement = 0.01     # Post 1c above best bid each side (2c tighter, fills faster)

        # Stats
        self.stats = {
            "pairs_created": 0,
            "both_filled": 0,
            "single_filled": 0,
            "shield_saves": 0,
            "shield_failures": 0,
            "early_exits": 0,
            "stop_losses": 0,
            "take_profits": 0,
            "time_cutoffs": 0,
            "total_pnl": 0.0,
            "wins": 0,
            "losses": 0,
        }


    def reprice_unfilled_side(self, pair: MarketPair, unfilled_side: str,
                               current_best_bid: float) -> Optional[dict]:
        """
        After one side fills, try to reprice the unfilled side more aggressively
        while keeping total cost < $1.00 (still guaranteed profit).
        
        Args:
            pair: The market pair
            unfilled_side: "Up" or "Down" — which side hasn't filled
            current_best_bid: Current best bid on the unfilled side's order book
            
        Returns:
            Dict with reprice instructions, or None if repricing won't help
        """
        if unfilled_side == "Up":
            filled_leg = pair.down_leg
            unfilled_leg = pair.up_leg
        else:
            filled_leg = pair.up_leg
            unfilled_leg = pair.down_leg
        
        # Max price we can pay and still profit (at least $0.01/token)
        min_profit_per_token = 0.00  # Allow repricing all the way to break-even
        max_price = round(1.0 - filled_leg.filled_price - min_profit_per_token, 2)
        
        # Don't reprice above $0.99
        max_price = min(max_price, 0.99)
        
        if max_price <= unfilled_leg.price:
            # Already at or above max — can't improve
            return None
        
        # New price: best bid + 1 cent, capped at max_price
        new_price = round(current_best_bid + self._price_improvement, 2)
        new_price = min(new_price, max_price)
        
        # Don't reprice if the improvement is negligible (< 1 cent)
        if new_price <= unfilled_leg.price:
            return None
        
        new_total = filled_leg.filled_price + new_price
        new_profit = 1.0 - new_total
        
        logger.info("🔄 Repricing pair #%d %s: %.2f → %.2f "
                     "(filled %s@%.2f, new total=%.2f, profit=$%.4f/token)",
                     pair.pair_id, unfilled_side,
                     unfilled_leg.price, new_price,
                     "Down" if unfilled_side == "Up" else "Up",
                     filled_leg.filled_price, new_total, new_profit)
        
        return {
            "side": unfilled_side,
            "token_id": unfilled_leg.token_id,
            "old_order_id": unfilled_leg.clob_order_id,
            "old_price": unfilled_leg.price,
            "new_price": new_price,
            "size": unfilled_leg.size,
            "max_price": max_price,
            "new_profit": new_profit,
        }

    async def execute_reprice(self, pair: MarketPair, reprice_info: dict) -> bool:
        """Execute the reprice: cancel old order, place new one at better price."""
        if self.config.paper_mode:
            logger.info("📝 [PAPER] Repriced pair #%d %s: %.2f → %.2f",
                         pair.pair_id, reprice_info["side"],
                         reprice_info["old_price"], reprice_info["new_price"])
            return True
        
        if self.config.dry_run:
            logger.info("🔍 [DRY RUN] Would reprice pair #%d %s: %.2f → %.2f",
                         pair.pair_id, reprice_info["side"],
                         reprice_info["old_price"], reprice_info["new_price"])
            return False
        
        # Cancel old order
        if reprice_info["old_order_id"]:
            self.clob.cancel_order(reprice_info["old_order_id"])
        
        # Place new order at better price
        result = self.clob.place_order(
            token_id=reprice_info["token_id"],
            side="BUY",
            price=reprice_info["new_price"],
            size=reprice_info["size"],
        )
        
        if result.get("success"):
            # Update the leg
            if reprice_info["side"] == "Up":
                pair.up_leg.price = reprice_info["new_price"]
                pair.up_leg.clob_order_id = result["orderID"]
            else:
                pair.down_leg.price = reprice_info["new_price"]
                pair.down_leg.clob_order_id = result["orderID"]
            
            logger.info("✅ Repriced pair #%d %s: %.2f → %.2f (order %s)",
                         pair.pair_id, reprice_info["side"],
                         reprice_info["old_price"], reprice_info["new_price"],
                         result["orderID"][:12])
            return True
        else:
            logger.error("❌ Reprice FAILED for pair #%d %s: %s",
                         pair.pair_id, reprice_info["side"],
                         result.get("error", "unknown"))
            return False

    def check_early_exit(self, pair: MarketPair,
                         up_market_price: float, down_market_price: float) -> Optional[dict]:
        """
        Check if a single-sided fill should be exited early.

        Only applies when exactly one side is filled and the other is cancelled/unfilled.
        Returns exit instruction dict or None if no exit needed.
        """
        # Only check single-sided fills
        up_filled = pair.up_leg.is_filled
        down_filled = pair.down_leg.is_filled

        if up_filled and down_filled:
            return None  # Both filled — hold to resolution, guaranteed profit
        if not up_filled and not down_filled:
            return None  # Neither filled — nothing to exit

        # Determine which leg is filled
        if up_filled:
            filled_leg = pair.up_leg
            market_price = up_market_price  # Current bid price for Up token
            side_name = "Up"
        else:
            filled_leg = pair.down_leg
            market_price = down_market_price  # Current bid price for Down token
            side_name = "Down"

        if market_price <= 0:
            return None

        fill_price = filled_leg.filled_price
        time_since_fill = time.time() - filled_leg.filled_at

        # Take-profit: market moved in our favor
        if market_price >= fill_price + self._take_profit_ticks:
            profit = (market_price - fill_price) * filled_leg.size
            logger.info("📈 Take-profit triggered for pair #%d %s: "
                         "filled@%.2f → market@%.2f (+$%.4f)",
                         pair.pair_id, side_name, fill_price, market_price, profit)
            self.stats["take_profits"] += 1
            return {
                "action": "sell",
                "side": side_name,
                "token_id": filled_leg.token_id,
                "size": filled_leg.size,
                "price": market_price,
                "reason": "take_profit",
            }

        # Stop-loss: market moved against us
        if market_price <= fill_price - self._stop_loss_ticks:
            loss = (fill_price - market_price) * filled_leg.size
            logger.info("📉 Stop-loss triggered for pair #%d %s: "
                         "filled@%.2f → market@%.2f (-$%.4f)",
                         pair.pair_id, side_name, fill_price, market_price, loss)
            self.stats["stop_losses"] += 1
            return {
                "action": "sell",
                "side": side_name,
                "token_id": filled_leg.token_id,
                "size": filled_leg.size,
                "price": market_price,
                "reason": "stop_loss",
            }

        # Time cutoff: too long without resolution
        if time_since_fill > self._time_cutoff_seconds:
            pnl = (market_price - fill_price) * filled_leg.size
            logger.info("⏰ Time cutoff for pair #%d %s: %.0fs since fill, "
                         "market@%.2f (P&L: $%.4f)",
                         pair.pair_id, side_name, time_since_fill, market_price, pnl)
            self.stats["time_cutoffs"] += 1
            return {
                "action": "sell",
                "side": side_name,
                "token_id": filled_leg.token_id,
                "size": filled_leg.size,
                "price": market_price,
                "reason": "time_cutoff",
            }

        return None

    async def execute_early_exit(self, pair: MarketPair, exit_info: dict) -> bool:
        """Execute an early exit (taker sell) for a single-sided fill."""
        if self.config.paper_mode:
            # Simulate the exit
            sell_price = exit_info["price"]
            taker_fee_rate = 0.0156  # ~1.56% at 50% prices
            fee = sell_price * exit_info["size"] * taker_fee_rate
            proceeds = sell_price * exit_info["size"] - fee

            leg = pair.up_leg if exit_info["side"] == "Up" else pair.down_leg
            cost = leg.filled_price * leg.size
            pnl = proceeds - cost

            pair.pnl = pnl
            pair.status = PairStatus.RESOLVED
            pair.resolved_at = time.time()
            self.stats["total_pnl"] += pnl
            self.stats["early_exits"] += 1

            if pnl >= 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1

            logger.info("🔄 [PAPER] Early exit pair #%d: sold %s @%.2f, "
                         "fee=$%.4f, P&L=$%.4f (%s)",
                         pair.pair_id, exit_info["side"], sell_price,
                         fee, pnl, exit_info["reason"])
            return True

        if self.config.dry_run:
            logger.info("🔍 [DRY RUN] Would early exit pair #%d: sell %s @%.2f (%s)",
                         pair.pair_id, exit_info["side"], exit_info["price"],
                         exit_info["reason"])
            return False

        # Live mode: place a taker sell order
        result = self.clob.place_order(
            token_id=exit_info["token_id"],
            side="SELL",
            price=exit_info["price"],
            size=exit_info["size"],
        )

        if result.get("success"):
            leg = pair.up_leg if exit_info["side"] == "Up" else pair.down_leg
            sell_price = exit_info["price"]
            taker_fee_rate = 0.0156
            fee = sell_price * exit_info["size"] * taker_fee_rate
            proceeds = sell_price * exit_info["size"] - fee
            cost = leg.filled_price * leg.size
            pnl = proceeds - cost

            pair.pnl = pnl
            pair.status = PairStatus.RESOLVED
            pair.resolved_at = time.time()
            self.stats["total_pnl"] += pnl
            self.stats["early_exits"] += 1

            if pnl >= 0:
                self.stats["wins"] += 1
            else:
                self.stats["losses"] += 1

            logger.info("🔄 Early exit pair #%d: sold %s @%.2f, "
                         "fee=$%.4f, P&L=$%.4f (%s)",
                         pair.pair_id, exit_info["side"], sell_price,
                         fee, pnl, exit_info["reason"])
            return True
        else:
            logger.error("❌ Early exit FAILED for pair #%d: %s",
                         pair.pair_id, result.get("error", "unknown"))
            return False

    def evaluate_spread(self, up_best_bid: float, down_best_bid: float,
                         min_profit: float = 0.02) -> dict:
        """
        Evaluate if placing both sides is profitable.

        We post BUY orders at or just below the best bid on each side.
        If up_bid + down_bid < 1.0 - min_profit, it's worth trading.

        Returns dict with 'tradeable', 'up_price', 'down_price', 'expected_profit'.
        """
        if up_best_bid <= 0 or down_best_bid <= 0:
            return {"tradeable": False, "reason": "no bids"}

        # Our prices: best bid + price improvement (to be at front of queue)
        up_price = round(up_best_bid + self._price_improvement, 2)
        down_price = round(down_best_bid + self._price_improvement, 2)
        total_cost = up_price + down_price
        profit = 1.0 - total_cost

        if profit < min_profit:
            return {
                "tradeable": False,
                "reason": f"spread too tight ({profit:.4f} < {min_profit})",
                "total_cost": total_cost,
                "profit": profit,
            }

        return {
            "tradeable": True,
            "up_price": up_price,
            "down_price": down_price,
            "total_cost": total_cost,
            "profit": profit,
        }

    def create_pair(self, window_key: str, asset: str, timeframe: str,
                    window_start_ts: int, window_end_ts: int,
                    start_price: float,
                    up_token_id: str, down_token_id: str,
                    up_price: float, down_price: float,
                    size: float) -> MarketPair:
        """Create a new market making pair (both sides)."""
        pair = MarketPair(
            pair_id=self._next_pair_id,
            window_key=window_key,
            asset=asset,
            timeframe=timeframe,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            start_price=start_price,
            up_leg=OrderLeg(side="Up", token_id=up_token_id, price=up_price, size=size),
            down_leg=OrderLeg(side="Down", token_id=down_token_id, price=down_price, size=size),
            total_cost=(up_price + down_price) * size,
        )
        self._next_pair_id += 1
        self.pairs[window_key] = pair
        self.stats["pairs_created"] += 1

        logger.info("📊 Pair #%d created: %s %s | Up@%.2f + Down@%.2f = %.2f cost | "
                     "Spread profit: $%.4f/token | Size: %.1f",
                     pair.pair_id, asset, timeframe,
                     up_price, down_price, up_price + down_price,
                     pair.spread_profit, size)

        return pair

    async def place_pair_orders(self, pair: MarketPair) -> bool:
        """Place both buy orders for a pair on the CLOB."""
        if self.config.paper_mode:
            pair.up_leg.clob_order_id = f"paper_up_{pair.pair_id}"
            pair.down_leg.clob_order_id = f"paper_down_{pair.pair_id}"
            pair.status = PairStatus.OPEN
            logger.info("📝 [PAPER] Pair #%d orders placed", pair.pair_id)
            return True

        if self.config.dry_run:
            logger.info("🔍 [DRY RUN] Would place pair #%d: "
                         "BUY Up @%.2f + BUY Down @%.2f",
                         pair.pair_id, pair.up_leg.price, pair.down_leg.price)
            return False

        # Place Up order FIRST (more likely to fail due to API issues)
        # Note: clob_client.place_order already retries 3x on connection failures
        up_result = self.clob.place_order(
            token_id=pair.up_leg.token_id,
            side="BUY",
            price=pair.up_leg.price,
            size=pair.up_leg.size,
        )

        if up_result.get("success"):
            pair.up_leg.clob_order_id = up_result["orderID"]
        else:
            logger.error("❌ Failed to place Up order for pair #%d: %s",
                         pair.pair_id, up_result.get("error", "unknown"))
            return False

        # Place Down order (only if Up succeeded)
        down_result = self.clob.place_order(
            token_id=pair.down_leg.token_id,
            side="BUY",
            price=pair.down_leg.price,
            size=pair.down_leg.size,
        )

        if down_result.get("success"):
            pair.down_leg.clob_order_id = down_result["orderID"]
        else:
            # Down order failed — cancel the Up order
            logger.error("❌ Failed to place Down order for pair #%d, cancelling Up",
                         pair.pair_id)
            if pair.up_leg.clob_order_id:
                self.clob.cancel_order(pair.up_leg.clob_order_id)
            return False

        pair.status = PairStatus.OPEN
        logger.info("✅ Pair #%d orders live: Up=%s Down=%s",
                     pair.pair_id,
                     pair.up_leg.clob_order_id[:12],
                     pair.down_leg.clob_order_id[:12])
        return True

    def shield_check(self, pair: MarketPair, current_crypto_price: float) -> Optional[str]:
        """
        Check if Binance price has moved enough to trigger the shield.

        Returns: "Up" or "Down" (the losing side to cancel), or None if no action needed.
        """
        if pair.start_price <= 0:
            return None

        pct_move = (current_crypto_price - pair.start_price) / pair.start_price

        if abs(pct_move) < self._shield_threshold:
            return None

        # Price went UP → Down is losing → cancel Down
        if pct_move > 0:
            return "Down"
        else:
            return "Up"

    async def execute_shield(self, pair: MarketPair, cancel_side: str) -> bool:
        """
        Cancel the losing side order. This is the critical latency path.

        Args:
            pair: The market pair
            cancel_side: "Up" or "Down" — which side to cancel
        """
        shield_start = time.time()

        leg = pair.up_leg if cancel_side == "Up" else pair.down_leg
        keep_leg = pair.down_leg if cancel_side == "Up" else pair.up_leg

        # Don't cancel if already filled (too late)
        if leg.is_filled:
            logger.warning("⚠️ Shield too late — %s already filled for pair #%d",
                           cancel_side, pair.pair_id)
            self.stats["shield_failures"] += 1
            return False

        # Don't cancel if already cancelled
        if leg.is_cancelled:
            return True

        # Cancel the order
        if not self.config.paper_mode and leg.clob_order_id:
            success = self.clob.cancel_order(leg.clob_order_id)
            if not success:
                logger.error("❌ Shield cancel FAILED for pair #%d %s",
                             pair.pair_id, cancel_side)
                self.stats["shield_failures"] += 1
                return False

        leg.is_cancelled = True
        leg.cancel_reason = "shield"
        pair.shield_triggered = True
        pair.shield_latency_ms = (time.time() - shield_start) * 1000

        # Update pair status
        if keep_leg.is_filled:
            pair.status = PairStatus.SHIELD_CANCELLED
        else:
            pair.status = PairStatus.SHIELD_CANCELLED

        self.stats["shield_saves"] += 1

        logger.info("🛡️ Shield activated: cancelled %s for pair #%d "
                     "(latency: %.1fms, crypto moved %.3f%%)",
                     cancel_side, pair.pair_id, pair.shield_latency_ms,
                     ((pair.start_price - pair.start_price) / pair.start_price) * 100
                     if pair.start_price > 0 else 0)

        return True

    def update_fill_status(self, pair: MarketPair, up_filled: bool, down_filled: bool,
                           up_fill_price: float = 0, down_fill_price: float = 0):
        """Update fill status based on CLOB polling or WebSocket updates."""
        if up_filled and not pair.up_leg.is_filled:
            pair.up_leg.is_filled = True
            pair.up_leg.filled_price = up_fill_price or pair.up_leg.price
            pair.up_leg.filled_at = time.time()
            logger.info("💰 Up leg filled for pair #%d @ %.2f",
                         pair.pair_id, pair.up_leg.filled_price)

        if down_filled and not pair.down_leg.is_filled:
            pair.down_leg.is_filled = True
            pair.down_leg.filled_price = down_fill_price or pair.down_leg.price
            pair.down_leg.filled_at = time.time()
            logger.info("💰 Down leg filled for pair #%d @ %.2f",
                         pair.pair_id, pair.down_leg.filled_price)

        # Update pair status
        if pair.up_leg.is_filled and pair.down_leg.is_filled:
            pair.status = PairStatus.BOTH_FILLED
            pair.total_cost = (pair.up_leg.filled_price + pair.down_leg.filled_price) * pair.up_leg.size
            self.stats["both_filled"] += 1
            logger.info("🎯 BOTH SIDES FILLED for pair #%d! "
                         "Cost: $%.4f/token | Guaranteed profit: $%.4f/token",
                         pair.pair_id,
                         pair.up_leg.filled_price + pair.down_leg.filled_price,
                         1.0 - (pair.up_leg.filled_price + pair.down_leg.filled_price))
        elif pair.up_leg.is_filled and not pair.down_leg.is_filled:
            pair.status = PairStatus.PARTIAL_UP
        elif pair.down_leg.is_filled and not pair.up_leg.is_filled:
            pair.status = PairStatus.PARTIAL_DOWN

    def resolve_pair(self, pair: MarketPair, outcome: str):
        """
        Resolve a pair after the window ends.

        Args:
            outcome: "up" or "down" — what actually happened
        """
        if pair.status == PairStatus.RESOLVED:
            return

        winner = "Up" if outcome.lower() == "up" else "Down"
        pnl = 0.0

        if pair.up_leg.is_filled and pair.down_leg.is_filled:
            # Both filled — guaranteed profit
            cost = (pair.up_leg.filled_price + pair.down_leg.filled_price) * pair.up_leg.size
            payout = 1.0 * pair.up_leg.size  # Winner pays $1/token
            pnl = payout - cost
            logger.info("✅ Pair #%d resolved (BOTH FILLED): %s won | "
                         "Cost: $%.4f | Payout: $%.4f | P&L: $%.4f",
                         pair.pair_id, winner, cost, payout, pnl)

        elif pair.up_leg.is_filled and not pair.down_leg.is_filled:
            # Only Up filled
            if winner == "Up":
                pnl = (1.0 - pair.up_leg.filled_price) * pair.up_leg.size
                logger.info("✅ Pair #%d resolved (Up only, WON): P&L: $%.4f",
                             pair.pair_id, pnl)
            else:
                pnl = -pair.up_leg.filled_price * pair.up_leg.size
                logger.info("❌ Pair #%d resolved (Up only, LOST): P&L: $%.4f",
                             pair.pair_id, pnl)

        elif pair.down_leg.is_filled and not pair.up_leg.is_filled:
            # Only Down filled
            if winner == "Down":
                pnl = (1.0 - pair.down_leg.filled_price) * pair.down_leg.size
                logger.info("✅ Pair #%d resolved (Down only, WON): P&L: $%.4f",
                             pair.pair_id, pnl)
            else:
                pnl = -pair.down_leg.filled_price * pair.down_leg.size
                logger.info("❌ Pair #%d resolved (Down only, LOST): P&L: $%.4f",
                             pair.pair_id, pnl)

        else:
            # Neither filled
            logger.info("⏭️ Pair #%d expired with no fills", pair.pair_id)
            pair.status = PairStatus.EXPIRED
            pair.resolved_at = time.time()
            return

        pair.pnl = pnl
        pair.status = PairStatus.RESOLVED
        pair.resolved_at = time.time()
        self.stats["total_pnl"] += pnl

        if pnl >= 0:
            self.stats["wins"] += 1
        else:
            self.stats["losses"] += 1

        # Log metrics for analysis
        try:
            log_pair_result(pair.to_dict())
            update_hourly_stats()
        except Exception:
            pass  # Don't crash on metrics failure

    def cancel_pair(self, pair: MarketPair, reason: str = "manual"):
        """Cancel both sides of a pair."""
        for leg in [pair.up_leg, pair.down_leg]:
            if not leg.is_filled and not leg.is_cancelled and leg.clob_order_id:
                if not self.config.paper_mode:
                    self.clob.cancel_order(leg.clob_order_id)
                leg.is_cancelled = True
                leg.cancel_reason = reason

        pair.status = PairStatus.CANCELLED
        logger.info("🚫 Pair #%d cancelled: %s", pair.pair_id, reason)

    def get_active_pairs(self) -> list[MarketPair]:
        """Get all pairs that still have open/filled positions."""
        return [p for p in self.pairs.values()
                if p.status in (PairStatus.OPEN, PairStatus.PARTIAL_UP,
                                PairStatus.PARTIAL_DOWN, PairStatus.BOTH_FILLED,
                                PairStatus.SHIELD_CANCELLED)]

    def get_total_exposure(self) -> float:
        """Get total dollar exposure across all active pairs."""
        total = 0.0
        for pair in self.get_active_pairs():
            if pair.up_leg.is_filled:
                total += pair.up_leg.filled_price * pair.up_leg.size
            elif not pair.up_leg.is_cancelled:
                total += pair.up_leg.price * pair.up_leg.size
            if pair.down_leg.is_filled:
                total += pair.down_leg.filled_price * pair.down_leg.size
            elif not pair.down_leg.is_cancelled:
                total += pair.down_leg.price * pair.down_leg.size
        return total

    def summary(self) -> str:
        """Return a summary of market maker stats."""
        return (
            f"Pairs: {self.stats['pairs_created']} created | "
            f"Both filled: {self.stats['both_filled']} | "
            f"Single: {self.stats['single_filled']} | "
            f"Shield saves: {self.stats['shield_saves']} | "
            f"Shield fails: {self.stats['shield_failures']} | "
            f"P&L: ${self.stats['total_pnl']:.4f} | "
            f"W/L: {self.stats['wins']}/{self.stats['losses']}"
        )
