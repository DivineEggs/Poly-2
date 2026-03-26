#!/usr/bin/env python3
"""
Trades Logger — records every snipe order and checks outcomes after round ends.
Writes to data/trades.csv for accurate P&L tracking.

Outcome check: polls CLOB API for fill status, then checks token price after resolution.
100% self-contained — does not depend on Polymarket data API.
"""
import csv
import json
import os
import time
from datetime import datetime, timezone
from src.logger import get_logger

logger = get_logger("trades_logger")

TRADES_CSV  = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "trades.csv")
PENDING_JSON = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "pending_trades.json")

CSV_HEADERS = ["date", "time_et", "asset", "direction", "entry_price", "shares",
               "cost", "outcome", "payout", "pnl", "order_id"]


def _et_now():
    from datetime import timedelta
    ET = timezone(timedelta(hours=-4))
    return datetime.now(ET)


class TradesLogger:
    """Records snipe orders and checks outcomes after round resolution."""

    def __init__(self, clob_client):
        self.clob = clob_client
        self._pending = {}  # order_id → trade dict
        self._ensure_files()
        self._load_pending()

    def _ensure_files(self):
        os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
        if not os.path.exists(TRADES_CSV):
            with open(TRADES_CSV, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
            logger.info("Created trades.csv")

    def _load_pending(self):
        try:
            if os.path.exists(PENDING_JSON):
                with open(PENDING_JSON) as f:
                    self._pending = json.load(f)
                if self._pending:
                    logger.info("Loaded %d pending trades to check", len(self._pending))
        except Exception as e:
            logger.debug("Could not load pending trades: %s", e)

    def _save_pending(self):
        try:
            os.makedirs(os.path.dirname(PENDING_JSON), exist_ok=True)
            with open(PENDING_JSON, "w") as f:
                json.dump(self._pending, f, indent=2)
        except Exception as e:
            logger.debug("Could not save pending trades: %s", e)

    def record_order(self, order_id: str, asset: str, direction: str,
                     entry_price: float, shares: float, window_end: float,
                     token_id: str, condition_id: str = ""):
        """Record a placed order. Outcome checked after window_end."""
        et = _et_now()
        trade = {
            "order_id": order_id,
            "asset": asset,
            "direction": direction,
            "entry_price": entry_price,
            "shares": shares,
            "cost": round(entry_price * shares, 4),
            "window_end": window_end,
            "token_id": token_id,
            "condition_id": condition_id,
            "date": et.strftime("%Y-%m-%d"),
            "time_et": et.strftime("%I:%M %p"),
            "recorded_at": time.time(),
        }
        self._pending[order_id] = trade
        self._save_pending()
        logger.debug("Recorded order %s: %s %s @ %.0f¢", order_id[:8], asset, direction, entry_price * 100)

    def check_pending(self):
        """Check outcomes for orders whose windows have ended. Call every 30s."""
        if not self._pending:
            return

        now = time.time()
        resolved = []

        for order_id, trade in list(self._pending.items()):
            window_end = trade.get("window_end", 0)
            # Wait 90s after window end for market to resolve + clob to update
            if now < window_end + 90:
                continue

            try:
                outcome, payout = self._check_outcome(order_id, trade)
                self._write_result(trade, outcome, payout)
                resolved.append(order_id)
            except Exception as e:
                logger.debug("Could not check outcome for %s: %s", order_id[:8], e)
                # If too old (>30 min), give up
                if now > window_end + 1800:
                    self._write_result(trade, "UNKNOWN", 0)
                    resolved.append(order_id)

        for order_id in resolved:
            self._pending.pop(order_id, None)

        if resolved:
            self._save_pending()

    def _check_outcome(self, order_id: str, trade: dict):
        """
        Check if order filled and whether it won or lost.
        Returns (outcome, payout) where outcome is WIN/LOSS/UNFILLED/UNKNOWN.

        Order: check fill status FIRST. Only mark WIN/LOSS if order actually filled.
        Unfilled orders = pnl $0 (no USDC moved).
        """
        import requests

        token_id = trade.get("token_id", "")
        shares = trade.get("shares", 0)

        # Step 1: Check if order actually filled
        filled_shares = 0.0
        try:
            order = self.clob.get_order(order_id)
            if order:
                filled_shares = float(order.get("size_matched", 0))
                status = order.get("status", "").upper()
                # Cancelled or expired with no fill = UNFILLED
                if status in ("CANCELLED", "CANCELED", "EXPIRED") and filled_shares < shares * 0.1:
                    return "UNFILLED", 0.0
                # Open order with no fill yet — too early, skip
                if status == "LIVE" and filled_shares < shares * 0.1:
                    return "UNKNOWN", 0.0
        except Exception:
            pass

        # If filled less than 10% of shares, treat as unfilled
        if filled_shares > 0 and filled_shares < shares * 0.1:
            return "UNFILLED", 0.0

        # Step 2: Check token price to determine WIN/LOSS
        try:
            r = requests.get(
                f"https://clob.polymarket.com/price?token_id={token_id}&side=BUY",
                timeout=5
            )
            if r.status_code == 200:
                price_data = r.json()
                price = float(price_data.get("price", 0))
                if price >= 0.95:
                    # Use actual filled shares for payout if available
                    actual_shares = filled_shares if filled_shares >= shares * 0.1 else shares
                    payout = round(actual_shares * 1.0, 4)
                    return "WIN", payout
                elif price <= 0.05:
                    return "LOSS", 0.0
        except Exception:
            pass

        return "UNKNOWN", 0.0

    def _write_result(self, trade: dict, outcome: str, payout: float):
        """Append resolved trade to CSV."""
        cost = trade.get("cost", 0)
        # UNFILLED = order never executed, no USDC moved, pnl = $0
        if outcome == "UNFILLED":
            pnl = 0.0
        else:
            pnl = round(payout - cost, 4)
        emoji = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else ("🔄" if outcome == "UNFILLED" else "?")

        row = {
            "date":         trade.get("date", ""),
            "time_et":      trade.get("time_et", ""),
            "asset":        trade.get("asset", ""),
            "direction":    trade.get("direction", ""),
            "entry_price":  trade.get("entry_price", 0),
            "shares":       trade.get("shares", 0),
            "cost":         cost,
            "outcome":      outcome,
            "payout":       payout,
            "pnl":          pnl,
            "order_id":     trade.get("order_id", "")[:12],
        }

        with open(TRADES_CSV, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)

        logger.info("%s Trade resolved: %s %s @ %.0f¢ | cost=$%.2f payout=$%.2f pnl=%+.2f",
                    emoji, trade.get("asset"), trade.get("direction"),
                    trade.get("entry_price", 0) * 100,
                    cost, payout, pnl)

    def print_summary(self):
        """Print today's P&L summary from CSV."""
        try:
            today = _et_now().strftime("%Y-%m-%d")
            wins = losses = 0
            total_pnl = 0.0
            with open(TRADES_CSV, newline="") as f:
                for row in csv.DictReader(f):
                    if row["date"] != today:
                        continue
                    if row["outcome"] == "WIN":
                        wins += 1
                    elif row["outcome"] == "LOSS":
                        losses += 1
                    try:
                        total_pnl += float(row["pnl"])
                    except Exception:
                        pass
            total = wins + losses
            rate = wins / total * 100 if total > 0 else 0
            logger.info("📊 Today: %dW/%dL (%.0f%%) | P&L: %+.2f",
                        wins, losses, rate, total_pnl)
        except Exception as e:
            logger.debug("Summary error: %s", e)
