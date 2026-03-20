#!/usr/bin/env python3
"""
State Persistence — Saves and restores bot state across restarts.

State includes:
- All open/pending orders
- Stats counters (wins, losses, P&L, etc.)
- Position size and order ID counter
- Kill switch state (consecutive losses)

Uses atomic writes (temp file + rename) to prevent corruption on crash.
"""
import json
import os
import tempfile
import time
from datetime import datetime, timezone

from src.logger import get_logger
from src.config import DATA_DIR

logger = get_logger("state_manager")

# Increment this when the state schema changes
STATE_VERSION = 1


class StateManager:
    """
    Manages persistent bot state in data/bot_state.json.

    Saves after every order event. Loads on startup to resume.
    Uses atomic writes to prevent corruption.

    Args:
        data_dir: Directory for state file. Defaults to project data/.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self._state_file = os.path.join(self.data_dir, "bot_state.json")
        os.makedirs(self.data_dir, exist_ok=True)

    def save(self, state: dict):
        """
        Atomically save state to disk.

        Writes to a temp file first, then renames. This ensures the
        state file is never partially written.

        Args:
            state: Dict of bot state to persist.
        """
        state["_version"] = STATE_VERSION
        state["_saved_at"] = datetime.now(timezone.utc).isoformat()
        state["_saved_ts"] = time.time()

        try:
            # Write to temp file in same directory (ensures same filesystem for rename)
            fd, tmp_path = tempfile.mkstemp(
                dir=self.data_dir, prefix="bot_state_", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, default=str)

            # Atomic rename
            os.replace(tmp_path, self._state_file)
            logger.debug("State saved to %s", self._state_file)

        except OSError as e:
            logger.error("Failed to save state: %s", e)
            # Clean up temp file if rename failed
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass

    def load(self) -> dict | None:
        """
        Load state from disk.

        Returns:
            State dict if file exists and is valid, None otherwise.
        """
        if not os.path.exists(self._state_file):
            logger.info("No state file found — starting fresh")
            return None

        try:
            with open(self._state_file, "r") as f:
                state = json.load(f)

            version = state.get("_version", 0)
            if version != STATE_VERSION:
                logger.warning(
                    "State version mismatch: file=%d, expected=%d — starting fresh",
                    version, STATE_VERSION,
                )
                return None

            saved_at = state.get("_saved_at", "unknown")
            logger.info("Loaded state from %s (saved at %s)", self._state_file, saved_at)
            return state

        except (json.JSONDecodeError, OSError) as e:
            logger.error("Failed to load state: %s — starting fresh", e)
            return None

    def exists(self) -> bool:
        """Check if a state file exists."""
        return os.path.exists(self._state_file)

    @staticmethod
    def serialize_orders(orders: list) -> list[dict]:
        """
        Serialize a list of PaperOrder objects to dicts for JSON storage.

        Args:
            orders: List of PaperOrder objects.

        Returns:
            List of serializable dicts.
        """
        result = []
        for o in orders:
            d = {
                "id": o.id,
                "window_key": o.window_key,
                "asset": o.asset,
                "timeframe": o.timeframe,
                "side": o.side,
                "price": o.price,
                "size": o.size,
                "fair_value": o.fair_value,
                "market_price": o.market_price,
                "edge": o.edge,
                "placed_at": o.placed_at,
                "window_end_ts": o.window_end_ts,
                "start_price": o.start_price,
                "crypto_price_at_place": o.crypto_price_at_place,
                "status": o.status.value,
                "filled_at": o.filled_at,
                "filled_price": o.filled_price,
                "resolution": o.resolution.value,
                "pnl": o.pnl,
                "cancel_reason": o.cancel_reason,
            }
            result.append(d)
        return result

    @staticmethod
    def deserialize_orders(data: list[dict]) -> list:
        """
        Deserialize order dicts back to PaperOrder objects.

        Imports PaperOrder here to avoid circular imports.

        Args:
            data: List of order dicts from state file.

        Returns:
            List of PaperOrder objects.
        """
        from src.paper_trader import PaperOrder, OrderStatus, Resolution

        orders = []
        for d in data:
            order = PaperOrder(
                id=d["id"],
                window_key=d["window_key"],
                asset=d["asset"],
                timeframe=d["timeframe"],
                side=d["side"],
                price=d["price"],
                size=d["size"],
                fair_value=d["fair_value"],
                market_price=d["market_price"],
                edge=d["edge"],
                placed_at=d["placed_at"],
                window_end_ts=d["window_end_ts"],
                start_price=d["start_price"],
                crypto_price_at_place=d["crypto_price_at_place"],
            )
            order.status = OrderStatus(d.get("status", "pending"))
            order.filled_at = d.get("filled_at", 0.0)
            order.filled_price = d.get("filled_price", 0.0)
            order.resolution = Resolution(d.get("resolution", "unknown"))
            order.pnl = d.get("pnl", 0.0)
            order.cancel_reason = d.get("cancel_reason", "")
            orders.append(order)
        return orders
