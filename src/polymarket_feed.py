#!/usr/bin/env python3
"""
Polymarket Feed — Real-time order book streaming via WebSocket.

Connects to the Polymarket CLOB WebSocket market channel and streams
live best bid/ask updates for subscribed token IDs. Replaces REST polling
for order book prices with sub-second updates.

Uses exponential backoff for reconnection (same pattern as price_feed.py).
"""
import asyncio
import json
import time
from dataclasses import dataclass, field

import websockets

from src.logger import get_logger

logger = get_logger("polymarket_feed")

POLYMARKET_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


@dataclass
class TokenBookState:
    """Tracks order book state for a single token."""
    token_id: str
    best_bid: float = 0.0
    best_ask: float = 0.0
    midpoint: float = 0.0
    last_update_time: float = 0.0

    # Full book for price_change updates
    bids: dict[float, float] = field(default_factory=dict)  # price -> size
    asks: dict[float, float] = field(default_factory=dict)  # price -> size

    def update_from_book(self, bids: list, asks: list):
        """Update from a full book snapshot."""
        self.bids.clear()
        self.asks.clear()

        for entry in bids:
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if size > 0:
                self.bids[price] = size

        for entry in asks:
            price = float(entry.get("price", 0))
            size = float(entry.get("size", 0))
            if size > 0:
                self.asks[price] = size

        self._recalc_best()
        self.last_update_time = time.time()

    def update_price_level(self, side: str, price: float, size: float):
        """Update a single price level from a price_change event."""
        book = self.bids if side == "BUY" else self.asks

        if size <= 0:
            book.pop(price, None)
        else:
            book[price] = size

        self._recalc_best()
        self.last_update_time = time.time()

    def update_best_bid_ask(self, bid: float = None, ask: float = None):
        """Direct update from best_bid_ask event (fastest path)."""
        if bid is not None and bid > 0:
            self.best_bid = bid
        if ask is not None and ask > 0:
            self.best_ask = ask
        if self.best_bid > 0 and self.best_ask > 0:
            self.midpoint = (self.best_bid + self.best_ask) / 2
        elif self.best_bid > 0:
            self.midpoint = self.best_bid
        elif self.best_ask > 0:
            self.midpoint = self.best_ask
        self.last_update_time = time.time()

    def _recalc_best(self):
        """Recalculate best bid/ask from the full book."""
        self.best_bid = max(self.bids.keys()) if self.bids else 0.0
        self.best_ask = min(self.asks.keys()) if self.asks else 0.0
        if self.best_bid > 0 and self.best_ask > 0:
            self.midpoint = (self.best_bid + self.best_ask) / 2
        elif self.best_bid > 0:
            self.midpoint = self.best_bid
        elif self.best_ask > 0:
            self.midpoint = self.best_ask


class PolymarketFeed:
    """
    Real-time order book feed from Polymarket WebSocket.

    Subscribes to the market channel for given token IDs and streams
    live best bid/ask updates. Thread-safe for reads from the trading loop.
    """

    def __init__(self, on_book_update=None):
        self._states: dict[str, TokenBookState] = {}
        self._subscribed_ids: set[str] = set()
        self._pending_subscribes: list[list[str]] = []
        self._pending_unsubscribes: list[list[str]] = []
        self._running = False
        self._ws = None
        self._connected = asyncio.Event()
        self.on_book_update = on_book_update

    def get_best_bid(self, token_id: str) -> float:
        """Get current best bid for a token."""
        state = self._states.get(token_id)
        return state.best_bid if state else 0.0

    def get_best_ask(self, token_id: str) -> float:
        """Get current best ask for a token."""
        state = self._states.get(token_id)
        return state.best_ask if state else 0.0

    def get_midpoint(self, token_id: str) -> float:
        """Get midpoint price for a token."""
        state = self._states.get(token_id)
        return state.midpoint if state else 0.0

    def get_last_update_time(self, token_id: str) -> float:
        """Get timestamp of last update for a token."""
        state = self._states.get(token_id)
        return state.last_update_time if state else 0.0

    def get_feed_age(self, token_id: str) -> float | None:
        """Get age of data in seconds for a token. Returns None if no data."""
        state = self._states.get(token_id)
        if not state or state.last_update_time == 0:
            return None
        return time.time() - state.last_update_time

    def is_stale(self, token_id: str, max_age: float = 30) -> bool:
        """Check if data for a token is stale (no recent updates)."""
        state = self._states.get(token_id)
        if not state or state.last_update_time == 0:
            return True
        return (time.time() - state.last_update_time) > max_age

    def has_data(self, token_id: str) -> bool:
        """Check if we have any data for a token."""
        state = self._states.get(token_id)
        return state is not None and state.last_update_time > 0

    def subscribe(self, token_ids: list[str]):
        """Subscribe to order book updates for given token IDs."""
        new_ids = [tid for tid in token_ids if tid and tid not in self._subscribed_ids]
        if not new_ids:
            return

        # Initialize states for new tokens
        for tid in new_ids:
            if tid not in self._states:
                self._states[tid] = TokenBookState(token_id=tid)
            self._subscribed_ids.add(tid)

        self._pending_subscribes.append(new_ids)
        logger.info("Queued subscribe for %d token(s): %s",
                     len(new_ids), [tid[:12] + "..." for tid in new_ids])

    def unsubscribe(self, token_ids: list[str]):
        """Unsubscribe from order book updates for given token IDs."""
        active_ids = [tid for tid in token_ids if tid in self._subscribed_ids]
        if not active_ids:
            return

        for tid in active_ids:
            self._subscribed_ids.discard(tid)

        self._pending_unsubscribes.append(active_ids)
        logger.debug("Queued unsubscribe for %d token(s)", len(active_ids))

    async def connect(self):
        """Connect to Polymarket WebSocket with exponential backoff."""
        logger.info("Starting Polymarket WebSocket feed...")
        self._running = True
        backoff_delay = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                async with websockets.connect(
                    POLYMARKET_WS,
                    ping_interval=20,
                    ping_timeout=30,
                    close_timeout=5,
                ) as ws:
                    self._ws = ws
                    self._connected.set()
                    connect_time = time.time()
                    logger.info("Connected to Polymarket WebSocket")

                    # Re-subscribe to all active token IDs on reconnect
                    if self._subscribed_ids:
                        await self._send_subscribe(list(self._subscribed_ids), initial=True)

                    # Message loop with periodic pending check
                    async def _process_pending_loop():
                        """Process pending subscribes even when no messages arrive."""
                        while self._running and self._ws:
                            await self._process_pending()
                            await asyncio.sleep(0.5)

                    async def _heartbeat_loop():
                        """Send PING every 10 seconds per Polymarket docs."""
                        while self._running and self._ws:
                            try:
                                await self._ws.send("PING")
                            except Exception:
                                break
                            await asyncio.sleep(10)

                    pending_task = asyncio.create_task(_process_pending_loop())
                    heartbeat_task = asyncio.create_task(_heartbeat_loop())
                    try:
                        async for message in ws:
                            if not self._running:
                                break

                            # Handle text control messages
                            if message in ("PONG", "INVALID OPERATION"):
                                if message == "INVALID OPERATION":
                                    logger.warning("Polymarket WebSocket: INVALID OPERATION received")
                                continue

                            try:
                                data = json.loads(message)
                                if isinstance(data, list):
                                    for item in data:
                                        if isinstance(item, dict):
                                            self._handle_message(item)
                                elif isinstance(data, dict):
                                    self._handle_message(data)
                            except json.JSONDecodeError:
                                logger.debug("Invalid JSON from Polymarket WebSocket: %s", message[:200] if len(message) < 200 else message[:200] + "...")
                    finally:
                        pending_task.cancel()
                        heartbeat_task.cancel()
                        try:
                            await pending_task
                        except asyncio.CancelledError:
                            pass
                        try:
                            await heartbeat_task
                        except asyncio.CancelledError:
                            pass

            except websockets.exceptions.ConnectionClosed as e:
                self._connected.clear()
                self._ws = None
                if self._running:
                    # Only reset backoff if connection was stable (>10s)
                    if time.time() - connect_time > 10:
                        backoff_delay = 1.0
                    logger.warning("Polymarket connection closed (code=%s), reconnecting in %.1fs...",
                                   getattr(e, 'code', 'unknown'), backoff_delay)
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)
            except Exception as e:
                self._connected.clear()
                self._ws = None
                if self._running:
                    if time.time() - connect_time > 10:
                        backoff_delay = 1.0
                    logger.error("Polymarket WebSocket error: %s, reconnecting in %.1fs...",
                                 e, backoff_delay)
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)

    async def _process_pending(self):
        """Send any pending subscribe/unsubscribe messages."""
        while self._pending_subscribes:
            ids = self._pending_subscribes.pop(0)
            await self._send_subscribe(ids)

        while self._pending_unsubscribes:
            ids = self._pending_unsubscribes.pop(0)
            await self._send_unsubscribe(ids)

    async def _send_subscribe(self, token_ids: list[str], initial: bool = False):
        """Send a subscribe message to the WebSocket."""
        if not self._ws:
            return
        if initial:
            # Initial subscribe uses type field
            msg = {
                "assets_ids": token_ids,
                "type": "market",
                "custom_feature_enabled": True,
            }
        else:
            # Dynamic subscribe uses operation field
            msg = {
                "assets_ids": token_ids,
                "operation": "subscribe",
                "custom_feature_enabled": True,
            }
        try:
            await self._ws.send(json.dumps(msg))
            logger.info("Subscribed to %d token(s)", len(token_ids))
        except Exception as e:
            logger.error("Failed to send subscribe: %s", e)

    async def _send_unsubscribe(self, token_ids: list[str]):
        """Send an unsubscribe message to the WebSocket."""
        if not self._ws:
            return
        msg = {
            "assets_ids": token_ids,
            "operation": "unsubscribe",
        }
        try:
            await self._ws.send(json.dumps(msg))
            logger.debug("Unsubscribed from %d token(s)", len(token_ids))
        except Exception as e:
            logger.error("Failed to send unsubscribe: %s", e)

    def _handle_message(self, data: dict):
        """Route incoming WebSocket messages to appropriate handlers."""
        # The market channel can send different event types
        event_type = data.get("event_type", "")

        if event_type == "book":
            self._handle_book(data)
        elif event_type == "price_change":
            self._handle_price_change(data)
        elif event_type == "best_bid_ask":
            self._handle_best_bid_ask(data)
        elif event_type == "last_trade_price":
            pass  # We don't need last trade price
        elif event_type == "tick_size_change":
            pass  # Ignore tick size changes
        else:
            # Some messages are arrays or have different structure
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        self._handle_message(item)
            elif "market" in data or "asset_id" in data:
                # Try to extract useful data from unknown format
                asset_id = data.get("asset_id", "")
                if asset_id and asset_id in self._states:
                    logger.debug("Unknown event for token %s...: %s",
                                 asset_id[:12], list(data.keys()))

    def _handle_book(self, data: dict):
        """Handle full order book snapshot."""
        asset_id = data.get("asset_id", "")
        if not asset_id or asset_id not in self._states:
            return

        state = self._states[asset_id]
        bids = data.get("bids", [])
        asks = data.get("asks", [])

        state.update_from_book(bids, asks)

        logger.debug("Book snapshot for %s...: bid=%.4f ask=%.4f mid=%.4f (%d bids, %d asks)",
                     asset_id[:12], state.best_bid, state.best_ask, state.midpoint,
                     len(bids), len(asks))

        if self.on_book_update:
            self.on_book_update(asset_id, state)

    def _handle_price_change(self, data: dict):
        """Handle individual price level update."""
        asset_id = data.get("asset_id", "")
        if not asset_id or asset_id not in self._states:
            return

        state = self._states[asset_id]

        # price_change events contain changes to specific levels
        changes = data.get("changes", [])
        for change in changes:
            # Each change: {"side": "BUY"/"SELL", "price": "0.50", "size": "100"}
            side = change.get("side", "")
            price = float(change.get("price", 0))
            size = float(change.get("size", 0))
            if price > 0:
                state.update_price_level(side, price, size)

        if self.on_book_update:
            self.on_book_update(asset_id, state)

    def _handle_best_bid_ask(self, data: dict):
        """Handle best bid/ask update (fastest path, requires custom_feature_enabled)."""
        asset_id = data.get("asset_id", "")
        if not asset_id or asset_id not in self._states:
            return

        state = self._states[asset_id]

        bid = float(data.get("best_bid", 0)) if data.get("best_bid") else None
        ask = float(data.get("best_ask", 0)) if data.get("best_ask") else None

        state.update_best_bid_ask(bid=bid, ask=ask)

        logger.debug("Best bid/ask for %s...: bid=%.4f ask=%.4f",
                     asset_id[:12], state.best_bid, state.best_ask)

        if self.on_book_update:
            self.on_book_update(asset_id, state)

    def stop(self):
        """Stop the feed."""
        self._running = False
        self._connected.clear()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()
