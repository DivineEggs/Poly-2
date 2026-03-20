#!/usr/bin/env python3
"""
Price Feed — Real-time crypto price streaming from Binance via WebSocket.

Connects to Binance's WebSocket API and streams BTC, ETH, SOL prices in real-time.
Detects price moves and calculates percentage changes from reference prices.
Uses exponential backoff for reconnection instead of fixed delays.
"""
import asyncio
import json
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

import websockets

from src.logger import get_logger

logger = get_logger("price_feed")

# Binance WebSocket streams
BINANCE_WS = "wss://stream.binance.com:9443/ws"
BINANCE_STREAM = "wss://stream.binance.com:9443/stream"

# Symbols to track (Binance format)
SYMBOLS = {
    "btcusdt": "BTC",
    "ethusdt": "ETH",
    "solusdt": "SOL",
}


@dataclass
class PriceState:
    """Tracks the current state of a price feed."""
    symbol: str
    price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    last_update: float = 0.0
    reference_price: float = 0.0
    price_history: list = field(default_factory=list)

    @property
    def pct_change(self):
        if self.reference_price <= 0:
            return 0.0
        return (self.price - self.reference_price) / self.reference_price * 100

    def update(self, price, bid=None, ask=None):
        self.price = price
        if bid is not None:
            self.bid = bid
        if ask is not None:
            self.ask = ask
        self.last_update = time.time()
        self.price_history.append((self.last_update, price))
        cutoff = self.last_update - 300
        self.price_history = [(t, p) for t, p in self.price_history if t > cutoff]

    def set_reference(self, price=None):
        self.reference_price = price if price is not None else self.price

    def realized_volatility(self, lookback_seconds=60):
        cutoff = time.time() - lookback_seconds
        recent = [(t, p) for t, p in self.price_history if t > cutoff]
        if len(recent) < 2:
            return 0.0
        returns = []
        for i in range(1, len(recent)):
            r = (recent[i][1] - recent[i-1][1]) / recent[i-1][1]
            returns.append(r)
        if not returns:
            return 0.0
        import statistics
        return statistics.stdev(returns) if len(returns) > 1 else abs(returns[0])


class PriceFeed:
    """Real-time price feed from Binance WebSocket with exponential backoff."""

    def __init__(self, symbols=None, on_price_update=None):
        self.symbols = symbols or SYMBOLS
        self.states: dict[str, PriceState] = {}
        self.on_price_update = on_price_update
        self._running = False
        self._ws = None

        self.last_latency_ms = 0
        self._last_msg_time = 0
        self.is_connected = False

        for binance_sym, our_sym in self.symbols.items():
            self.states[our_sym] = PriceState(symbol=our_sym)

    def get_price(self, symbol: str) -> float:
        state = self.states.get(symbol)
        return state.price if state else 0.0

    def get_state(self, symbol: str) -> PriceState | None:
        return self.states.get(symbol)

    def get_last_update_time(self) -> float:
        """Return the timestamp of the most recent price update across all symbols."""
        times = [s.last_update for s in self.states.values() if s.last_update > 0]
        return max(times) if times else 0.0

    async def connect(self):
        """Connect to Binance with exponential backoff on reconnection."""
        streams = [f"{sym}@bookTicker" for sym in self.symbols.keys()]
        stream_path = "/".join(streams)
        url = f"{BINANCE_STREAM}?streams={stream_path}"

        logger.info("Connecting to Binance WebSocket...")
        logger.info("  Tracking: %s", ", ".join(self.symbols.values()))
        logger.debug("  URL: %s", url[:80])

        self._running = True
        backoff_delay = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10, close_timeout=10) as ws:
                    self._ws = ws
                    logger.info("Connected to Binance WebSocket")
                    self.is_connected = True
                    backoff_delay = 1.0  # Reset backoff on successful connect

                    async for message in ws:
                        if not self._running:
                            break
                        try:
                            data = json.loads(message)
                            self._handle_message(data)
                        except json.JSONDecodeError:
                            logger.debug("Invalid JSON from Binance WebSocket")

            except websockets.exceptions.ConnectionClosed as e:
                self.is_connected = False
                if self._running:
                    logger.warning("Binance connection closed (code=%s), reconnecting in %.1fs...",
                                   getattr(e, 'code', 'unknown'), backoff_delay)
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)
            except Exception as e:
                if self._running:
                    logger.error("Binance WebSocket error: %s, reconnecting in %.1fs...", e, backoff_delay)
                    await asyncio.sleep(backoff_delay)
                    backoff_delay = min(backoff_delay * 2, max_backoff)

    def _handle_message(self, data):
        stream_data = data.get("data", data)
        symbol_raw = stream_data.get("s", "").lower()
        our_symbol = self.symbols.get(symbol_raw)

        if not our_symbol:
            return

        # Track inter-message latency (how fresh our data is)
        now = time.time()
        if self._last_msg_time > 0:
            self.last_latency_ms = (now - self._last_msg_time) * 1000
        self._last_msg_time = now

        bid = float(stream_data.get("b", 0))
        ask = float(stream_data.get("a", 0))
        mid = (bid + ask) / 2 if bid and ask else bid or ask

        state = self.states[our_symbol]
        state.update(mid, bid=bid, ask=ask)

        if self.on_price_update:
            self.on_price_update(our_symbol, state)

    def stop(self):
        self._running = False


async def demo():
    """Demo: stream prices and print updates."""
    from src.config import load_config
    from src.logger import setup_logging

    config = load_config()
    setup_logging(config)

    update_count = [0]
    last_print = [0.0]

    def on_update(symbol, state: PriceState):
        update_count[0] += 1
        now = time.time()
        if now - last_print[0] >= 2.0:
            last_print[0] = now
            for sym in ["BTC", "ETH", "SOL"]:
                s = feed.states.get(sym)
                if s and s.price > 0:
                    logger.info("  %s: $%.2f  bid/ask: $%.2f/$%.2f",
                                sym, s.price, s.bid, s.ask)

    feed = PriceFeed(on_price_update=on_update)

    async def set_refs():
        await asyncio.sleep(3)
        for state in feed.states.values():
            if state.price > 0:
                state.set_reference()
                logger.info("📌 %s reference set: $%.2f", state.symbol, state.price)

    asyncio.create_task(set_refs())

    logger.info("Starting Binance price feed demo (Ctrl+C to stop)...")
    try:
        await feed.connect()
    except KeyboardInterrupt:
        feed.stop()
        logger.info("Stopped.")


if __name__ == "__main__":
    asyncio.run(demo())
