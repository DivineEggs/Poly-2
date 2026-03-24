#!/usr/bin/env python3
"""
Window Tracker — Tracks prediction market windows and captures start prices.

Since Polymarket doesn't expose the Chainlink start price via API, we capture
the Binance price at the exact moment each window opens.

Uses structured logging instead of print statements.
"""
import asyncio
import time
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests

from src.logger import get_logger

logger = get_logger("window_tracker")


@dataclass
class Window:
    """A single prediction market window (e.g., BTC 5-min 10:00-10:05)."""
    asset: str
    timeframe: str
    tf_seconds: int
    start_ts: int
    end_ts: int
    start_price: float = 0.0
    slug: str = ""
    up_token_id: str = ""
    down_token_id: str = ""
    condition_id: str = ""
    market_up_price: float = 0.5
    market_down_price: float = 0.5
    best_bid_up: float = 0.0
    best_ask_up: float = 0.0
    best_bid_down: float = 0.0
    best_ask_down: float = 0.0
    accepting_orders: bool = False
    fees_enabled: bool = True
    liquidity: float = 0.0
    resolved: bool = False
    resolution: str = ""

    @property
    def time_remaining(self):
        return max(0, self.end_ts - time.time())

    @property
    def time_elapsed(self):
        return max(0, time.time() - self.start_ts)

    @property
    def progress(self):
        total = self.end_ts - self.start_ts
        if total <= 0:
            return 1.0
        return min(1.0, self.time_elapsed / total)

    @property
    def is_active(self):
        now = time.time()
        return self.start_ts <= now < self.end_ts and self.accepting_orders

    @property
    def is_upcoming(self):
        return time.time() < self.start_ts

    @property
    def is_expired(self):
        return time.time() >= self.end_ts

    @property
    def key(self):
        return f"{self.asset}-{self.timeframe}-{self.start_ts}"


class WindowTracker:
    """
    Tracks all active prediction windows and captures start prices.

    Call tick() frequently with current prices to capture window opens.
    """

    ALL_TIMEFRAMES = [
        ("5m", 300),
        ("15m", 900),
        ("4h", 14400),
    ]

    ALL_ASSETS = ["BTC", "ETH", "SOL"]

    def __init__(self, config=None, lookback_windows=1, lookahead_windows=4):
        self.config = config
        self.windows: dict[str, Window] = {}
        self.lookback = lookback_windows
        self.lookahead = lookahead_windows
        self._start_prices_captured: set[str] = set()
        self._kline_attempted: set[str] = set()  # avoid repeated kline fetches
        self._log_file = None

        # Filter to only configured timeframes and assets
        if config:
            allowed_tf = set(config.trading.allowed_timeframes)
            self.TIMEFRAMES = [(n, s) for n, s in self.ALL_TIMEFRAMES if n in allowed_tf]
            self.ASSETS = [a for a in self.ALL_ASSETS if a in config.trading.allowed_assets]
        else:
            self.TIMEFRAMES = self.ALL_TIMEFRAMES
            self.ASSETS = self.ALL_ASSETS

    def set_log_file(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._log_file = path

    def _log(self, record: dict):
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except OSError as e:
                logger.error("Failed to write event log: %s", e)

    def generate_windows(self):
        now = int(time.time())
        new_windows = 0

        for tf_name, tf_seconds in self.TIMEFRAMES:
            current_start = now - (now % tf_seconds)

            for offset in range(-self.lookback, self.lookahead + 1):
                window_start = current_start + (offset * tf_seconds)
                window_end = window_start + tf_seconds

                if window_end < now - 120:
                    continue

                for asset in self.ASSETS:
                    key = f"{asset}-{tf_name}-{window_start}"

                    if key not in self.windows:
                        slug_asset = asset.lower()
                        slug = f"{slug_asset}-updown-{tf_name}-{window_start}"

                        self.windows[key] = Window(
                            asset=asset,
                            timeframe=tf_name,
                            tf_seconds=tf_seconds,
                            start_ts=window_start,
                            end_ts=window_end,
                            slug=slug,
                        )
                        new_windows += 1

        expired_keys = [
            k for k, w in self.windows.items()
            if w.end_ts < now - 300
        ]
        for k in expired_keys:
            del self.windows[k]

        if new_windows > 0:
            logger.debug("Generated %d new windows, %d total, %d expired removed",
                         new_windows, len(self.windows), len(expired_keys))

        return new_windows

    def tick(self, prices: dict[str, float]):
        now = time.time()

        for key, window in self.windows.items():
            if key in self._start_prices_captured:
                continue

            asset_price = prices.get(window.asset, 0)
            if asset_price <= 0:
                continue

            time_since_start = now - window.start_ts

            if -2 <= time_since_start <= 5:
                window.start_price = asset_price
                self._start_prices_captured.add(key)

                logger.debug("Start price captured: %s %s $%.2f (delay: %dms)",
                             window.asset, window.timeframe, asset_price,
                             int(time_since_start * 1000))

                self._log({
                    "event": "start_price_captured",
                    "time": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "asset": window.asset,
                    "timeframe": window.timeframe,
                    "start_price": asset_price,
                    "window_start": window.start_ts,
                    "window_end": window.end_ts,
                    "delay_ms": int(time_since_start * 1000),
                })

            elif time_since_start > 5 and window.start_price <= 0:
                # Bot started mid-round — kline fetch handled in scan() to avoid blocking
                # tick() is called every 100ms, we must not do HTTP here
                pass

    BINANCE_SYMBOL_MAP = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

    def _fetch_candle_open(self, asset: str, start_ts: int) -> float:
        """
        Fetch the opening price of the 5-min Binance candle at start_ts.
        Uses Binance REST klines API — always accurate regardless of when bot started.
        """
        symbol = self.BINANCE_SYMBOL_MAP.get(asset)
        if not symbol:
            return 0.0
        try:
            import requests
            url = "https://api.binance.com/api/v3/klines"
            params = {
                "symbol": symbol,
                "interval": "5m",
                "startTime": start_ts * 1000,  # milliseconds
                "limit": 1,
            }
            r = requests.get(url, params=params, timeout=5)
            data = r.json()
            if data and isinstance(data, list) and len(data) > 0:
                candle_open_ts = data[0][0] // 1000  # ms → s
                open_price = float(data[0][1])        # index 1 = open price
                # Verify the candle matches our window start
                if abs(candle_open_ts - start_ts) <= 60:
                    return open_price
        except Exception as e:
            logger.debug("Kline fetch error for %s: %s", asset, e)
        return 0.0

    GAMMA_API = "https://gamma-api.polymarket.com"

    async def scan(self):
        """Generate windows and fetch market data (token IDs, prices) from Gamma API."""
        self.generate_windows()

        # Fetch market data for windows that don't have token IDs yet
        for key, window in self.windows.items():
            if window.up_token_id and window.down_token_id:
                continue  # Already have token IDs
            if window.is_expired:
                continue

            try:
                r = requests.get(
                    f"{self.GAMMA_API}/events",
                    params={"slug": window.slug},
                    timeout=5,
                )
                if r.status_code != 200:
                    continue

                data = r.json()
                if not data:
                    continue

                event = data[0]
                markets = event.get("markets", [])
                if not markets:
                    continue

                m = markets[0]
                token_ids = json.loads(m.get("clobTokenIds", "[]"))

                if len(token_ids) >= 2:
                    window.up_token_id = token_ids[0]
                    window.down_token_id = token_ids[1]
                    window.condition_id = m.get("conditionId", "")
                    window.accepting_orders = m.get("acceptingOrders", False)
                    window.fees_enabled = m.get("feesEnabled", True)
                    window.liquidity = float(m.get("liquidityNum") or 0)

                    outcome_prices = json.loads(m.get("outcomePrices", "[]"))
                    if len(outcome_prices) >= 2:
                        window.market_up_price = float(outcome_prices[0])
                        window.market_down_price = float(outcome_prices[1])

                    logger.debug("Fetched market data for %s: Up=%s Down=%s accepting=%s",
                                 key, window.up_token_id[:12], window.down_token_id[:12],
                                 window.accepting_orders)

            except requests.exceptions.Timeout:
                logger.debug("Timeout fetching %s", key)
            except requests.exceptions.ConnectionError:
                logger.debug("Connection error fetching %s", key)
            except (json.JSONDecodeError, KeyError, IndexError, ValueError) as e:
                logger.debug("Parse error for %s: %s", key, e)
            except Exception as e:
                logger.warning("Error fetching market data for %s: %s", key, e)

            await asyncio.sleep(0.15)  # Rate limit

        # Fetch kline start prices for windows that are mid-round (bot started late)
        # Done here (scan, every 30s) not in tick() to avoid blocking the async loop
        for key, window in self.windows.items():
            if key in self._start_prices_captured:
                continue
            if key in self._kline_attempted:
                continue
            if window.is_expired:
                continue
            now = time.time()
            time_since_start = now - window.start_ts
            if time_since_start <= 5:
                continue  # tick() will handle this one
            self._kline_attempted.add(key)
            try:
                open_price = self._fetch_candle_open(window.asset, window.start_ts)
                if open_price > 0:
                    window.start_price = open_price
                    self._start_prices_captured.add(key)
                    logger.info("Kline start price: %s %s $%.2f (%.0fs into round)",
                                window.asset, window.timeframe, open_price, time_since_start)
            except Exception as e:
                logger.debug("Kline fetch error %s: %s", key, e)

    def get_active_windows(self) -> list[Window]:
        return [w for w in self.windows.values() if w.is_active]

    def get_windows_by_asset(self, asset: str) -> list[Window]:
        return [w for w in self.windows.values() if w.asset == asset]

    def get_window(self, key: str) -> Window | None:
        return self.windows.get(key)

    def summary(self) -> str:
        lines = []

        for tf_name, _ in self.TIMEFRAMES:
            tf_windows = sorted(
                [w for w in self.windows.values() if w.timeframe == tf_name],
                key=lambda w: (w.asset, w.start_ts),
            )

            if not tf_windows:
                continue

            lines.append(f"\n  {tf_name} Windows:")
            for w in tf_windows:
                status = "🟢 LIVE" if w.is_active else ("⏳ SOON" if w.is_upcoming else "⬛ DONE")
                start_str = f"${w.start_price:,.2f}" if w.start_price > 0 else "---"
                time_str = f"{w.time_remaining:.0f}s left" if not w.is_expired else "expired"
                lines.append(
                    f"    {status} {w.asset} | start: {start_str} | {time_str} | "
                    f"mkt: ↑{w.market_up_price:.1%}/↓{w.market_down_price:.1%}"
                )

        return "\n".join(lines)


if __name__ == "__main__":
    from src.config import load_config
    from src.logger import setup_logging
    config = load_config()
    setup_logging(config)

    tracker = WindowTracker()
    tracker.generate_windows()

    logger.info("Generated %d windows", len(tracker.windows))
    logger.info(tracker.summary())

    tracker.tick({"BTC": 70880.0, "ETH": 2093.0, "SOL": 88.3})
    logger.info("After tick: %d start prices captured", len(tracker._start_prices_captured))
