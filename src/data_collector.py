#!/usr/bin/env python3
"""
Data Collector — Passive market observer. No orders placed.

Watches BTC/ETH/SOL 5-min Polymarket markets every 30 seconds and records:
- Spread width (up_ask + down_ask vs $1.00)
- Book depth on each side
- Binance price at time of observation
- Time of day / hour

Run this instead of the main bot when unsupervised.
Outputs to: data/spread_data.jsonl (one JSON record per line)
Use the data later to find best trading hours, assets, and thresholds.
"""

import json
import math
import os
import sys
import time
import threading
import asyncio
import requests
from datetime import datetime, timezone

# ── Config ──────────────────────────────────────────────────────────────────
SYMBOLS      = ["btcusdt", "ethusdt", "solusdt"]
SLUG_MAP     = {"btcusdt": "btc-updown-5m", "ethusdt": "eth-updown-5m", "solusdt": "sol-updown-5m"}
GAMMA_API    = "https://gamma-api.polymarket.com"
CLOB_API     = "https://clob.polymarket.com"
BINANCE_WS   = "wss://stream.binance.com:9443/ws"
SCAN_INTERVAL = 30   # seconds between scans
OUTPUT_FILE  = "data/spread_data.jsonl"

# ── Price feed (WebSocket) ───────────────────────────────────────────────────
_prices = {}
_prices_lock = threading.Lock()

def _ws_thread():
    async def _run():
        import websockets
        streams = "/".join(f"{s}@bookTicker" for s in SYMBOLS)
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"
        while True:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    print("✅ Binance connected")
                    async for raw in ws:
                        try:
                            d = json.loads(raw)
                            data = d.get("data", d)
                            sym = data.get("s", "").lower()
                            bid = float(data.get("b", 0))
                            ask = float(data.get("a", 0))
                            mid = (bid + ask) / 2 if bid and ask else bid or ask
                            if sym and mid > 0:
                                with _prices_lock:
                                    _prices[sym] = mid
                        except Exception:
                            pass
            except Exception as e:
                print(f"WS disconnected: {e}, reconnecting...")
                await asyncio.sleep(3)
    asyncio.run(_run())

def get_price(symbol):
    with _prices_lock:
        return _prices.get(symbol, 0)

# ── Market fetching ──────────────────────────────────────────────────────────
def get_current_round(symbol):
    prefix = SLUG_MAP.get(symbol)
    now = time.time()
    ts = int(now - (now % 300))
    for offset in [0, -300]:
        slug = f"{prefix}-{ts + offset}"
        try:
            r = requests.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=5)
            data = r.json()
            if not data:
                continue
            event = data[0]
            markets = event.get("markets", [])
            if not markets:
                continue
            m = markets[0]
            if not m.get("acceptingOrders", False):
                continue
            token_ids = json.loads(m.get("clobTokenIds", "[]"))
            if len(token_ids) < 2:
                continue
            end_ts = ts + offset + 300
            try:
                from datetime import datetime, timezone
                end_ts = datetime.fromisoformat(
                    m["endDate"].replace("Z", "+00:00")
                ).timestamp()
            except Exception:
                pass
            return {
                "slug": slug,
                "up_token": token_ids[0],
                "down_token": token_ids[1],
                "end_ts": end_ts,
                "time_remaining": end_ts - time.time(),
                "liquidity": float(m.get("liquidityNum") or 0),
            }
        except Exception:
            continue
    return None

def get_book_prices(token_id):
    """Returns best bid, best ask, and ask-side depth."""
    try:
        r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=5)
        data = r.json()
        bids = sorted(data.get("bids", []), key=lambda x: -float(x["price"]))
        asks = sorted(data.get("asks", []), key=lambda x: float(x["price"]))
        best_bid = float(bids[0]["price"]) if bids else 0
        best_ask = float(asks[0]["price"]) if asks else 0
        # Depth: total shares available up to 3¢ above best ask
        depth = sum(float(a["size"]) for a in asks
                    if float(a["price"]) <= best_ask + 0.03)
        return best_bid, best_ask, round(depth, 2)
    except Exception:
        return 0, 0, 0

# ── Record saving ─────────────────────────────────────────────────────────────
def save_record(record):
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

# ── Main scan ─────────────────────────────────────────────────────────────────
def scan():
    now = time.time()
    dt = datetime.fromtimestamp(now, tz=timezone.utc)
    hour = dt.hour
    records = []

    for symbol in SYMBOLS:
        asset = symbol.replace("usdt", "").upper()
        price = get_price(symbol)

        rnd = get_current_round(symbol)
        if not rnd:
            continue

        time_remaining = rnd["time_remaining"]
        if time_remaining < 30 or time_remaining > 290:
            continue  # Skip rounds that just started or nearly ended

        # Get order books
        up_bid, up_ask, up_depth = get_book_prices(rnd["up_token"])
        time.sleep(0.2)
        down_bid, down_ask, down_depth = get_book_prices(rnd["down_token"])

        if up_ask <= 0 or down_ask <= 0:
            continue

        spread = round(1.0 - up_ask - down_ask, 4)
        total_cost = round(up_ask + down_ask, 4)

        record = {
            "ts": now,
            "datetime": dt.isoformat(),
            "hour_utc": hour,
            "asset": asset,
            "slug": rnd["slug"],
            "price": price,
            "time_remaining": round(time_remaining),
            "up_ask": up_ask,
            "down_ask": down_ask,
            "up_bid": up_bid,
            "down_bid": down_bid,
            "up_depth": up_depth,
            "down_depth": down_depth,
            "spread": spread,
            "total_cost": total_cost,
            "liquidity": rnd["liquidity"],
            # Profitable if spread > 0 and both have depth
            "tradeable": spread >= 0.03 and up_depth >= 5 and down_depth >= 5,
        }

        save_record(record)
        records.append(record)

        spread_str = f"+{spread*100:.1f}¢" if spread > 0 else f"{spread*100:.1f}¢"
        tradeable = "✅" if record["tradeable"] else "❌"
        print(f"  {tradeable} {asset} | spread={spread_str} | "
              f"Up ask={up_ask:.2f} Down ask={down_ask:.2f} | "
              f"depth={up_depth:.0f}/{down_depth:.0f} | "
              f"{time_remaining:.0f}s left")

    return records

# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  DATA COLLECTOR — passive observer, no orders placed")
    print(f"  Output: {OUTPUT_FILE}")
    print(f"  Scan interval: {SCAN_INTERVAL}s")
    print(f"  Assets: BTC, ETH, SOL")
    print("=" * 55)

    # Start Binance WebSocket
    t = threading.Thread(target=_ws_thread, daemon=True)
    t.start()

    # Wait for first prices
    print("Waiting for Binance prices...")
    for _ in range(50):
        with _prices_lock:
            if len(_prices) >= 2:
                break
        time.sleep(0.2)

    with _prices_lock:
        for sym, price in _prices.items():
            asset = sym.replace("usdt", "").upper()
            print(f"  {asset}: ${price:,.2f}")

    print(f"\nScanning every {SCAN_INTERVAL}s. Ctrl+C to stop.\n")

    scan_count = 0
    try:
        while True:
            scan_count += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Scan #{scan_count}")
            records = scan()
            if not records:
                print("  No active rounds found")
            time.sleep(SCAN_INTERVAL)
    except KeyboardInterrupt:
        print(f"\nStopped. {scan_count} scans saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
