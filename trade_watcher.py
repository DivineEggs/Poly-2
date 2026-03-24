#!/usr/bin/env python3
"""
Trade Watcher — polls Polymarket activity for wbz-egg every 60s.
When a new trade is detected, sends a Telegram notification via OpenClaw.

Run on Mac:
    nohup python3 ~/polymarket-spread-bot/trade_watcher.py > /tmp/trade_watcher.log 2>&1 &
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

# Force unbuffered output so logs appear immediately
sys.stdout.reconfigure(line_buffering=True)

WALLET       = "0xC97Afd8dD13F6b0B99a33eAcE5C8983E5e84A1f1"
TELEGRAM_ID  = "8203650754"
API_URL      = f"https://data-api.polymarket.com/activity?user={WALLET}&limit=5"
POLL_SECONDS = 60
STATE_FILE   = os.path.expanduser("~/polymarket-spread-bot/data/watcher_state.json")

def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"last_ts": 0}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def fetch_trades():
    import urllib.request
    try:
        req = urllib.request.Request(
            API_URL,
            headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"[{ts()}] Fetch error: {e}")
        return []

def send_telegram(message: str):
    result = subprocess.run(
        ["openclaw", "message", "send",
         "--channel", "telegram",
         "--target", TELEGRAM_ID,
         "--message", message],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[{ts()}] Send failed: {result.stderr}")
    else:
        print(f"[{ts()}] Sent: {message}")

def ts():
    return datetime.now().strftime("%H:%M:%S")

def format_trade(trade: dict) -> str:
    side      = trade.get("side", "?")
    size      = float(trade.get("size", 0))
    price     = float(trade.get("price", 0))
    cost      = float(trade.get("usdcSize", 0))
    outcome   = trade.get("outcome", "?")
    slug      = trade.get("slug", "")
    asset     = "BTC" if "btc" in slug else "ETH" if "eth" in slug else "SOL" if "sol" in slug else "?"
    dt        = datetime.fromtimestamp(trade["timestamp"], tz=timezone.utc)
    time_str  = dt.strftime("%H:%M UTC")

    emoji = "🟢" if side == "BUY" else "🔴"
    return (
        f"{emoji} Trade on wbz-egg\n"
        f"{side} {outcome} {asset} @ {price:.0%}\n"
        f"{size:.1f} shares · ${cost:.2f} USDC\n"
        f"{time_str}"
    )

def main():
    print(f"[{ts()}] Trade watcher started for {WALLET[:10]}…")
    state = load_state()
    print(f"[{ts()}] Last known trade ts: {state['last_ts']}")

    while True:
        trades = fetch_trades()

        new_trades = [t for t in trades if t.get("timestamp", 0) > state["last_ts"]]
        new_trades.sort(key=lambda t: t["timestamp"])

        for trade in new_trades:
            msg = format_trade(trade)
            send_telegram(msg)
            state["last_ts"] = max(state["last_ts"], trade["timestamp"])
            save_state(state)

        if not new_trades:
            print(f"[{ts()}] No new trades")

        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    main()
