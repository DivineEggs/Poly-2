#!/usr/bin/env python3
"""
Market Scanner — Discovers and tracks active Polymarket crypto up/down markets.

Queries the Gamma API to find all active crypto resolution markets and extracts
their token IDs for order book monitoring. Uses structured logging and rate limiting.
"""
import json
import time
from datetime import datetime, timezone

import requests

from src.logger import get_logger

logger = get_logger("scanner")

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

UPDOWN_SLUGS = ["updown-5m", "updown-15m", "updown-1h", "updown-4h", "updown-1d", "updown-1w"]
UPDOWN_KEYWORDS = ["up or down", "up/down"]
CRYPTO_ASSETS = ["btc", "bitcoin", "eth", "ethereum", "sol", "solana"]


def fetch_active_crypto_events(limit=100):
    """Fetch active crypto events from Gamma API."""
    url = f"{GAMMA_API}/events"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume24hr",
        "ascending": "false",
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def is_updown_market(event):
    """Check if an event is a crypto up/down resolution market."""
    title = (event.get("title") or "").lower()
    slug = (event.get("slug") or "").lower()
    has_updown = any(kw in title for kw in UPDOWN_KEYWORDS) or any(s in slug for s in UPDOWN_SLUGS)
    if not has_updown:
        return False
    has_crypto = any(asset in title for asset in CRYPTO_ASSETS) or any(asset in slug for asset in CRYPTO_ASSETS)
    return has_crypto


def extract_market_info(event):
    """Extract useful market info from an event."""
    markets = event.get("markets", [])
    result = {
        "event_id": event.get("id"),
        "title": event.get("title"),
        "slug": event.get("slug"),
        "start_date": event.get("startDate"),
        "end_date": event.get("endDate"),
        "volume_24h": event.get("volume24hr", 0),
        "liquidity": event.get("liquidity", 0),
        "markets": [],
    }

    for m in markets:
        clob_token_ids = json.loads(m.get("clobTokenIds", "[]"))
        outcomes = json.loads(m.get("outcomes", "[]"))
        outcome_prices = json.loads(m.get("outcomePrices", "[]"))

        market_info = {
            "market_id": m.get("id"),
            "question": m.get("question"),
            "condition_id": m.get("conditionId"),
            "token_ids": dict(zip(outcomes, clob_token_ids)) if len(outcomes) == len(clob_token_ids) else {},
            "prices": dict(zip(outcomes, outcome_prices)) if len(outcomes) == len(outcome_prices) else {},
            "volume": m.get("volumeNum", 0),
            "liquidity": m.get("liquidityNum", 0),
            "end_date": m.get("endDate"),
            "tick_size": m.get("orderPriceMinTickSize", 0.01),
            "min_order_size": m.get("orderMinSize", 5),
            "fees_enabled": m.get("feesEnabled", False),
            "accepting_orders": m.get("acceptingOrders", False),
            "spread": m.get("spread"),
            "best_bid": m.get("bestBid"),
            "best_ask": m.get("bestAsk"),
        }
        result["markets"].append(market_info)

    return result


def fetch_order_book(token_id):
    """Fetch order book for a token from the CLOB API."""
    url = f"{CLOB_API}/book"
    params = {"token_id": token_id}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def scan_markets():
    """Main scan: find all active crypto up/down markets."""
    logger.info("=" * 70)
    logger.info("  POLYMARKET CRYPTO UP/DOWN MARKET SCANNER")
    logger.info("  %s", datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))
    logger.info("=" * 70)

    logger.info("Fetching active events...")
    events = fetch_active_crypto_events(limit=200)
    logger.info("Found %d total active events", len(events))

    updown_events = [e for e in events if is_updown_market(e)]
    logger.info("Found %d crypto up/down events", len(updown_events))

    if not updown_events:
        logger.info("Trying slug-based search...")
        for slug_pattern in ["btc-updown", "eth-updown", "sol-updown"]:
            url = f"{GAMMA_API}/events"
            params = {"slug": slug_pattern, "active": "true", "closed": "false", "limit": 5}
            try:
                resp = requests.get(url, params=params, timeout=10)
                if resp.status_code == 200:
                    results = resp.json()
                    for e in results:
                        if e not in updown_events:
                            updown_events.append(e)
            except requests.RequestException as ex:
                logger.warning("Error searching %s: %s", slug_pattern, ex)

    all_markets = []
    for event in updown_events:
        info = extract_market_info(event)
        all_markets.append(info)

        logger.info("📊 %s", info['title'])
        logger.info("   24h Volume: $%.0f  |  Liquidity: $%.0f",
                     info['volume_24h'], info['liquidity'])

        for m in info["markets"]:
            prices_str = ", ".join(f"{k}: {v}" for k, v in m["prices"].items())
            status = "✅ ACTIVE" if m["accepting_orders"] else "❌ CLOSED"
            fees = "💰 FEES" if m["fees_enabled"] else "FREE"
            logger.info("   %s | %s | %s", status, fees, prices_str)
            logger.info("   Spread: %s | Bid: %s | Ask: %s", m['spread'], m['best_bid'], m['best_ask'])
            logger.info("   Tick: %s | Min size: %s", m['tick_size'], m['min_order_size'])

            if m["accepting_orders"] and m["token_ids"]:
                for outcome, token_id in m["token_ids"].items():
                    try:
                        book = fetch_order_book(token_id)
                        bids = book.get("bids", [])
                        asks = book.get("asks", [])
                        bid_depth = sum(float(b.get("size", 0)) for b in bids[:5])
                        ask_depth = sum(float(a.get("size", 0)) for a in asks[:5])
                        logger.info("   📖 %s Book: %d bids (depth: %.0f) | %d asks (depth: %.0f)",
                                    outcome, len(bids), bid_depth, len(asks), ask_depth)
                        if bids:
                            logger.info("      Best bid: %s × %s", bids[0]['price'], bids[0]['size'])
                        if asks:
                            logger.info("      Best ask: %s × %s", asks[0]['price'], asks[0]['size'])
                    except requests.RequestException as ex:
                        logger.warning("Could not fetch %s book: %s", outcome, ex)
                    except (KeyError, IndexError, ValueError) as ex:
                        logger.warning("Error parsing %s book: %s", outcome, ex)
                    time.sleep(0.2)

    return all_markets


if __name__ == "__main__":
    from src.config import load_config
    from src.logger import setup_logging
    config = load_config()
    setup_logging(config)
    markets = scan_markets()
    logger.info("Total markets found: %d", sum(len(m['markets']) for m in markets))
