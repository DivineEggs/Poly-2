#!/usr/bin/env python3
"""
Order Book Analyzer — Analyzes Polymarket order book depth.

Answers three questions:
1. Is there enough activity? (volume, trade count)
2. How deep is the book? (can our order realistically fill?)
3. What's the spread like? (are we competitive?)

Uses structured logging instead of print statements.
"""
import time
from dataclasses import dataclass

import requests

from src.logger import get_logger

logger = get_logger("orderbook")

CLOB_API = "https://clob.polymarket.com"


@dataclass
class BookLevel:
    """One price level in the order book."""
    price: float
    size: float

    @property
    def value(self):
        return self.price * self.size


@dataclass
class BookAnalysis:
    """Analysis of one side of an order book."""
    levels: list[BookLevel]

    @property
    def depth_shares(self):
        return sum(l.size for l in self.levels)

    @property
    def depth_dollars(self):
        return sum(l.value for l in self.levels)

    @property
    def best_price(self):
        return self.levels[0].price if self.levels else 0.0

    @property
    def n_levels(self):
        return len(self.levels)

    def depth_at_price(self, price: float, side: str = "bid") -> float:
        total = 0
        for l in self.levels:
            if side == "bid" and l.price >= price:
                total += l.size
            elif side == "ask" and l.price <= price:
                total += l.size
        return total


@dataclass
class MarketBookAnalysis:
    """Full analysis of a Polymarket market's order book."""
    token_id: str
    outcome: str
    bids: BookAnalysis
    asks: BookAnalysis
    timestamp: float

    @property
    def spread(self):
        if self.bids.best_price > 0 and self.asks.best_price > 0:
            return self.asks.best_price - self.bids.best_price
        return 1.0

    @property
    def spread_pct(self):
        mid = self.midpoint
        if mid > 0:
            return self.spread / mid * 100
        return 100.0

    @property
    def midpoint(self):
        if self.bids.best_price > 0 and self.asks.best_price > 0:
            return (self.bids.best_price + self.asks.best_price) / 2
        return 0.5

    @property
    def total_depth_dollars(self):
        return self.bids.depth_dollars + self.asks.depth_dollars

    def score(self) -> dict:
        scores = {}

        if self.spread_pct <= 2:
            scores["spread"] = 10
        elif self.spread_pct <= 5:
            scores["spread"] = 7
        elif self.spread_pct <= 10:
            scores["spread"] = 4
        else:
            scores["spread"] = 1

        total_depth = self.total_depth_dollars
        if total_depth >= 5000:
            scores["depth"] = 10
        elif total_depth >= 1000:
            scores["depth"] = 7
        elif total_depth >= 200:
            scores["depth"] = 4
        else:
            scores["depth"] = 1

        total_levels = self.bids.n_levels + self.asks.n_levels
        if total_levels >= 20:
            scores["levels"] = 10
        elif total_levels >= 10:
            scores["levels"] = 7
        elif total_levels >= 4:
            scores["levels"] = 4
        else:
            scores["levels"] = 1

        if self.bids.depth_dollars > 0 and self.asks.depth_dollars > 0:
            ratio = min(self.bids.depth_dollars, self.asks.depth_dollars) / \
                    max(self.bids.depth_dollars, self.asks.depth_dollars)
            scores["balance"] = int(ratio * 10)
        else:
            scores["balance"] = 0

        weights = {"spread": 3, "depth": 3, "levels": 2, "balance": 2}
        total_weight = sum(weights.values())
        overall = sum(scores[k] * weights[k] for k in scores) / total_weight

        if overall >= 7:
            recommendation = "EXCELLENT"
        elif overall >= 5:
            recommendation = "GOOD"
        elif overall >= 3:
            recommendation = "MARGINAL"
        else:
            recommendation = "SKIP"

        return {
            "scores": scores,
            "overall": round(overall, 1),
            "recommendation": recommendation,
            "spread_pct": round(self.spread_pct, 2),
            "total_depth": round(self.total_depth_dollars, 2),
            "bid_depth": round(self.bids.depth_dollars, 2),
            "ask_depth": round(self.asks.depth_dollars, 2),
            "n_levels": self.bids.n_levels + self.asks.n_levels,
        }

    def summary(self) -> str:
        s = self.score()
        return (
            f"{self.outcome}: spread {s['spread_pct']:.1f}% | "
            f"depth ${s['total_depth']:,.0f} ({self.bids.n_levels}b/{self.asks.n_levels}a) | "
            f"score {s['overall']}/10 → {s['recommendation']}"
        )


def fetch_order_book(token_id: str) -> MarketBookAnalysis | None:
    """Fetch and analyze an order book from the CLOB API."""
    try:
        r = requests.get(
            f"{CLOB_API}/book",
            params={"token_id": token_id},
            timeout=5,
        )
        r.raise_for_status()
        data = r.json()

        bids = BookAnalysis(
            levels=[
                BookLevel(price=float(b["price"]), size=float(b["size"]))
                for b in sorted(data.get("bids", []), key=lambda x: -float(x["price"]))
            ]
        )

        asks = BookAnalysis(
            levels=[
                BookLevel(price=float(a["price"]), size=float(a["size"]))
                for a in sorted(data.get("asks", []), key=lambda x: float(x["price"]))
            ]
        )

        return MarketBookAnalysis(
            token_id=token_id,
            outcome="",
            bids=bids,
            asks=asks,
            timestamp=time.time(),
        )

    except requests.exceptions.Timeout:
        logger.warning("Timeout fetching order book for %s", token_id[:16])
        return None
    except requests.exceptions.ConnectionError as e:
        logger.warning("Connection error fetching order book: %s", e)
        return None
    except (KeyError, ValueError) as e:
        logger.warning("Error parsing order book for %s: %s", token_id[:16], e)
        return None
    except Exception as e:
        logger.error("Unexpected error fetching order book for %s: %s", token_id[:16], e)
        return None


def analyze_market(up_token_id: str, down_token_id: str) -> dict:
    """Analyze both sides of a market (Up and Down order books)."""
    result = {"up": None, "down": None, "tradeable": False}

    up_book = fetch_order_book(up_token_id)
    if up_book:
        up_book.outcome = "Up"
        result["up"] = up_book

    time.sleep(0.2)

    down_book = fetch_order_book(down_token_id)
    if down_book:
        down_book.outcome = "Down"
        result["down"] = down_book

    if up_book and down_book:
        up_score = up_book.score()
        down_score = down_book.score()
        best_score = max(up_score["overall"], down_score["overall"])
        result["tradeable"] = best_score >= 3.0
        result["best_score"] = best_score
        result["best_side"] = "Up" if up_score["overall"] >= down_score["overall"] else "Down"

    return result
