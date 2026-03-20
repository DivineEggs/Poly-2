"""
Metrics Collector — Tracks detailed performance data for analysis.
Runs alongside the bot, reads from live_output.log and health.json.
Writes to data/metrics_history.jsonl (one JSON object per resolved pair).
"""
import json
import os
import re
import time
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
METRICS_FILE = os.path.join(DATA_DIR, "metrics_history.jsonl")
HOURLY_FILE = os.path.join(DATA_DIR, "hourly_stats.json")


def log_pair_result(pair_data: dict):
    """Append a resolved pair's metrics to the JSONL file."""
    now = datetime.now(timezone.utc)
    
    record = {
        "timestamp": now.isoformat(),
        "hour_utc": now.hour,
        "day_of_week": now.strftime("%A"),
        "pair_id": pair_data.get("pair_id"),
        "asset": pair_data.get("asset"),
        "timeframe": pair_data.get("timeframe"),
        "status": pair_data.get("status"),
        "up_price": pair_data.get("up_price"),
        "down_price": pair_data.get("down_price"),
        "entry_spread": pair_data.get("spread_profit"),
        "total_cost": pair_data.get("total_cost"),
        "pnl": pair_data.get("pnl"),
        "both_filled": pair_data.get("up_filled") and pair_data.get("down_filled"),
        "repriced": pair_data.get("repriced", False),
        "shield_triggered": pair_data.get("shield_triggered", False),
        "time_to_fill_seconds": pair_data.get("time_to_fill", 0),
        "window_duration": pair_data.get("timeframe", "5m"),
    }
    
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def update_hourly_stats():
    """Aggregate metrics by hour from the JSONL file."""
    if not os.path.exists(METRICS_FILE):
        return
    
    hourly = {}  # hour -> {trades, wins, losses, pnl, both_filled, singles, assets}
    
    with open(METRICS_FILE) as f:
        for line in f:
            try:
                r = json.loads(line.strip())
            except:
                continue
            
            hour = r.get("hour_utc", 0)
            if hour not in hourly:
                hourly[hour] = {
                    "hour_utc": hour,
                    "trades": 0, "wins": 0, "losses": 0, "breakeven": 0,
                    "pnl": 0.0, "both_filled": 0, "single_filled": 0,
                    "assets": {}, "avg_spread": 0.0, "spreads": [],
                    "double_fill_rate": 0.0,
                }
            
            h = hourly[hour]
            h["trades"] += 1
            pnl = r.get("pnl", 0)
            h["pnl"] += pnl
            
            if pnl > 0.001:
                h["wins"] += 1
            elif pnl < -0.001:
                h["losses"] += 1
            else:
                h["breakeven"] += 1
            
            if r.get("both_filled"):
                h["both_filled"] += 1
            else:
                h["single_filled"] += 1
            
            asset = r.get("asset", "unknown")
            if asset not in h["assets"]:
                h["assets"][asset] = {"trades": 0, "pnl": 0.0, "both_filled": 0}
            h["assets"][asset]["trades"] += 1
            h["assets"][asset]["pnl"] += pnl
            if r.get("both_filled"):
                h["assets"][asset]["both_filled"] += 1
            
            spread = r.get("entry_spread", 0)
            if spread:
                h["spreads"].append(spread)
    
    # Calculate averages
    for h in hourly.values():
        if h["spreads"]:
            h["avg_spread"] = round(sum(h["spreads"]) / len(h["spreads"]), 4)
        del h["spreads"]
        if h["trades"] > 0:
            h["double_fill_rate"] = round(h["both_filled"] / h["trades"] * 100, 1)
        h["pnl"] = round(h["pnl"], 4)
    
    # Sort by hour
    stats = {
        "updated": datetime.now(timezone.utc).isoformat(),
        "hourly": sorted(hourly.values(), key=lambda x: x["hour_utc"]),
        "summary": _build_summary(hourly),
    }
    
    with open(HOURLY_FILE, "w") as f:
        json.dump(stats, f, indent=2)


def _build_summary(hourly: dict) -> dict:
    """Build overall summary stats."""
    total_trades = sum(h["trades"] for h in hourly.values())
    total_pnl = sum(h["pnl"] for h in hourly.values())
    total_wins = sum(h["wins"] for h in hourly.values())
    total_losses = sum(h["losses"] for h in hourly.values())
    total_both = sum(h["both_filled"] for h in hourly.values())
    total_single = sum(h["single_filled"] for h in hourly.values())
    
    # Best/worst hours
    best_hour = max(hourly.values(), key=lambda x: x["pnl"]) if hourly else {}
    worst_hour = min(hourly.values(), key=lambda x: x["pnl"]) if hourly else {}
    
    # Best asset
    all_assets = {}
    for h in hourly.values():
        for asset, data in h["assets"].items():
            if asset not in all_assets:
                all_assets[asset] = {"trades": 0, "pnl": 0.0, "both_filled": 0}
            all_assets[asset]["trades"] += data["trades"]
            all_assets[asset]["pnl"] += data["pnl"]
            all_assets[asset]["both_filled"] += data["both_filled"]
    
    return {
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 4),
        "wins": total_wins,
        "losses": total_losses,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades else 0,
        "double_fill_rate": round(total_both / total_trades * 100, 1) if total_trades else 0,
        "both_filled": total_both,
        "single_filled": total_single,
        "best_hour_utc": best_hour.get("hour_utc"),
        "best_hour_pnl": round(best_hour.get("pnl", 0), 4),
        "worst_hour_utc": worst_hour.get("hour_utc"),
        "worst_hour_pnl": round(worst_hour.get("pnl", 0), 4),
        "by_asset": {k: {"trades": v["trades"], "pnl": round(v["pnl"], 4),
                         "fill_rate": round(v["both_filled"] / v["trades"] * 100, 1) if v["trades"] else 0}
                     for k, v in all_assets.items()},
    }


if __name__ == "__main__":
    update_hourly_stats()
    if os.path.exists(HOURLY_FILE):
        with open(HOURLY_FILE) as f:
            print(json.dumps(json.load(f), indent=2))
