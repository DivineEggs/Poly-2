#!/usr/bin/env python3
"""
Health Monitoring — Writes health.json for external monitoring.

Updated every 30 seconds with:
- Bot status (running/halted/error)
- Uptime
- Feed freshness (detect stale Binance data)
- Trading stats (P&L, win rate, position)
- Config hash (detect changes)
- Last error
"""
import json
import os
import time
import tempfile
from datetime import datetime, timezone

from src.logger import get_logger
from src.config import Config, DATA_DIR

logger = get_logger("health")


class HealthMonitor:
    """
    Writes health status to data/health.json for external monitoring.

    Args:
        config: Bot configuration.
        data_dir: Directory for health file. Defaults to project data/.
    """

    def __init__(self, config: Config, data_dir: str = None):
        self.config = config
        self.data_dir = data_dir or DATA_DIR
        self._health_file = os.path.join(self.data_dir, "health.json")
        self._start_time = time.time()
        self._last_write = 0.0
        self._last_error = ""
        self._status = "starting"

        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def write_interval(self) -> float:
        return self.config.timing.health_write_interval

    def set_status(self, status: str):
        """Set bot status: 'running', 'halted', 'error', 'shutting_down'."""
        self._status = status

    def set_error(self, error: str):
        """Record the last error message."""
        self._last_error = error
        self._status = "error"

    def clear_error(self):
        """Clear the last error."""
        self._last_error = ""
        if self._status == "error":
            self._status = "running"

    def should_write(self) -> bool:
        """Check if it's time to write health.json."""
        return time.time() - self._last_write >= self.write_interval

    def write(
        self,
        stats: dict = None,
        last_binance_update: float = 0.0,
        last_polymarket_fetch: float = 0.0,
        total_position: float = 0.0,
        kill_switch_status: dict = None,
        polymarket_ws_status: dict = None,
    ):
        """
        Write health.json with current bot status.

        Args:
            stats: Trading statistics dict.
            last_binance_update: Timestamp of last Binance price update.
            last_polymarket_fetch: Timestamp of last Polymarket data fetch.
            total_position: Current total position in dollars.
            kill_switch_status: Kill switch status dict.
            polymarket_ws_status: WebSocket feed status dict.
        """
        now = time.time()
        stats = stats or {}

        total_trades = stats.get("wins", 0) + stats.get("losses", 0)
        win_rate = stats.get("wins", 0) / total_trades if total_trades > 0 else 0.0

        # Detect stale Binance feed
        binance_age = now - last_binance_update if last_binance_update > 0 else -1
        binance_stale = binance_age > self.config.timing.stale_feed_timeout if binance_age >= 0 else True

        health = {
            "status": self._status,
            "uptime_seconds": round(now - self._start_time, 1),
            "uptime_human": _format_duration(now - self._start_time),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "binance": {
                "last_update": last_binance_update,
                "age_seconds": round(binance_age, 1) if binance_age >= 0 else None,
                "stale": binance_stale,
            },
            "polymarket": {
                "last_fetch": last_polymarket_fetch,
                "age_seconds": round(now - last_polymarket_fetch, 1) if last_polymarket_fetch > 0 else None,
                "ws": polymarket_ws_status or {"connected": False, "subscriptions": 0, "last_update": None},
            },
            "trading": {
                "pnl": round(stats.get("total_pnl", 0.0), 4),
                "position": round(total_position, 2),
                "win_rate": round(win_rate, 4),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "total_orders": stats.get("total_orders", 0),
                "filled": stats.get("filled", 0),
            },
            "kill_switch": kill_switch_status or {},
            "config_hash": self.config.config_hash,
            "last_error": self._last_error,
            "dry_run": self.config.dry_run,
            "paper_mode": self.config.paper_mode,
        }

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.data_dir, prefix="health_", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(health, f, indent=2)
            os.replace(tmp_path, self._health_file)
            self._last_write = now
            logger.debug("Health written: status=%s, pnl=%.4f", self._status, stats.get("total_pnl", 0))
        except OSError as e:
            logger.error("Failed to write health.json: %s", e)
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass


class MetricsExporter:
    """
    Writes metrics.json for dashboard consumption.

    Updated every 60 seconds with detailed trading metrics including:
    - Order metrics (fill rate, time-to-fill, cancel breakdown)
    - Edge metrics (avg edge at entry/fill, edge accuracy)
    - Latency metrics (CLOB round-trip, feed age, binance→order)
    - P&L metrics (per hour, per asset, per side, drawdown, Sharpe)
    - Position metrics (utilization, time at max)
    - Taker vs maker breakdown

    Args:
        config: Bot configuration.
        data_dir: Directory for metrics file. Defaults to project data/.
    """

    def __init__(self, config: Config, data_dir: str = None):
        self.config = config
        self.data_dir = data_dir or DATA_DIR
        self._metrics_file = os.path.join(self.data_dir, "metrics.json")
        self._last_write = 0.0
        self._trade_history: list[dict] = []  # Recent trades for rolling stats

        # Order lifecycle tracking
        self._order_events: list[dict] = []  # All order events for lifetime calc

        # Latency tracking
        self._clob_latencies: list[float] = []      # CLOB round-trip times (seconds)
        self._feed_ages: list[float] = []            # Polymarket feed age at order time
        self._binance_to_order: list[float] = []     # Binance update → order placement

        # Position utilization samples
        self._position_samples: list[dict] = []  # {"ts": ..., "position": ..., "max": ...}

        # Windows tracking
        self._windows_traded: set = set()
        self._windows_available: set = set()

        os.makedirs(self.data_dir, exist_ok=True)

    @property
    def write_interval(self) -> float:
        return self.config.timing.metrics_write_interval

    def should_write(self) -> bool:
        """Check if it's time to write metrics.json."""
        return time.time() - self._last_write >= self.write_interval

    def record_trade(self, trade: dict):
        """
        Record a completed trade for metrics tracking.

        Args:
            trade: Dict with keys: asset, side, pnl, edge, filled_price, timestamp,
                   edge_at_fill, is_taker, predicted_up, actual_up
        """
        self._trade_history.append(trade)
        # Keep last 500 trades
        if len(self._trade_history) > 500:
            self._trade_history = self._trade_history[-500:]

    def record_order_event(self, event: dict):
        """
        Record an order lifecycle event.

        Args:
            event: Dict with keys: order_id, event_type (placed/filled/cancelled/expired),
                   timestamp, cancel_reason, edge_at_fill, time_to_fill, order_lifetime
        """
        self._order_events.append(event)
        if len(self._order_events) > 1000:
            self._order_events = self._order_events[-1000:]

    def record_clob_latency(self, latency_seconds: float):
        """Record a CLOB round-trip time measurement."""
        self._clob_latencies.append(latency_seconds)
        if len(self._clob_latencies) > 200:
            self._clob_latencies = self._clob_latencies[-200:]

    def record_feed_age(self, age_seconds: float):
        """Record Polymarket feed age at order placement time."""
        self._feed_ages.append(age_seconds)
        if len(self._feed_ages) > 200:
            self._feed_ages = self._feed_ages[-200:]

    def record_binance_to_order_latency(self, latency_seconds: float):
        """Record time from Binance price update to order placement."""
        self._binance_to_order.append(latency_seconds)
        if len(self._binance_to_order) > 200:
            self._binance_to_order = self._binance_to_order[-200:]

    def record_position_sample(self, position: float, max_position: float):
        """Record a position utilization sample."""
        self._position_samples.append({
            "ts": time.time(),
            "position": position,
            "max": max_position,
        })
        if len(self._position_samples) > 500:
            self._position_samples = self._position_samples[-500:]

    def record_window_available(self, window_key: str):
        """Track an available trading window."""
        self._windows_available.add(window_key)

    def record_window_traded(self, window_key: str):
        """Track a window that was traded."""
        self._windows_traded.add(window_key)

    def write(self, stats: dict = None, orders: list = None):
        """
        Write metrics.json with comprehensive trading metrics.

        Args:
            stats: Trading statistics dict.
            orders: List of all orders (for position tracking).
        """
        now = time.time()
        stats = stats or {}
        orders = orders or []

        # Rolling win rate (last 50 trades)
        recent_trades = self._trade_history[-50:]
        recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        rolling_win_rate = recent_wins / len(recent_trades) if recent_trades else 0.0

        # Average edge at entry
        edges = [t.get("edge", 0) for t in self._trade_history if t.get("edge")]
        avg_edge = sum(edges) / len(edges) if edges else 0.0

        # Average edge at fill
        edges_at_fill = [t.get("edge_at_fill", 0) for t in self._trade_history if t.get("edge_at_fill")]
        avg_edge_at_fill = sum(edges_at_fill) / len(edges_at_fill) if edges_at_fill else 0.0

        # Edge accuracy: how often we predicted correctly
        correct_predictions = sum(1 for t in self._trade_history if t.get("predicted_correct"))
        total_with_resolution = sum(1 for t in self._trade_history if "predicted_correct" in t)
        edge_accuracy = correct_predictions / total_with_resolution if total_with_resolution > 0 else 0.0

        # Fill rate
        total_orders = stats.get("total_orders", 0)
        filled = stats.get("filled", 0)
        fill_rate = filled / total_orders if total_orders > 0 else 0.0

        # ── Order Metrics ──
        # Time-to-fill
        fill_times = [e.get("time_to_fill", 0) for e in self._order_events
                      if e.get("event_type") == "filled" and e.get("time_to_fill")]
        avg_time_to_fill = sum(fill_times) / len(fill_times) if fill_times else 0.0

        # Order lifetime
        lifetimes = [e.get("order_lifetime", 0) for e in self._order_events
                     if e.get("order_lifetime")]
        avg_order_lifetime = sum(lifetimes) / len(lifetimes) if lifetimes else 0.0

        # Orders by outcome
        orders_by_outcome = {
            "filled": stats.get("filled", 0),
            "cancelled_edge_gone": 0,
            "cancelled_time": 0,
            "cancelled_other": 0,
            "expired": stats.get("expired", 0),
        }

        # Cancel reasons breakdown
        cancel_reasons = {}
        for e in self._order_events:
            if e.get("event_type") == "cancelled":
                reason = e.get("cancel_reason", "unknown")
                if "edge_gone" in reason:
                    orders_by_outcome["cancelled_edge_gone"] += 1
                    cancel_reasons["edge_gone"] = cancel_reasons.get("edge_gone", 0) + 1
                elif "time" in reason:
                    orders_by_outcome["cancelled_time"] += 1
                    cancel_reasons["time_expiry"] = cancel_reasons.get("time_expiry", 0) + 1
                else:
                    orders_by_outcome["cancelled_other"] += 1
                    cancel_reasons[reason] = cancel_reasons.get(reason, 0) + 1

        # ── Latency Metrics ──
        avg_clob_latency = (sum(self._clob_latencies) / len(self._clob_latencies)
                           if self._clob_latencies else 0.0)
        avg_feed_age = (sum(self._feed_ages) / len(self._feed_ages)
                       if self._feed_ages else 0.0)
        avg_binance_to_order = (sum(self._binance_to_order) / len(self._binance_to_order)
                               if self._binance_to_order else 0.0)

        # ── P&L Metrics ──
        # P&L by side
        pnl_by_side = {"BUY_UP": 0.0, "SELL_UP": 0.0}
        trades_by_side = {"BUY_UP": 0, "SELL_UP": 0}
        for t in self._trade_history:
            side = t.get("side", "")
            if side in pnl_by_side:
                pnl_by_side[side] += t.get("pnl", 0)
                trades_by_side[side] += 1

        # Max drawdown
        cum_pnl = 0.0
        peak = 0.0
        max_drawdown = 0.0
        biggest_win = 0.0
        biggest_loss = 0.0
        pnl_values = []
        for t in self._trade_history:
            pnl = t.get("pnl", 0)
            cum_pnl += pnl
            pnl_values.append(pnl)
            if cum_pnl > peak:
                peak = cum_pnl
            dd = cum_pnl - peak
            if dd < max_drawdown:
                max_drawdown = dd
            if pnl > biggest_win:
                biggest_win = pnl
            if pnl < biggest_loss:
                biggest_loss = pnl

        # Running Sharpe ratio (annualized, using per-trade returns)
        sharpe_ratio = 0.0
        if len(pnl_values) >= 10:
            import math
            mean_pnl = sum(pnl_values) / len(pnl_values)
            variance = sum((p - mean_pnl) ** 2 for p in pnl_values) / len(pnl_values)
            std_pnl = math.sqrt(variance) if variance > 0 else 0
            if std_pnl > 0:
                # Annualize assuming ~288 5-min windows per day
                sharpe_ratio = (mean_pnl / std_pnl) * math.sqrt(288)

        # P&L per hour (runtime-based)
        start_time = self._trade_history[0].get("timestamp", now) if self._trade_history else now
        runtime_hours = max((now - start_time) / 3600, 0.01)
        pnl_per_hour = stats.get("total_pnl", 0) / runtime_hours if runtime_hours > 0 else 0.0

        # Taker vs maker breakdown
        taker_orders = sum(1 for t in self._trade_history if t.get("is_taker"))
        maker_orders = len(self._trade_history) - taker_orders
        taker_pnl = sum(t.get("pnl", 0) for t in self._trade_history if t.get("is_taker"))
        maker_pnl = sum(t.get("pnl", 0) for t in self._trade_history if not t.get("is_taker"))

        # ── Position Metrics ──
        avg_utilization = 0.0
        time_at_max = 0.0
        if self._position_samples:
            utils = [s["position"] / s["max"] if s["max"] > 0 else 0
                    for s in self._position_samples]
            avg_utilization = sum(utils) / len(utils)
            at_max = sum(1 for s in self._position_samples
                        if s["max"] > 0 and s["position"] >= s["max"] * 0.95)
            time_at_max = at_max / len(self._position_samples)

        # Per-asset breakdown
        per_asset = {}
        for t in self._trade_history:
            asset = t.get("asset", "unknown")
            if asset not in per_asset:
                per_asset[asset] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
            per_asset[asset]["trades"] += 1
            per_asset[asset]["pnl"] += t.get("pnl", 0)
            if t.get("pnl", 0) > 0:
                per_asset[asset]["wins"] += 1
            else:
                per_asset[asset]["losses"] += 1

        # Per-hour breakdown (last 24h)
        per_hour = {}
        cutoff_24h = now - 86400
        for t in self._trade_history:
            ts = t.get("timestamp", 0)
            if ts < cutoff_24h:
                continue
            hour = int(ts // 3600) * 3600
            hour_key = datetime.fromtimestamp(hour, tz=timezone.utc).strftime("%H:00")
            if hour_key not in per_hour:
                per_hour[hour_key] = {"trades": 0, "pnl": 0.0}
            per_hour[hour_key]["trades"] += 1
            per_hour[hour_key]["pnl"] += t.get("pnl", 0)

        # Current open positions
        from src.paper_trader import OrderStatus
        open_positions = [
            {"id": o.id, "asset": o.asset, "side": o.side, "size": o.size, "price": o.price}
            for o in orders
            if o.status in (OrderStatus.PENDING, OrderStatus.FILLED)
        ]

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "totals": {
                "orders_placed": stats.get("total_orders", 0),
                "orders_filled": stats.get("filled", 0),
                "orders_cancelled": stats.get("cancelled", 0),
                "wins": stats.get("wins", 0),
                "losses": stats.get("losses", 0),
                "pnl": round(stats.get("total_pnl", 0.0), 4),
                "rebates_est": round(stats.get("total_rebate_est", 0.0), 4),
            },
            "rates": {
                "fill_rate": round(fill_rate, 4),
                "rolling_win_rate_50": round(rolling_win_rate, 4),
                "avg_edge_at_entry": round(avg_edge, 4),
                "avg_edge_at_fill": round(avg_edge_at_fill, 4),
                "edge_accuracy": round(edge_accuracy, 4),
            },
            "order_metrics": {
                "avg_time_to_fill": round(avg_time_to_fill, 2),
                "avg_order_lifetime": round(avg_order_lifetime, 2),
                "orders_by_outcome": orders_by_outcome,
                "cancel_reasons": cancel_reasons,
            },
            "latency": {
                "avg_clob_round_trip_ms": round(avg_clob_latency * 1000, 1),
                "avg_feed_age_ms": round(avg_feed_age * 1000, 1),
                "avg_binance_to_order_ms": round(avg_binance_to_order * 1000, 1),
                "samples": len(self._clob_latencies),
            },
            "pnl_metrics": {
                "pnl_per_hour": round(pnl_per_hour, 4),
                "max_drawdown": round(max_drawdown, 4),
                "biggest_win": round(biggest_win, 4),
                "biggest_loss": round(biggest_loss, 4),
                "sharpe_ratio": round(sharpe_ratio, 2),
                "pnl_by_side": {k: round(v, 4) for k, v in pnl_by_side.items()},
                "trades_by_side": trades_by_side,
            },
            "taker_maker": {
                "taker_orders": taker_orders,
                "maker_orders": maker_orders,
                "taker_pnl": round(taker_pnl, 4),
                "maker_pnl": round(maker_pnl, 4),
            },
            "position_metrics": {
                "avg_utilization": round(avg_utilization, 4),
                "time_at_max_pct": round(time_at_max, 4),
                "windows_traded": len(self._windows_traded),
                "windows_available": len(self._windows_available),
            },
            "per_asset": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                              for kk, vv in v.items()} for k, v in per_asset.items()},
            "per_hour": per_hour,
            "open_positions": open_positions,
            "trade_count": len(self._trade_history),
        }

        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self.data_dir, prefix="metrics_", suffix=".tmp"
            )
            with os.fdopen(fd, "w") as f:
                json.dump(metrics, f, indent=2, default=str)
            os.replace(tmp_path, self._metrics_file)
            self._last_write = now
            logger.debug("Metrics written: %d trades tracked", len(self._trade_history))
        except OSError as e:
            logger.error("Failed to write metrics.json: %s", e)
            try:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
            except OSError:
                pass


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    hours = s // 3600
    minutes = (s % 3600) // 60
    return f"{hours}h {minutes}m"
