#!/usr/bin/env python3
"""
Configuration loader — Loads and validates config.yaml.

Provides a typed Config object with all tunable parameters.
Falls back to sensible defaults if config file is missing.
"""
import os
import hashlib
from dataclasses import dataclass, field

import yaml

# Project root (parent of src/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


@dataclass
class TradingConfig:
    spread_capture_enabled: bool = True
    order_size: float = 5.0
    max_position: float = 30.0
    max_single_exposure: float = 10.0
    min_spread_profit: float = 0.02
    max_orders_per_window: int = 1
    cancel_time_remaining: int = 15
    allowed_assets: list = field(default_factory=lambda: ["BTC", "ETH", "SOL"])
    allowed_timeframes: list = field(default_factory=lambda: ["5m"])
    max_concurrent_pairs: int = 1      # One trade at a time (low capital mode)
    rechase_enabled: bool = True       # Re-chase unfilled side
    rechase_max_attempts: int = 3      # Max re-chase attempts
    rechase_wait_seconds: float = 5.0  # Wait before first re-chase
    stop_loss_enabled: bool = True     # SL after failed re-chase
    stop_loss_cents: float = 0.03      # Max loss per token on single-side exit
    min_usdc_balance: float = 3.0      # Min USDC to trade
    # Legacy fields (kept for backward compat, unused in V2)
    min_edge: float = 0.05
    cancel_edge_threshold: float = 0.01
    spread_buffer: float = 0.01


@dataclass
class ShieldConfig:
    threshold: float = 0.001     # 0.1% price move triggers cancel
    cancel_timeout_ms: int = 500
    enabled: bool = True


@dataclass
class KillSwitchConfig:
    max_drawdown: float = -20.0
    max_consecutive_losses: int = 10


@dataclass
class TimingConfig:
    scan_interval: int = 30
    order_check_interval: int = 2
    display_interval: int = 10
    market_fetch_interval: int = 3
    health_write_interval: int = 30
    metrics_write_interval: int = 60
    stale_feed_timeout: int = 30
    shutdown_timeout: int = 10
    # Legacy
    vol_refresh_interval: int = 300


@dataclass
class ApiConfig:
    gamma_url: str = "https://gamma-api.polymarket.com"
    clob_url: str = "https://clob.polymarket.com"
    binance_ws_url: str = "wss://stream.binance.com:9443/stream"
    binance_rest_url: str = "https://api.binance.com"
    gamma_rate_limit: int = 10
    gamma_burst: int = 20
    clob_rate_limit: int = 5
    clob_burst: int = 10
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 60.0


@dataclass
class TakerModeConfig:
    enabled: bool = True
    min_edge: float = 0.15
    max_taker_fee_bps: int = 200


@dataclass
class ArbConfig:
    enabled: bool = True
    order_size: float = 5.0
    max_exposure: float = 15.0
    min_edge: float = 0.05
    cooldown_seconds: float = 30.0
    move_threshold_pct: float = 0.2
    time_weight_early: float = 0.3
    time_weight_late: float = 0.7
    sell_discount_from_fair: float = 0.02
    sell_timeout_seconds: float = 60.0


@dataclass
class SnipeConfig:
    enabled: bool = True
    min_buy_price: float = 0.79        # global fallback (overridden per asset below)
    max_buy_price: float = 0.99
    maker_bid_offset: float = 0.01
    min_edge: float = 0.01
    entry_times: tuple = (15, 10, 6)
    taker_seconds_remaining: int = 6
    min_shares: float = 5.0
    min_dollar_move: float = 50.0      # default fallback (overridden by per-asset below)
    dollar_move_btc: float = 50.0      # BTC: $50
    dollar_move_eth: float = 50.0      # ETH: $50 = ~1.4% move (working well)
    dollar_move_sol: float = 0.0       # SOL: disabled
    min_buy_price_btc: float = 0.85    # BTC: 85c minimum
    min_buy_price_eth: float = 0.79    # ETH: 79c (leave alone for now)
    disabled_assets: list = field(default_factory=lambda: ["SOL"])  # SOL disabled
    max_concurrent: int = 2
    min_seconds_remaining: int = 3


@dataclass
class LoggingConfig:
    level: str = "INFO"
    file_level: str = "DEBUG"
    log_file: str = "data/logs/bot.log"
    max_bytes: int = 10_485_760  # 10 MB
    backup_count: int = 5


@dataclass
class Config:
    """Top-level configuration object."""
    trading: TradingConfig = field(default_factory=TradingConfig)
    shield: ShieldConfig = field(default_factory=ShieldConfig)
    kill_switch: KillSwitchConfig = field(default_factory=KillSwitchConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    taker_mode: TakerModeConfig = field(default_factory=TakerModeConfig)
    arb: ArbConfig = field(default_factory=ArbConfig)
    snipe: SnipeConfig = field(default_factory=SnipeConfig)
    dry_run: bool = False
    paper_mode: bool = False
    shield_threshold: float = 0.001  # Convenience access for market_maker
    _raw_yaml: str = ""
    _config_hash: str = ""

    @property
    def config_hash(self) -> str:
        """SHA256 hash of the raw config for change detection."""
        return self._config_hash

    def resolve_path(self, relative_path: str) -> str:
        """Resolve a path relative to the project root."""
        if os.path.isabs(relative_path):
            return relative_path
        return os.path.join(PROJECT_ROOT, relative_path)


def _apply_section(target, data: dict):
    """Apply a dict of values onto a dataclass instance, ignoring unknown keys."""
    if not data or not isinstance(data, dict):
        return
    for key, value in data.items():
        if hasattr(target, key):
            expected_type = type(getattr(target, key))
            try:
                if expected_type == list and isinstance(value, list):
                    setattr(target, key, value)
                elif expected_type == bool:
                    setattr(target, key, bool(value))
                elif expected_type == int:
                    setattr(target, key, int(value))
                elif expected_type == float:
                    setattr(target, key, float(value))
                elif expected_type == str:
                    setattr(target, key, str(value))
                else:
                    setattr(target, key, value)
            except (ValueError, TypeError):
                pass  # Keep default if conversion fails


def _validate(cfg: Config) -> list[str]:
    """Validate config values. Returns list of error messages (empty = OK)."""
    errors = []
    t = cfg.trading
    if t.min_edge < 0 or t.min_edge > 1:
        errors.append(f"trading.min_edge must be 0-1, got {t.min_edge}")
    if t.order_size <= 0:
        errors.append(f"trading.order_size must be > 0, got {t.order_size}")
    if t.max_position <= 0:
        errors.append(f"trading.max_position must be > 0, got {t.max_position}")
    if t.max_orders_per_window < 1:
        errors.append(f"trading.max_orders_per_window must be >= 1, got {t.max_orders_per_window}")
    if t.order_size > t.max_position:
        errors.append(f"trading.order_size ({t.order_size}) > max_position ({t.max_position})")
    if not t.allowed_assets:
        errors.append("trading.allowed_assets cannot be empty")
    if not t.allowed_timeframes:
        errors.append("trading.allowed_timeframes cannot be empty")

    k = cfg.kill_switch
    if k.max_drawdown >= 0:
        errors.append(f"kill_switch.max_drawdown must be negative, got {k.max_drawdown}")
    if k.max_consecutive_losses < 1:
        errors.append(f"kill_switch.max_consecutive_losses must be >= 1, got {k.max_consecutive_losses}")

    if cfg.timing.stale_feed_timeout < 5:
        errors.append(f"timing.stale_feed_timeout must be >= 5, got {cfg.timing.stale_feed_timeout}")

    if cfg.api.max_retries < 0:
        errors.append(f"api.max_retries must be >= 0, got {cfg.api.max_retries}")

    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if cfg.logging.level.upper() not in valid_levels:
        errors.append(f"logging.level must be one of {valid_levels}, got {cfg.logging.level}")

    return errors


def load_config(path: str = None) -> Config:
    """
    Load configuration from a YAML file.

    Args:
        path: Path to config.yaml. Defaults to project root config.yaml.

    Returns:
        Validated Config object.

    Raises:
        ValueError: If config has validation errors.
        FileNotFoundError: If path is specified but doesn't exist.
    """
    cfg = Config()
    config_path = path or DEFAULT_CONFIG_PATH

    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            raw = f.read()
        cfg._raw_yaml = raw
        cfg._config_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
        data = yaml.safe_load(raw) or {}
    else:
        if path:  # Explicit path that doesn't exist
            raise FileNotFoundError(f"Config file not found: {config_path}")
        data = {}
        cfg._config_hash = "defaults"

    _apply_section(cfg.trading, data.get("trading"))
    _apply_section(cfg.shield, data.get("shield"))
    _apply_section(cfg.kill_switch, data.get("kill_switch"))
    _apply_section(cfg.timing, data.get("timing"))
    _apply_section(cfg.api, data.get("api"))
    _apply_section(cfg.logging, data.get("logging"))
    _apply_section(cfg.taker_mode, data.get("taker_mode"))
    _apply_section(cfg.arb, data.get("arb"))
    _apply_section(cfg.snipe, data.get("snipe"))

    # Sync shield threshold for convenience
    cfg.shield_threshold = cfg.shield.threshold

    if "dry_run" in data:
        cfg.dry_run = bool(data["dry_run"])
    if "paper_mode" in data:
        cfg.paper_mode = bool(data["paper_mode"])

    # Convert allowed_assets to set-compatible format (uppercase)
    cfg.trading.allowed_assets = [a.upper() for a in cfg.trading.allowed_assets]

    errors = _validate(cfg)
    if errors:
        raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))

    return cfg


# Singleton for global access
_global_config: Config | None = None


def get_config() -> Config:
    """Get the global config (loads defaults if not yet loaded)."""
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_config(cfg: Config):
    """Set the global config (used during startup)."""
    global _global_config
    _global_config = cfg
