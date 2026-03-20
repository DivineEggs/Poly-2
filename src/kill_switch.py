#!/usr/bin/env python3
"""
Kill Switch — Emergency halt mechanism for the trading bot.

Three triggers:
1. File-based: if data/KILL exists, halt immediately
2. Max drawdown: if cumulative P&L drops below threshold, halt
3. Max consecutive losses: if N losses in a row, halt

When triggered:
- Creates the KILL file with the reason
- Stops all new order placement
- Signals the bot to cancel pending orders

To resume: manually delete data/KILL
"""
import os
import time
from datetime import datetime, timezone

from src.logger import get_logger
from src.config import Config, DATA_DIR

logger = get_logger("kill_switch")


class KillSwitch:
    """
    Monitors trading safety conditions and triggers emergency halt.

    Check this every iteration of the main loop.

    Args:
        config: Bot configuration with kill switch thresholds.
        data_dir: Directory for the KILL file. Defaults to project data/.
    """

    def __init__(self, config: Config, data_dir: str = None):
        self.config = config
        self.data_dir = data_dir or DATA_DIR
        self._kill_file = os.path.join(self.data_dir, "KILL")
        self._triggered = False
        self._trigger_reason = ""
        self._consecutive_losses = 0

        os.makedirs(self.data_dir, exist_ok=True)

        # Check if KILL file already exists on startup
        if os.path.exists(self._kill_file):
            self._triggered = True
            try:
                with open(self._kill_file, "r") as f:
                    self._trigger_reason = f.read().strip() or "KILL file present on startup"
            except OSError:
                self._trigger_reason = "KILL file present on startup"
            logger.critical("Kill switch ACTIVE on startup: %s", self._trigger_reason)

    @property
    def is_triggered(self) -> bool:
        """Check if the kill switch is currently active."""
        return self._triggered

    @property
    def reason(self) -> str:
        """Reason the kill switch was triggered."""
        return self._trigger_reason

    def check(self, cumulative_pnl: float = 0.0) -> bool:
        """
        Run all kill switch checks. Call this every loop iteration.

        Args:
            cumulative_pnl: Current cumulative P&L in dollars.

        Returns:
            True if kill switch is triggered (halt trading).
        """
        # Already triggered — check if user removed the file to resume
        if self._triggered:
            if not os.path.exists(self._kill_file):
                logger.info("KILL file removed — resuming trading")
                self._triggered = False
                self._trigger_reason = ""
                self._consecutive_losses = 0
                return False
            return True

        # Check 1: File-based kill switch
        if os.path.exists(self._kill_file):
            try:
                with open(self._kill_file, "r") as f:
                    reason = f.read().strip() or "Manual KILL file"
            except OSError:
                reason = "Manual KILL file"
            self._trigger("file", reason)
            return True

        # Check 2: Max drawdown
        max_dd = self.config.kill_switch.max_drawdown
        if cumulative_pnl < max_dd:
            self._trigger(
                "drawdown",
                f"Cumulative P&L ${cumulative_pnl:.2f} below max drawdown ${max_dd:.2f}",
            )
            return True

        # Check 3: Max consecutive losses
        max_losses = self.config.kill_switch.max_consecutive_losses
        if self._consecutive_losses >= max_losses:
            self._trigger(
                "consecutive_losses",
                f"{self._consecutive_losses} consecutive losses (limit: {max_losses})",
            )
            return True

        return False

    def record_trade_result(self, is_win: bool):
        """
        Record a trade result for consecutive loss tracking.

        Args:
            is_win: True if the trade was profitable.
        """
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1
            logger.debug("Consecutive losses: %d / %d",
                         self._consecutive_losses,
                         self.config.kill_switch.max_consecutive_losses)

    def set_consecutive_losses(self, count: int):
        """Restore consecutive loss count from persisted state."""
        self._consecutive_losses = count

    def _trigger(self, trigger_type: str, reason: str):
        """Activate the kill switch and create the KILL file."""
        self._triggered = True
        self._trigger_reason = reason

        logger.critical("🚨 KILL SWITCH TRIGGERED [%s]: %s", trigger_type, reason)

        # Write KILL file
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            content = (
                f"Kill switch triggered at {timestamp}\n"
                f"Type: {trigger_type}\n"
                f"Reason: {reason}\n"
                f"\nTo resume: delete this file (data/KILL)\n"
            )
            with open(self._kill_file, "w") as f:
                f.write(content)
        except OSError as e:
            logger.error("Failed to write KILL file: %s", e)

    def force_trigger(self, reason: str = "Manual trigger"):
        """Manually trigger the kill switch."""
        self._trigger("manual", reason)

    def status(self) -> dict:
        """Return current kill switch status."""
        return {
            "triggered": self._triggered,
            "reason": self._trigger_reason,
            "consecutive_losses": self._consecutive_losses,
            "max_consecutive_losses": self.config.kill_switch.max_consecutive_losses,
            "max_drawdown": self.config.kill_switch.max_drawdown,
            "kill_file_exists": os.path.exists(self._kill_file),
        }
