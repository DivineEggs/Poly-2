#!/usr/bin/env python3
"""
CLOB Client — Wrapper around py-clob-client for real order submission.

Handles:
  - Level 2 authentication (API key derivation)
  - Order placement with neg_risk support
  - Order cancellation (single + all)
  - Order status polling
  - Thread-safe CLOB access
  - Rate limiting and error handling
"""
import os
import threading
import time

from py_clob_client.client import ClobClient
from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions

from src.config import Config, PROJECT_ROOT
from src.logger import get_logger
from src.rate_limiter import RateLimiter

logger = get_logger("clob")

# Path to .env file
_ENV_PATH = os.path.join(PROJECT_ROOT, ".env")


def _load_env_value(key_name: str) -> str:
    """Load a value from .env file by key name."""
    if not os.path.exists(_ENV_PATH):
        raise FileNotFoundError(f".env file not found at {_ENV_PATH}")

    with open(_ENV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() == key_name:
                return value.strip().strip("'\"")

    return ""


def _load_private_key() -> str:
    """Load POLYGON_PRIVATE_KEY from .env file."""
    value = _load_env_value("POLYGON_PRIVATE_KEY")
    if not value:
        raise ValueError("POLYGON_PRIVATE_KEY not found or empty in .env")
    return value


def _load_funder() -> str:
    """Load PROXY_WALLET from .env file."""
    return _load_env_value("PROXY_WALLET")


class ClobManager:
    """
    Manages the Polymarket CLOB client for real order submission.

    Thread-safe wrapper around py-clob-client with rate limiting,
    error handling, and logging.

    Args:
        config: Bot configuration object.
    """

    CLOB_HOST = "https://clob.polymarket.com"
    CHAIN_ID = 137
    SIGNATURE_TYPE = 2  # POLY_GNOSIS_SAFE (MetaMask via Polymarket proxy)
    def __init__(self, config: Config):
        self.config = config
        self._lock = threading.Lock()

        # Rate limiter — Polymarket allows ~20 req/s, be generous
        self._limiter = RateLimiter(
            "clob_orders",
            rate=15,
            burst=20,
        )

        # Initialize the CLOB client at Level 2
        logger.info("Initializing CLOB client (Level 2)...")
        private_key = _load_private_key()
        funder = _load_funder()

        # Step 1: Create base client
        base_client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            signature_type=2,
            funder=funder,
        )

        # Step 2: Derive API credentials
        creds = base_client.create_or_derive_api_creds()
        logger.info("API credentials derived successfully")

        # Step 3: Re-create client with creds (Level 2)
        self._client = ClobClient(
            host="https://clob.polymarket.com",
            chain_id=137,
            key=private_key,
            creds=creds,
            signature_type=2,
            funder=funder,
        )

        self._funder = funder
        self._private_key = private_key
        logger.info("CLOB client initialized — address: %s", self._client.get_address())

    def get_usdc_balance(self) -> float:
        """Get USDC balance from Polymarket account."""
        try:
            # Use CLOB client's native balance method (checks Polymarket ledger)
            balance = self._client.get_balance()
            if balance is not None:
                return float(balance)
            return -1.0
        except Exception as e:
            logger.warning("Failed to check balance: %s", e)
            return -1.0

    def place_order(self, token_id: str, side: str, price: float, size: float,
                    neg_risk: bool = False) -> dict:
        """
        Place a limit order on the CLOB.

        Args:
            token_id: The token ID for the outcome (UP or DOWN).
            side: "BUY" or "SELL".
            price: Probability price (0.01–0.99).
            size: Order size in USDC.
            neg_risk: Whether this is a neg_risk market (default False, auto-detected by client).

        Returns:
            Dict with 'orderID', 'success', and raw response data.
            On failure, returns {'orderID': None, 'success': False, 'error': str}.
        """
        self._limiter.acquire()

        order_args = OrderArgs(
            token_id=token_id,
            price=price,
            size=size,
            side=side.upper(),
        )

        # Let the client auto-detect neg_risk unless explicitly set
        options = None
        if neg_risk:
            options = PartialCreateOrderOptions(neg_risk=True)

        logger.info("Placing %s order: token=%s…%s price=%.2f size=$%.2f",
                     side, token_id[:8], token_id[-4:], price, size)

        try:
            # Retry up to 3 times on connection failures
            result = None
            last_err = None
            for attempt in range(3):
                try:
                    with self._lock:
                        result = self._client.create_and_post_order(order_args, options=options)
                    break  # Success
                except Exception as retry_err:
                    last_err = retry_err
                    err_str = str(retry_err).lower()
                    if "request exception" in err_str or "connection" in err_str or "timeout" in err_str:
                        import time as _time
                        wait = 0.2 * (attempt + 1)  # 0.2s, 0.4s, 0.6s — fast retries
                        logger.warning("Order attempt %d failed (retrying in %.1fs): %s",
                                      attempt + 1, wait, retry_err)
                        _time.sleep(wait)  # Sleep OUTSIDE the lock
                        continue
                    else:
                        raise  # Non-retryable error
            else:
                # All retries exhausted
                raise last_err

            # result is typically a dict with 'orderID', 'success', etc.
            if isinstance(result, dict):
                order_id = result.get("orderID") or result.get("order_id") or result.get("id")
                success = result.get("success", order_id is not None)
            else:
                # Handle unexpected response format
                order_id = str(result) if result else None
                success = order_id is not None

            if success and order_id:
                logger.info("✅ Order placed: %s %s @ %.2f ($%.2f) → ID: %s",
                            side, token_id[:8], price, size, order_id)
            else:
                error_msg = result.get("errorMsg", str(result)) if isinstance(result, dict) else str(result)
                logger.error("❌ Order rejected: %s %s @ %.2f — %s",
                             side, token_id[:8], price, error_msg)

            return {
                "orderID": order_id,
                "success": bool(success),
                "raw": result,
            }

        except Exception as e:
            logger.error("❌ Order placement failed: %s %s @ %.2f — %s",
                         side, token_id[:8], price, e)
            return {
                "orderID": None,
                "success": False,
                "error": str(e),
            }

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order by ID.

        Args:
            order_id: The CLOB order ID to cancel.

        Returns:
            True if cancellation succeeded, False otherwise.
        """
        if not order_id:
            logger.warning("Cannot cancel order: no order ID provided")
            return False

        self._limiter.acquire()

        try:
            with self._lock:
                result = self._client.cancel_orders([order_id])

            logger.info("Order cancelled: %s (result: %s)", order_id, result)
            return True

        except Exception as e:
            logger.error("Failed to cancel order %s: %s", order_id, e)
            return False

    def cancel_all(self) -> int:
        """
        Cancel all open orders.

        Returns:
            Number of orders that were cancelled (best effort count).
        """
        self._limiter.acquire()

        try:
            # First get count of open orders
            open_orders = self.get_open_orders()
            count = len(open_orders)

            if count == 0:
                logger.info("No open orders to cancel")
                return 0

            with self._lock:
                result = self._client.cancel_all()

            logger.info("Cancelled all orders (%d open) — result: %s", count, result)
            return count

        except Exception as e:
            logger.error("Failed to cancel all orders: %s", e)
            return 0

    def get_open_orders(self) -> list:
        """
        Get all currently open orders.

        Returns:
            List of open order dicts from the CLOB.
        """
        self._limiter.acquire()

        try:
            with self._lock:
                result = self._client.get_orders()

            # result may be a list or a paginated response
            if isinstance(result, list):
                return result
            elif isinstance(result, dict):
                return result.get("orders", result.get("data", []))
            else:
                return []

        except Exception as e:
            logger.error("Failed to fetch open orders: %s", e)
            return []

    def get_order_status(self, order_id: str) -> dict:
        """
        Get the status of a specific order.

        Args:
            order_id: The CLOB order ID.

        Returns:
            Dict with order details including fill status.
            Returns empty dict on failure.
        """
        if not order_id:
            return {}

        self._limiter.acquire()

        try:
            with self._lock:
                result = self._client.get_order(order_id)

            if isinstance(result, dict):
                return result
            else:
                return {"raw": result}

        except Exception as e:
            logger.error("Failed to get order status for %s: %s", order_id, e)
            return {}
