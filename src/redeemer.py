"""Auto-redeem resolved Polymarket positions via Builder Relayer."""
import os
import time
from src.logger import get_logger

logger = get_logger("redeemer")


class AutoRedeemer:
    """Periodically redeems all claimable positions via poly-web3."""

    def __init__(self, config):
        self.config = config
        self._service = None
        self._last_redeem = 0
        self._redeem_interval = 60  # Check every 60 seconds
        self._initialized = False

    def _init_service(self):
        """Lazy-initialize the PolyWeb3Service."""
        if self._initialized:
            return self._service is not None

        try:
            # Load env
            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")) as f:
                env = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip().strip("'\"")

            builder_key = env.get("BUILDER_KEY")
            builder_secret = env.get("BUILDER_SECRET")
            builder_passphrase = env.get("BUILDER_PASSPHRASE")
            private_key = env.get("POLYGON_PRIVATE_KEY")

            if not all([builder_key, builder_secret, builder_passphrase, private_key]):
                logger.warning("Builder API keys not configured — auto-redeem disabled")
                self._initialized = True
                return False

            from py_clob_client.client import ClobClient
            from py_builder_relayer_client.client import RelayClient
            from py_builder_signing_sdk.config import BuilderConfig
            from py_builder_signing_sdk.sdk_types import BuilderApiKeyCreds
            from poly_web3 import RELAYER_URL, PolyWeb3Service

            funder = "os.getenv("PROXY_WALLET", "YOUR_PROXY_WALLET")"

            client = ClobClient(
                "https://clob.polymarket.com",
                key=private_key,
                chain_id=137,
                signature_type=2,
                funder=funder,
            )
            client.set_api_creds(client.create_or_derive_api_creds())

            relayer_client = RelayClient(
                RELAYER_URL,
                137,
                private_key,
                BuilderConfig(
                    local_builder_creds=BuilderApiKeyCreds(
                        key=builder_key,
                        secret=builder_secret,
                        passphrase=builder_passphrase,
                    )
                ),
            )

            self._service = PolyWeb3Service(
                clob_client=client,
                relayer_client=relayer_client,
                rpc_url="https://polygon-bor-rpc.publicnode.com",
            )

            self._initialized = True
            logger.info("✅ Auto-redeemer initialized")
            return True

        except Exception as e:
            logger.error("Failed to initialize auto-redeemer: %s", e)
            self._initialized = True
            return False

    def check_and_redeem(self):
        """Check for redeemable positions and redeem them. Call periodically."""
        now = time.time()
        if now - self._last_redeem < self._redeem_interval:
            return

        self._last_redeem = now

        if not self._init_service():
            return

        try:
            result = self._service.redeem_all(batch_size=10)
            if result:
                # Filter out None (failed) results
                success = [r for r in result if r is not None]
                failed = [r for r in result if r is None]
                if success:
                    logger.info("💰 Auto-redeemed %d positions", len(success))
                if failed:
                    logger.warning("⚠️ %d redemptions failed — will retry", len(failed))
            # Empty list = nothing to redeem, that's fine
        except Exception as e:
            logger.error("Auto-redeem error: %s", e)
