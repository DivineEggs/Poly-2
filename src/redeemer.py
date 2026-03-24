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
        self._redeem_interval = 15  # Check every 15 seconds
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

            funder = env.get("PROXY_WALLET", "")

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
            logger.info("✅ Auto-redeemer initialized (builder relayer)")
            return True

        except ImportError as e:
            logger.warning("Auto-redeemer disabled: missing package (%s) — falling back to web3.py", e)
            self._initialized = True
            return self._init_web3_fallback()

        except Exception as e:
            logger.error("Failed to initialize auto-redeemer: %s", e)
            self._initialized = True
            return False

    def _init_web3_fallback(self):
        """Fallback redeemer using web3.py directly if poly_web3 not installed."""
        try:
            from web3 import Web3
            from eth_account import Account

            with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")) as f:
                env = {}
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip().strip("'\"")

            pk = env.get("POLYGON_PRIVATE_KEY", "")
            proxy = env.get("PROXY_WALLET", "")
            if not pk:
                return False

            self._w3 = Web3(Web3.HTTPProvider("https://polygon-bor-rpc.publicnode.com"))
            self._account = Account.from_key(pk)
            self._proxy_wallet = Web3.to_checksum_address(proxy) if proxy else self._account.address
            self._use_web3_fallback = True
            logger.info("✅ Auto-redeemer initialized (web3.py fallback)")
            return True
        except Exception as e:
            logger.error("web3 fallback init failed: %s", e)
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
            if getattr(self, '_use_web3_fallback', False):
                self._web3_redeem()
            else:
                result = self._service.redeem_all(batch_size=10)
                if result:
                    success = [r for r in result if r is not None]
                    failed = [r for r in result if r is None]
                    if success:
                        logger.info("💰 Auto-redeemed %d positions", len(success))
                    if failed:
                        logger.warning("⚠️ %d redemptions failed — will retry", len(failed))
        except Exception as e:
            logger.error("Auto-redeem error: %s", e)

    def _web3_redeem(self):
        """Web3 fallback: redeem via CTF contract directly."""
        import requests
        from web3 import Web3

        CTF_ADDRESS  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
        USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"
        CTF_ABI = [{
            "name": "redeemPositions", "type": "function", "stateMutability": "nonpayable",
            "inputs": [
                {"name": "collateralToken", "type": "address"},
                {"name": "parentCollectionId", "type": "bytes32"},
                {"name": "conditionId", "type": "bytes32"},
                {"name": "indexSets", "type": "uint256[]"},
            ], "outputs": [],
        }]

        try:
            r = requests.get(
                f"https://data-api.polymarket.com/positions?user={self._proxy_wallet}",
                headers={"User-Agent": "Mozilla/5.0"}, timeout=10,
            )
            positions = r.json() if r.status_code == 200 else []
        except Exception:
            return

        ctf = self._w3.eth.contract(address=Web3.to_checksum_address(CTF_ADDRESS), abi=CTF_ABI)
        redeemed = 0

        for pos in positions:
            try:
                if float(pos.get("curPrice", 0)) < 0.99:
                    continue
                if float(pos.get("size", 0)) < 0.1:
                    continue
                cid = pos.get("conditionId", "")
                if not cid:
                    continue
                outcome = pos.get("outcome", "").lower()
                index_set = 1 if outcome in ("yes", "up") else 2
                cid_bytes = bytes.fromhex(cid[2:].zfill(64) if cid.startswith("0x") else cid.zfill(64))
                nonce = self._w3.eth.get_transaction_count(self._account.address)
                tx = ctf.functions.redeemPositions(
                    Web3.to_checksum_address(USDC_ADDRESS), b"\x00"*32, cid_bytes, [index_set]
                ).build_transaction({
                    "from": self._account.address, "nonce": nonce,
                    "gas": 200_000, "gasPrice": int(self._w3.eth.gas_price * 1.2), "chainId": 137,
                })
                signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
                tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
                receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)
                if receipt.status == 1:
                    redeemed += 1
                    logger.info("💰 Redeemed position (web3): %s", tx_hash.hex()[:16])
            except Exception as e:
                logger.debug("web3 redeem error: %s", e)

        if redeemed:
            logger.info("💰 web3 fallback redeemed %d positions", redeemed)
