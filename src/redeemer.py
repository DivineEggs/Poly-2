"""Auto-redeem resolved Polymarket positions via web3.py + CTF contract."""
import os
import time
import requests
from src.logger import get_logger

logger = get_logger("redeemer")

# Polygon contracts
CTF_ADDRESS  = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
USDC_ADDRESS = "0x3c499c542cEF5E3811e1192ce70d8cC03d5c3359"  # Native USDC on Polygon
RPC_URL      = "https://polygon-bor-rpc.publicnode.com"

# Minimal CTF ABI — only redeemPositions
CTF_ABI = [
    {
        "name": "redeemPositions",
        "type": "function",
        "stateMutability": "nonpayable",
        "inputs": [
            {"name": "collateralToken", "type": "address"},
            {"name": "parentCollectionId", "type": "bytes32"},
            {"name": "conditionId", "type": "bytes32"},
            {"name": "indexSets", "type": "uint256[]"},
        ],
        "outputs": [],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "account", "type": "address"},
            {"name": "id", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


def _load_env():
    env = {}
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip("'\"")
    except Exception:
        pass
    return env


class AutoRedeemer:
    """Periodically redeems resolved Polymarket positions using web3.py."""

    def __init__(self, config):
        self.config = config
        self._w3 = None
        self._ctf = None
        self._account = None
        self._proxy_wallet = None
        self._last_redeem = 0
        self._redeem_interval = 60
        self._initialized = False

    def _init(self):
        if self._initialized:
            return self._w3 is not None

        self._initialized = True
        try:
            from web3 import Web3
            from eth_account import Account

            env = _load_env()
            pk = env.get("POLYGON_PRIVATE_KEY", "")
            proxy = env.get("PROXY_WALLET", "")

            if not pk:
                logger.warning("Auto-redeemer disabled: POLYGON_PRIVATE_KEY not set")
                return False

            self._w3 = Web3(Web3.HTTPProvider(RPC_URL))
            if not self._w3.is_connected():
                logger.warning("Auto-redeemer disabled: cannot connect to Polygon RPC")
                return False

            self._account = Account.from_key(pk)
            self._proxy_wallet = Web3.to_checksum_address(proxy) if proxy else self._account.address
            self._ctf = self._w3.eth.contract(
                address=Web3.to_checksum_address(CTF_ADDRESS),
                abi=CTF_ABI,
            )
            logger.info("✅ Auto-redeemer initialized (wallet: %s)", self._proxy_wallet[:10])
            return True

        except Exception as e:
            logger.error("Auto-redeemer init failed: %s", e)
            return False

    def check_and_redeem(self):
        """Check for redeemable positions every 60s and redeem them."""
        now = time.time()
        if now - self._last_redeem < self._redeem_interval:
            return
        self._last_redeem = now

        if not self._init():
            return

        try:
            self._do_redeem()
        except Exception as e:
            logger.error("Auto-redeem error: %s", e)

    def _do_redeem(self):
        """Fetch resolved positions and call redeemPositions for each."""
        # Get positions from Polymarket data API
        try:
            r = requests.get(
                f"https://data-api.polymarket.com/positions?user={self._proxy_wallet}",
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10,
            )
            if r.status_code != 200:
                return
            positions = r.json()
        except Exception as e:
            logger.debug("Positions fetch error: %s", e)
            return

        if not positions:
            return

        redeemed = 0
        for pos in positions:
            try:
                # Only redeem if price is 1.0 (resolved winner) or 0.0 (auto-resolved loser)
                cur_price = float(pos.get("curPrice", 0))
                size = float(pos.get("size", 0))
                if size < 0.1:
                    continue
                if cur_price < 0.99:
                    continue  # Not resolved yet

                condition_id = pos.get("conditionId", "")
                if not condition_id:
                    continue

                # Determine index set from outcome (Up=1, Down=2, Yes=1, No=2)
                outcome = pos.get("outcome", "").lower()
                index_set = 1 if outcome in ("yes", "up") else 2

                success = self._redeem_position(condition_id, index_set)
                if success:
                    redeemed += 1
                    logger.info("💰 Redeemed %s %.1f shares (outcome=%s)",
                                pos.get("title", "")[:30], size, outcome)
            except Exception as e:
                logger.debug("Redeem error for position: %s", e)

        if redeemed:
            logger.info("💰 Auto-redeemed %d positions", redeemed)

    def _redeem_position(self, condition_id_hex: str, index_set: int) -> bool:
        """Call CTF.redeemPositions for a single condition."""
        try:
            from web3 import Web3

            # condition_id must be bytes32
            if condition_id_hex.startswith("0x"):
                condition_id = bytes.fromhex(condition_id_hex[2:].zfill(64))
            else:
                condition_id = bytes.fromhex(condition_id_hex.zfill(64))

            parent_collection_id = b"\x00" * 32

            nonce = self._w3.eth.get_transaction_count(self._account.address)
            gas_price = self._w3.eth.gas_price

            tx = self._ctf.functions.redeemPositions(
                Web3.to_checksum_address(USDC_ADDRESS),
                parent_collection_id,
                condition_id,
                [index_set],
            ).build_transaction({
                "from": self._account.address,
                "nonce": nonce,
                "gas": 200_000,
                "gasPrice": int(gas_price * 1.2),
                "chainId": 137,
            })

            signed = self._w3.eth.account.sign_transaction(tx, self._account.key)
            tx_hash = self._w3.eth.send_raw_transaction(signed.raw_transaction)
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash, timeout=60)

            if receipt.status == 1:
                logger.info("✅ Redeem tx confirmed: %s", tx_hash.hex()[:16])
                return True
            else:
                logger.warning("⚠️ Redeem tx failed: %s", tx_hash.hex()[:16])
                return False

        except Exception as e:
            logger.debug("Redeem tx error: %s", e)
            return False
