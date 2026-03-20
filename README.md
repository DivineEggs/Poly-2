# Polymarket Spread Bot

Two-sided market making bot for Polymarket crypto prediction markets (BTC/ETH/SOL 5-minute up/down).

## How It Works

1. **Posts BUY orders on BOTH sides** (Up and Down) of a 5-minute crypto prediction market
2. **If both fill** → guaranteed profit (you paid < $1.00 total, one side always pays $1.00)
3. **Binance latency shield** → monitors real-time Binance price to cancel the losing side before it fills

The edge: Polymarket order books lag behind Binance by 10-30 seconds. The bot uses this delay to cancel bad positions before the market catches up.

## Key Features

- **Maker-only orders** — 0% fees + maker rebate (vs 1.5%+ taker fees)
- **Real-time Binance WebSocket** — sub-second price updates
- **Polymarket WebSocket** — real-time order book and fill notifications
- **Auto-redemption** — automatically claims winnings from resolved markets
- **Kill switch** — stops trading at configurable max drawdown
- **Paper mode** — test without real money

## Setup

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install py-clob-client python-dotenv websockets aiohttp pyyaml requests web3
```

### 2. Configure credentials
```bash
cp .env.example .env
# Edit .env with your wallet private key and API credentials
```

### 3. Configure trading
Edit `config.yaml` — key settings:
- `trading.order_size` — dollars per order (minimum $5)
- `trading.max_position` — max total exposure
- `kill_switch.max_drawdown` — stop loss (e.g., -20)
- `paper_mode` — set `true` to test without real money

### 4. Run
```bash
python3 -m src.trader_v2
```

## Architecture

```
src/
├── trader_v2.py        # Main trading loop
├── market_maker.py     # Two-sided order management + shield logic
├── arb_engine.py       # Directional arbitrage engine
├── clob_client.py      # Polymarket CLOB API wrapper
├── price_feed.py       # Binance WebSocket price feed
├── window_tracker.py   # Tracks active 5-min market windows
├── market_scanner.py   # Discovers new markets via Gamma API
├── orderbook.py        # Order book fetching and parsing
├── redeemer.py         # Auto-redemption of winning positions
├── kill_switch.py      # Safety: stops bot at max loss
├── health.py           # Health monitoring and metrics
├── state_manager.py    # Persistent state across restarts
├── logger.py           # Logging setup
└── config.py           # Configuration loading
```

## Requirements

- Python 3.9+
- Polymarket account with USDC deposited
- MetaMask wallet connected to Polymarket

## Disclaimer

This is experimental trading software. Use at your own risk. Start with paper mode. The crypto prediction markets are competitive and most strategies lose money to adverse selection and fees.
