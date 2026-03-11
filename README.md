# 🤖 AI Futures Trading Bot

Scalable, professional-grade AI trading bot for crypto futures.  
**Hybrid model**: Technical Analysis (RSI, MACD, EMA, BB, ADX) + LSTM neural network.

---

## 📁 Project Structure

```
trading-bot/
├── config/              # Settings, logging
├── core/
│   ├── exchange/        # Exchange connectors (Binance + future exchanges)
│   ├── models/          # Technical, ML (LSTM), Hybrid AI models
│   ├── strategy/        # Loss recovery strategy
│   ├── risk/            # Risk & position management
│   └── trader/          # Trade orchestrator per symbol
├── data/                # Data fetching & indicators
├── training/            # ML training & backtesting
├── scripts/             # Entry points
└── saved_models/        # Trained model weights (auto-generated)
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env and add your Binance Testnet API keys
```

Get testnet API keys from: **https://testnet.binancefuture.com**

### 3. Verify connection
```bash
python -c "
import asyncio, sys
sys.path.insert(0, '.')
from core.exchange.exchange_factory import create_exchange
async def test():
    ex = create_exchange('binance')
    ok = await ex.connect()
    if ok:
        bal = await ex.get_balance()
        print(f'Balance: {bal.total_balance} USDT')
    await ex._exchange.close()
asyncio.run(test())
"
```

---

## 🚀 Usage

### Run the bot (Technical model, no ML training yet)
```bash
python scripts/run_bot.py
```

### Run with specific pairs
```bash
python scripts/run_bot.py --pairs BTCUSDT ETHUSDT
```

### Train ML models first, then trade
```bash
# Step 1: Train
python training/trainer.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT --candles 1500

# Step 2: Run bot with trained models
python scripts/run_bot.py --train
```

### Backtest before going live
```bash
python training/backtester.py --symbol BTCUSDT --candles 500 --tp 2.0 --sl 1.0
```

### Custom risk settings
```bash
python scripts/run_bot.py \
  --leverage 10 \
  --tp 2.5 \
  --sl 1.2 \
  --risk 1.0 \
  --interval 60
```

---

## 📊 Command Reference

| Command | Description |
|---------|-------------|
| `--pairs` | Symbols to trade (e.g. `BTCUSDT ETHUSDT`) |
| `--train` | Train ML models on startup |
| `--leverage` | Futures leverage (default: 10) |
| `--tp` | Take profit % (default: 2.0) |
| `--sl` | Stop loss % (default: 1.0) |
| `--risk` | Risk per trade % of balance (default: 1.0) |
| `--interval` | Seconds between cycles (default: 60) |

---

## 🧠 How the AI Works

```
Market Data (OHLCV 15m)
        ↓
Technical Indicators (RSI, MACD, EMA, BB, ADX, Stoch, ATR...)
        ↓
   ┌────────────┬─────────────┐
   │  Technical │  LSTM Model │  ← 40% / 60% weight
   │   Model    │  (trained)  │
   └─────┬──────┴──────┬──────┘
         └──────┬───────┘
           Hybrid Model
                ↓
        LONG / SHORT / HOLD
        + Confidence Score
                ↓
         Risk Manager
      (position size, TP/SL)
                ↓
         Execute Trade
```

### Hybrid Weighting
- **Technical model only** — when ML not yet trained
- **40% Technical + 60% ML** — when ML is trained
- **Agreement bonus** (+10% confidence) — when both models agree
- **Disagreement penalty** (×0.75 confidence) — when models disagree
- **Confidence gate** — if below 65%, force HOLD

---

## 🔄 Loss Recovery Strategy

When a trade closes at a loss:
1. Recovery state is activated for that symbol
2. Next recovery trade opens only when signal confidence > 70%
3. Position size multiplied: 1x → 1.5x → 2.25x (max 3x)
4. Max 3 recovery attempts per loss event
5. If all 3 fail → 60-minute cooldown for that symbol

---

## 🔒 Risk Management

- **Per-trade risk**: % of available balance (default 1%)
- **Max open trades**: configurable (default 5)
- **Daily loss limit**: % of starting balance (default 5%)
- **Leverage cap**: configurable (default 10x)
- **Automatic TP/SL**: attached to every trade at order level

---

## 📈 Phase Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| 1 | ✅ **Current** | Terminal bot, Binance testnet, Hybrid AI |
| 2 | ⏳ | Model retraining scheduler, performance monitoring |
| 3 | ⏳ | Laravel API integration, multi-user support |
| 4 | ⏳ | Add OKX, Bybit, other exchanges |
| 5 | ⏳ | Web dashboard, Celery workers, production deploy |

---

## 🗂️ Adding a New Exchange (Future)

1. Create `core/exchange/okx_exchange.py` implementing `BaseExchange`
2. Add to `core/exchange/exchange_factory.py`:
   ```python
   from core.exchange.okx_exchange import OKXExchange
   EXCHANGE_MAP = {
       "binance": BinanceExchange,
       "okx":     OKXExchange,   # ← add here
   }
   ```
3. Add credentials to `.env`
4. Done — no other code changes needed

---

## ⚠️ Important Notes

- **NEVER use live keys during testing** — always use testnet first
- The bot never commits `.env` to git (add to `.gitignore`)
- Losses are possible even with AI — always test with small amounts
- Past backtest performance does not guarantee future results
