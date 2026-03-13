# 🤖 AI Futures Trading Bot

Scalable, professional-grade AI trading bot for crypto futures.  
**Hybrid model**: Technical Analysis (RSI, MACD, EMA, BB, ADX) + LSTM neural network.

---

## 📁 Project Structure

```
trading-bot/
├── api/                 # API endpoints & routing (Prepared for Phase 3)
├── config/              # Configuration & settings, logger
├── core/
│   ├── exchange/        # Exchange connectors (Binance + future exchanges)
│   ├── models/          # Technical, ML (LSTM), Hybrid AI models
│   ├── strategy/        # Loss recovery strategy
│   ├── risk/            # Risk & position management
│   └── trader/          # Trade orchestrator per symbol
├── data/                # Data fetching & technical indicators
├── database/            # Database schemas & migrations (Prepared for Phase 3)
├── training/            # ML training & backtesting
├── scripts/             # Entry points & runners
├── workers/             # Background task workers (Prepared for Phase 5)
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

## 🧠 Internal Architecture & AI Pipeline

The bot's decision engine is split into three main components: a **Technical Model**, a **Machine Learning Model**, and a **Hybrid Orchestrator** that bridges the two. 

### 1. The Data Pipeline
Market data is streamed at 15m intervals (converting OHLCV). The system then computes baseline technical indicators (RSI, MACD, EMA 9/21/50, Bollinger Bands, Stochastic K/D, ATR, ADX, and Volume Ratios).

### 2. The Technical Model (Rule-Based Scoring)
The `TechnicalModel` evaluates the latest candle against 7 distinct indicator groups. It works instantly (zero training required) and assigns a score from `-1.0` (Strong SHORT) to `+1.0` (Strong LONG).
* **RSI:** Penalizes extreme readings (<30 or >70).
* **MACD:** Looks for structural crossovers & histogram shifts.
* **EMA Trend:** Scores alignment between Price, EMA 9, EMA 21, and EMA 50.
* **Bollinger Bands:** Counters price action at the extreme bands.
* **Stochastic & ADX:** Confirms momentum and dampens signals in weak (ranging) markets.
* **Volume Profiling:** Amplifies signals on volume spikes (>1.5x ratio) and penalizes low engagement.

### 3. The ML Model (LightGBM + XGBoost Ensemble)
The `MLModel` is a production-grade predictive engine designed to classify the next significant price movement.
* **Feature Engineering:** Constructs **80+ features**, including price action wicks, rolling statistics, multi-period return percentages, time-based features, and volatility spikes.
* **Label Generation:** It looks forward 4 candles. If the return is `> 0.4%`, it labels `LONG`. If `< -0.4%`, it labels `SHORT`. Otherwise, `HOLD`.
* **Ensemble Soft-Voting:** By default, it trains both `LightGBM` and `XGBoost` classifiers on a walk-forward validation split. Predicts probabilities, weights them, and arrives at a consensus. *(Falls back to Sklearn GradientBoosting & Random Forests if booster libs are omitted).*

### 4. The Hybrid Orchestrator
The `HybridModel` is where Technical and ML predictions meet.
* **Base Weights:** Technical Analysis has a `40%` baseline weight. The trained ML model has a `60%` baseline weight.
* **Agreement Bonus (+10%):** If both the Technical and ML models agree on LONG or SHORT, the hybrid confidence score receives an artificial `+10%` boost.
* **Disagreement Penalty (×0.75):** If the models clash, the signal's confidence is drastically reduced.
* **The Confidence Gate (>65%):** The final signal requires a mathematically combined confidence of strictly **>65%**. If it falls below this gate, the bot forces a **HOLD**, refusing to enter a trade in choppy conditions.

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
| 3 | ⏳ | Python/Laravel API integration, multi-user support |
| 4 | ⏳ | Add OKX, Bybit, other exchanges |
| 5 | ⏳ | Web dashboard, Celery workers, production deploy |

> **Note:** Core skeleton directories for the future Python backend API (`api/`, `database/`, `workers/`) are already initialized.

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
