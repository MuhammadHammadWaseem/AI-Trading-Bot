# AI Trading Bot — Upgrade Integration Guide

## Files Created / Modified

### SECTION 1 — New Files (copy-paste into your project)

| File | Purpose |
|------|---------|
| `data/indicators.py` | **CRITICAL** — missing file that ml_model.py imports. Self-contained technical indicators (RSI, MACD, EMA, ATR, BB, ADX, Stochastic, OBV, VWAP, volatility, candle patterns) |
| `data/__init__.py` | Package init |
| `models/training/label_engine.py` | Volatility-adaptive label generation (ATR triple-barrier, vol-adaptive, quantile methods) |
| `models/training/trainer.py` | Full walk-forward training pipeline with class balancing and calibration |
| `models/training/__init__.py` | Package init |
| `research/walk_forward.py` | Walk-forward validation engine (replaces 80/20 split) |
| `research/__init__.py` | Package init |
| `backtesting/engine.py` | Full backtesting engine (fees, slippage, ATR TP/SL, equity curve) |
| `backtesting/metrics.py` | Sharpe, Sortino, Calmar, profit factor, win rate, drawdown, strategy grade |
| `backtesting/portfolio.py` | Multi-symbol portfolio backtest with correlation analysis |
| `backtesting/__init__.py` | Package init |
| `risk_management/position_sizer.py` | Kelly Criterion position sizing (replaces Martingale) |
| `risk_management/__init__.py` | Package init |

### SECTION 2 — Modified Files

| File | Changes |
|------|---------|
| `core/models/ml_model.py` | Probability-based signals (P>0.38 not argmax), adaptive thresholds, delegates training to ModelTrainer |
| `core/models/hybrid_model.py` | **REMOVED** 0.75× disagreement penalty, lowered confidence gate to 0.50, probability-based blending |
| `core/risk/risk_manager.py` | Kelly-based position sizing, ATR TP/SL, confidence-weighted sizing |
| `core/trader/futures_trader.py` | Passes ATR + confidence to risk manager, removed RecoveryStrategy |

### SECTION 3 — Files to KEEP unchanged

- `config/settings.py` — no changes needed
- `config/logger.py` — no changes needed
- `core/exchange/` — no changes needed
- `core/models/base_model.py` — no changes needed
- `core/models/technical_model.py` — no changes needed

### SECTION 4 — Files that can be DELETED

- `core/strategy/recovery_strategy.py` — **DELETE** (Martingale removed, replaced by Kelly)

---

## SECTION 3 — How to Integrate the Changes

### Step 1: Install missing dependency

```bash
pip install ta scikit-learn lightgbm xgboost imbalanced-learn joblib
```

### Step 2: Create directory structure

```bash
mkdir -p data features models/training models/inference research \
         backtesting risk_management execution strategy
touch data/__init__.py models/__init__.py models/training/__init__.py \
      models/inference/__init__.py research/__init__.py \
      backtesting/__init__.py risk_management/__init__.py
```

### Step 3: Copy all new files

Copy all files from SECTION 1 into your project root.

### Step 4: Replace modified files

Replace the 4 modified files from SECTION 2.

### Step 5: Update requirements.txt

Add these lines:
```
imbalanced-learn==0.12.3
```

### Step 6: Test your indicators

```python
# Quick sanity check
import pandas as pd
from data.indicators import add_all_indicators, get_feature_columns

# Load any OHLCV data
df = pd.read_csv("your_data.csv", index_col=0, parse_dates=True)
df = add_all_indicators(df)
print(f"Columns: {len(df.columns)}")
print(f"Feature columns: {len(get_feature_columns())}")
```

### Step 7: Test label distribution

```python
from models.training.label_engine import generate_labels, label_distribution

labels = generate_labels(df, method="triple_barrier", future_bars=8, atr_mult=1.5)
dist   = label_distribution(labels)
print(dist)
# Target: HOLD ~25-40%, LONG ~30-38%, SHORT ~30-38%
# If HOLD > 60% → increase atr_mult to 2.0
# If HOLD < 15% → decrease atr_mult to 1.0
```

### Step 8: Run a backtest

```python
from backtesting.engine import BacktestEngine
from backtesting.metrics import compute_metrics, print_report, strategy_grade

# df must have 'signal' column (0=HOLD, 1=LONG, 2=SHORT) and 'confidence' column
engine = BacktestEngine(
    initial_capital = 10_000,
    leverage        = 10,
    risk_per_trade  = 0.01,
    taker_fee       = 0.0004,
    slippage        = 0.0002,
)
result  = engine.run(df_with_signals)
metrics = compute_metrics(result.equity_curve, result.trades)
print_report(metrics)
print(f"Grade: {strategy_grade(metrics)}")
# Accept only Grade B or better (Sharpe > 1.0)
```

### Step 9: Train the ML model

```python
from core.models.ml_model import MLModel
from data.indicators import add_all_indicators, ohlcv_to_dataframe

# Fetch at least 2000 candles for reliable walk-forward training
candles = await exchange.get_ohlcv("BTCUSDT", "15m", limit=3000)
df      = ohlcv_to_dataframe(candles)
df      = add_all_indicators(df)

model = MLModel(symbol="BTCUSDT")
model.train(df)
# Walk-forward validation runs automatically
# Model is saved only if Sharpe > 0.5 across folds
```

### Step 10: Tune the ATR multiplier per symbol

Different assets need different ATR multipliers for balanced labels:

| Symbol | Recommended atr_mult | Reason |
|--------|---------------------|--------|
| BTCUSDT | 1.5 | Moderate volatility |
| ETHUSDT | 1.8 | Higher vol than BTC |
| SOLUSDT | 2.0 | High volatility altcoin |
| BNBUSDT | 1.5 | Similar to BTC |

Override in `models/training/trainer.py`:
```python
class ModelTrainer:
    ATR_MULT = 1.5   # change per symbol
```

---

## Key Parameter Summary

### HybridModel thresholds
```python
LONG_THRESHOLD  = 0.42   # P(LONG) must exceed this to emit LONG signal
SHORT_THRESHOLD = 0.42   # P(SHORT) must exceed this to emit SHORT signal
MIN_CONFIDENCE  = 0.50   # minimum confidence to avoid HOLD gate
```

### MLModel adaptive thresholds
```python
BASE_LONG_THRESHOLD  = 0.38
VOL_THRESHOLD_BOOST  = 0.08   # added in high-vol (vol_pct > 70th percentile)
VOL_THRESHOLD_CUT    = 0.04   # subtracted in low-vol (vol_pct < 30th percentile)
```

### PositionSizer
```python
method        = "half_kelly"  # Options: fixed_fractional, kelly, half_kelly, vol_scaled
max_risk_pct  = 0.015         # Hard cap: 1.5% of account per trade
```

### BacktestEngine acceptance criteria
- Grade A or B: Sharpe > 1.0, MaxDD < 25%, WinRate > 50%
- Minimum: Sharpe > 0.5, ProfitFactor > 1.2

---

## What was removed

- `RecoveryStrategy` (Martingale multiplier 1.5× after losses) — **deleted**
- Fixed 0.4% label threshold — **replaced** with ATR-adaptive triple-barrier
- 0.75× disagreement penalty in HybridModel — **removed**
- Static 0.65 confidence gate — **lowered to 0.50**
- Naive 80/20 train/test split — **replaced** with 5-fold walk-forward
- Uncalibrated predict_proba — **wrapped** with CalibratedClassifierCV
- XGBoost missing class weights — **fixed** with sample_weight parameter
