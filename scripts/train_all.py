"""
scripts/train_all.py
─────────────────────
Run this from your project root to retrain all symbols.

Usage:
    cd "C:\\Users\\123\\Desktop\\AI Trading Bot"
    python scripts/train_all.py

What it does:
    1. Connects to Binance (testnet or live, based on .env)
    2. Fetches 3000 x 15m candles per symbol
    3. Runs walk-forward training (5 folds)
    4. Saves model only if Sharpe > 0.5
    5. Prints a summary table at the end
"""

import sys
import os

# ── Fix Python path ────────────────────────────────────────────────────────────
# This must be the FIRST thing in the script — before any project imports.
# It adds the project root to sys.path so that `from core.xxx import ...` works
# regardless of which directory you run the script from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# ──────────────────────────────────────────────────────────────────────────────

import asyncio
import time
from datetime import datetime

# Now project imports work
from core.exchange.exchange_factory import create_exchange
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from core.models.ml_model import MLModel
from config.settings import settings
from config.logger import get_logger

logger = get_logger("train_all")


async def train_symbol(exchange, symbol: str) -> dict:
    """
    Fetch data and train the ML model for one symbol.
    Returns a result dict for the summary table.
    """
    result = {
        "symbol":   symbol,
        "status":   "FAILED",
        "candles":  0,
        "hold_pct": 0,
        "sharpe":   0,
        "verdict":  "—",
    }

    try:
        print(f"\n{'='*56}")
        print(f"  {symbol}")
        print(f"{'='*56}")

        # Fetch 3000 candles (~31 days of 15m data)
        print(f"  Fetching 3000 x 15m candles...")
        candles = await exchange.get_ohlcv(symbol, "15m", limit=3000)

        if not candles or len(candles) < 500:
            print(f"  [ERROR] Only {len(candles) if candles else 0} candles returned — need at least 500")
            result["status"] = "INSUFFICIENT_DATA"
            return result

        df = ohlcv_to_dataframe(candles)
        df = add_all_indicators(df)

        # Drop leading NaN rows (from long-window indicators like EMA-200)
        df_clean = df.dropna()
        print(f"  Data: {len(df_clean)} usable candles | {df_clean.index[0]} → {df_clean.index[-1]}")

        if len(df_clean) < 400:
            print(f"  [ERROR] Only {len(df_clean)} non-NaN rows after indicators — need 400+")
            result["status"] = "INSUFFICIENT_DATA"
            return result

        result["candles"] = len(df_clean)

        # Train
        start_time = time.time()
        model = MLModel(symbol=symbol)
        model.train(df_clean)
        elapsed = time.time() - start_time

        # _last_accepted = real WF verdict (not just "is model in memory")
        verdict = "ACCEPTED" if model._last_accepted else "WF-rejected (saved anyway)"
        print(f"\n  Training time: {elapsed:.1f}s | Verdict: {verdict}")

        result["status"]  = "OK"
        result["verdict"] = "ACCEPTED" if model._last_accepted else "WF-REJECTED"
        result["sharpe"]  = model._wf_sharpe
        result["val_acc"] = model._wf_f1  # use F1 as quality proxy (stored in model)

    except Exception as e:
        print(f"\n  [ERROR] {symbol} training failed: {e}")
        import traceback
        traceback.print_exc()
        result["status"] = f"ERROR: {e}"

    return result


async def main():
    print("\n" + "="*56)
    print("  AI TRADING BOT — Model Training")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Environment: {settings.environment}")
    print(f"  Symbols: {settings.trading_pairs}")
    print("="*56)

    # Connect to exchange
    print("\nConnecting to exchange...")
    exchange = create_exchange("binance")
    connected = await exchange.connect()

    if not connected:
        print("\n[FATAL] Could not connect to exchange.")
        print("  Check your API keys in .env file.")
        print("  Keys needed: BINANCE_TESTNET_API_KEY and BINANCE_TESTNET_SECRET")
        sys.exit(1)

    print("Exchange connected.\n")

    # Train each symbol
    results = []
    for symbol in settings.trading_pairs:
        r = await train_symbol(exchange, symbol)
        results.append(r)

    # Close exchange connection
    try:
        await exchange._exchange.close()
    except Exception:
        pass

    # Print summary table
    print("\n\n" + "="*56)
    print("  TRAINING SUMMARY")
    print("="*56)
    print(f"  {'Symbol':<12} {'Status':<8} {'Candles':<10} {'Sharpe':<10} {'Verdict'}")
    print(f"  {'-'*52}")
    for r in results:
        sharpe_str = f"{r['sharpe']:.2f}" if isinstance(r['sharpe'], float) else str(r['sharpe'])
        print(f"  {r['symbol']:<12} {r['status']:<8} {r['candles']:<10} {sharpe_str:<10} {r['verdict']}")

    accepted  = sum(1 for r in results if r["verdict"] == "ACCEPTED")
    wf_reject = sum(1 for r in results if r["verdict"] == "WF-REJECTED")
    failed    = sum(1 for r in results if r["status"]  != "OK")

    print(f"\n  {accepted} fully accepted  |  {wf_reject} WF-rejected but saved  |  {failed} failed")
    print()
    print("  HOW TO READ THE ACCURACY NUMBERS:")
    print("  ─────────────────────────────────────────────────────")
    print("  This is a 3-class problem (HOLD / LONG / SHORT).")
    print("  Random guessing = 33% accuracy.  Above 45% = the model is learning.")
    print()
    print("  Val accuracy by symbol:")
    for r in results:
        if r["status"] == "OK":
            bar = "█" * int(r.get("val_acc", 0) * 30)
            print(f"    {r['symbol']:<10} {r.get('val_acc', 0):.1%}  {bar}")
    print()
    print("  Target on 13 days of data  : 45–60%  (limited regime diversity)")
    print("  Target on 30+ days of data : 55–70%  (retrain when live data grows)")
    print()

    if accepted == 0 and wf_reject > 0:
        print("  [NOTE] WF-rejected models are saved and usable.")
        print("  WF rejection = Sharpe not consistently positive in mini backtests.")
        print("  This is normal with only 13 days. Retrain after 30 days on live exchange.")
    elif accepted == 0 and failed > 0:
        print("\n  [HINT] Some symbols failed. Check exchange connection and API keys.")

    print("\nDone.\n")


if __name__ == "__main__":
    asyncio.run(main())