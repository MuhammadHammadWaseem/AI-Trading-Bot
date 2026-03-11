"""
training/trainer.py — No TensorFlow version.
Uses scikit-learn ensemble model.
"""

import asyncio, argparse, sys, os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exchange.exchange_factory import create_exchange
from core.models.ml_model import MLModel
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.settings import settings


async def fetch_data(symbol: str, candle_limit: int) -> list:
    exchange = create_exchange("binance")
    await exchange.connect()
    print(f"  >> Fetching {candle_limit} candles for {symbol}...")
    candles = await exchange.get_ohlcv(symbol, "15m", limit=candle_limit)
    await exchange._exchange.close()
    print(f"  >> Got {len(candles)} candles")
    return candles


def main(symbols: list, candle_limit: int):
    print("=" * 60)
    print("  AI Model Training (GradientBoosting + RandomForest)")
    print("=" * 60)

    results = {}
    for symbol in symbols:
        print(f"\n{'='*20} {symbol} {'='*20}")
        try:
            candles = asyncio.run(fetch_data(symbol, candle_limit))
        except Exception as e:
            print(f"  >> FETCH ERROR: {e}")
            results[symbol] = "FAILED (fetch)"
            continue

        try:
            df    = ohlcv_to_dataframe(candles)
            df    = add_all_indicators(df)
            model = MLModel(symbol=symbol)
            model.train(df)
            results[symbol] = "SUCCESS"
        except Exception as e:
            print(f"  >> TRAIN ERROR: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()
            results[symbol] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    for sym, status in results.items():
        print(f"  {sym}: {status}")
    print(f"\n  Saved to: {settings.model.saved_models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=settings.trading_pairs)
    parser.add_argument("--candles", type=int, default=1500)
    args = parser.parse_args()
    main(args.symbols, args.candles)