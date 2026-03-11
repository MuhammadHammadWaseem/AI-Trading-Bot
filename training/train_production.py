"""
training/train_production.py
------------------------------
Production training pipeline:
1. Load ALL downloaded historical data
2. Train on millions of candles
3. Walk-forward validation (proper time-series evaluation)
4. Save production-ready models

Usage:
    # First download data:
    python training/download_data.py

    # Then train:
    python training/train_production.py
    python training/train_production.py --symbols BTCUSDT ETHUSDT
"""

import asyncio
import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models.ml_model import MLModel
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.settings import settings

DATA_DIR = Path("training_data")


def load_and_merge_timeframes(symbol: str, timeframes: list) -> pd.DataFrame:
    """
    Load multiple timeframes and merge features.
    Primary timeframe: 15m
    Additional context: 1h and 4h indicators added as features.
    """
    sym_clean = symbol.replace("/", "")
    dfs = {}

    for tf in timeframes:
        path = DATA_DIR / f"{sym_clean}_{tf}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            df = add_all_indicators(df)
            dfs[tf] = df
            print(f"  >> Loaded {sym_clean} {tf}: {len(df):,} candles")
        else:
            print(f"  >> WARNING: {path} not found — run download_data.py first")

    if "15m" not in dfs:
        # Fallback: use whatever is available
        if not dfs:
            raise FileNotFoundError(
                f"No data found for {symbol}. Run: python training/download_data.py"
            )
        primary_tf = list(dfs.keys())[0]
    else:
        primary_tf = "15m"

    primary = dfs[primary_tf].copy()

    # Add higher timeframe features as extra columns
    for tf, df in dfs.items():
        if tf == primary_tf:
            continue
        suffix = f"_{tf}"
        cols_to_add = ["rsi", "macd", "adx", "ema_20", "ema_50", "bb_upper", "bb_lower", "atr"]
        for col in cols_to_add:
            if col in df.columns:
                # Reindex to primary timeframe (forward fill)
                reindexed = df[col].reindex(primary.index, method="ffill")
                primary[col + suffix] = reindexed

    print(f"  >> Merged: {len(primary):,} candles | {len(primary.columns)} columns")
    return primary


def train_symbol_production(symbol: str, timeframes: list) -> float:
    """Train production model for one symbol."""
    print(f"\n{'='*20} {symbol} {'='*20}")

    try:
        df = load_and_merge_timeframes(symbol, timeframes)
    except FileNotFoundError as e:
        print(f"  >> ERROR: {e}")
        return 0.0

    print(f"  >> Total candles for training: {len(df):,}")
    print(f"  >> Date range: {df.index[0].date()} to {df.index[-1].date()}")

    model   = MLModel(symbol=symbol)
    history = model.train(df)

    return history.history["val_accuracy"][-1]


def main(symbols: list, timeframes: list):
    print("=" * 60)
    print("  Production AI Model Training")
    print("=" * 60)
    print(f"  Symbols    : {symbols}")
    print(f"  Timeframes : {timeframes}")
    print(f"  Data dir   : {DATA_DIR.absolute()}")
    print()

    if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
        print("  ERROR: No training data found!")
        print("  Please run first:")
        print("    python training/download_data.py")
        return

    results = {}
    for symbol in symbols:
        try:
            acc = train_symbol_production(symbol, timeframes)
            results[symbol] = f"SUCCESS (accuracy={acc:.2%})"
        except Exception as e:
            print(f"  >> ERROR {symbol}: {e}")
            import traceback; traceback.print_exc()
            results[symbol] = f"FAILED: {e}"

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    for sym, status in results.items():
        print(f"  {sym}: {status}")
    print(f"\n  Models saved to: {settings.model.saved_models_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+",
                        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
    parser.add_argument("--timeframes", nargs="+",
                        default=["15m", "1h", "4h"])
    args = parser.parse_args()
    main(args.symbols, args.timeframes)