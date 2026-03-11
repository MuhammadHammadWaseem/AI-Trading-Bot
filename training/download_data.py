"""
training/download_data.py
--------------------------
Download maximum historical OHLCV data from Binance.
Uses spot API (binance) instead of futures for broader history.

Usage:
    python training/download_data.py
    python training/download_data.py --symbols BTCUSDT ETHUSDT --timeframes 15m 1h 4h --since 2020-01-01
"""

import asyncio
import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path("training_data")
DATA_DIR.mkdir(exist_ok=True)

TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}


async def download_ohlcv(symbol: str, timeframe: str, since_date: str) -> pd.DataFrame:
    """Download all candles using ccxt binance (spot — no load_markets needed)."""
    import ccxt.async_support as ccxt

    # Use spot Binance — more history, no futures overhead
    exchange = ccxt.binance({
        "options": {
            "fetchCurrencies": False,
            "defaultType": "spot",
        },
        "enableRateLimit": True,
    })

    since_ts  = int(pd.Timestamp(since_date).timestamp() * 1000)
    now_ts    = int(datetime.now().timestamp() * 1000)
    candle_ms = TF_MS.get(timeframe, 900_000)
    total_est = (now_ts - since_ts) // candle_ms

    print(f"  >> {symbol} {timeframe}: ~{total_est:,} candles from {since_date}")

    all_candles = []
    current_ts  = since_ts
    batch_num   = 0
    retries     = 0

    while current_ts < now_ts:
        try:
            batch = await exchange.fetch_ohlcv(
                symbol, timeframe,
                since=current_ts,
                limit=1000,
                params={"endTime": now_ts}
            )

            if not batch:
                break

            all_candles.extend(batch)
            current_ts = batch[-1][0] + candle_ms
            batch_num += 1
            retries = 0

            if batch_num % 20 == 0:
                pct = min((current_ts - since_ts) / (now_ts - since_ts) * 100, 100)
                dt  = pd.Timestamp(current_ts, unit="ms").strftime("%Y-%m")
                print(f"  >> {symbol} {timeframe}: {len(all_candles):,} candles | {dt} ({pct:.0f}%)")

            await asyncio.sleep(0.05)

        except asyncio.TimeoutError:
            retries += 1
            if retries > 5:
                print(f"  >> Too many timeouts, stopping at {len(all_candles):,} candles")
                break
            wait = retries * 3
            print(f"  >> Timeout — retry {retries}/5 in {wait}s...")
            await asyncio.sleep(wait)

        except Exception as e:
            retries += 1
            if retries > 5:
                print(f"  >> Too many errors: {e}")
                break
            print(f"  >> Error: {e} — retry {retries}/5...")
            await asyncio.sleep(retries * 2)

    await exchange.close()

    if not all_candles:
        print(f"  >> WARNING: No candles downloaded for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = df.astype(float)

    print(f"  >> {symbol} {timeframe}: DONE — {len(df):,} candles | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Download historical trading data")
    parser.add_argument("--symbols",    nargs="+", default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"])
    parser.add_argument("--timeframes", nargs="+", default=["15m", "1h", "4h"])
    parser.add_argument("--since",      default="2020-01-01")
    args = parser.parse_args()

    print("=" * 60)
    print("  Historical Data Downloader")
    print("=" * 60)
    print(f"  Symbols    : {args.symbols}")
    print(f"  Timeframes : {args.timeframes}")
    print(f"  Since      : {args.since}")
    print(f"  Save dir   : {DATA_DIR.absolute()}")
    print()

    total_downloaded = 0

    for symbol in args.symbols:
        sym_clean = symbol.replace("/", "")
        for tf in args.timeframes:
            save_path = DATA_DIR / f"{sym_clean}_{tf}.parquet"

            # Resume if already partially downloaded
            if save_path.exists():
                existing  = pd.read_parquet(save_path)
                last_date = existing.index[-1].date()
                today     = datetime.now().date()
                days_old  = (today - last_date).days

                if days_old <= 1:
                    print(f"  >> {sym_clean} {tf}: Up to date ({len(existing):,} candles) — skipping")
                    total_downloaded += len(existing)
                    continue
                else:
                    resume_from = (existing.index[-1] - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    print(f"  >> {sym_clean} {tf}: Resuming from {resume_from}...")
                    new_df = asyncio.run(download_ohlcv(symbol, tf, since_date=resume_from))
                    if not new_df.empty:
                        combined = pd.concat([existing, new_df])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        combined.to_parquet(save_path)
                        total_downloaded += len(combined)
                        size_mb = save_path.stat().st_size / 1024 / 1024
                        print(f"  >> Saved: {save_path.name} ({len(combined):,} candles, {size_mb:.1f} MB)")
                    continue

            # Fresh download
            df = asyncio.run(download_ohlcv(symbol, tf, since_date=args.since))

            if not df.empty:
                df.to_parquet(save_path)
                total_downloaded += len(df)
                size_mb = save_path.stat().st_size / 1024 / 1024
                print(f"  >> Saved: {save_path.name} ({len(df):,} candles, {size_mb:.1f} MB)\n")

    print("=" * 60)
    print(f"  DOWNLOAD COMPLETE")
    print(f"  Total candles: {total_downloaded:,}")
    print(f"  Files in: {DATA_DIR.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()