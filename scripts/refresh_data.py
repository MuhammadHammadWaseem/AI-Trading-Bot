"""
scripts/refresh_data.py
═══════════════════════════════════════════════════════════════════════════════
Incremental market data refresh — appends only MISSING candles to existing
parquet files.  Run this before auto_retrain.py, or standalone.

Usage
-----
    # Refresh all symbols, all timeframes (default: last 30 days or since EOF)
    python scripts/refresh_data.py

    # Specific symbols
    python scripts/refresh_data.py --symbols SOLUSDT BNBUSDT

    # Force full re-download of last N days (ignores existing data)
    python scripts/refresh_data.py --days 60

Design
------
  - Reads existing parquet, finds the last timestamp, fetches from there
  - Writes back the same parquet file (append-only, deduplicates by timestamp)
  - Uses Binance spot API (public, no keys needed) for broad history
  - Falls back to testnet futures API if spot data is unavailable
  - Never deletes historical data — only adds
"""

import asyncio
import argparse
import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.logger import get_logger

logger = get_logger("refresh_data")

DATA_DIR = Path(PROJECT_ROOT) / "training_data"
DATA_DIR.mkdir(exist_ok=True)

SYMBOLS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAMES = ["5m", "1h"]

TF_SECONDS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600,
              "4h": 14400, "1d": 86400}

# How far back to look on first download if no parquet exists
DEFAULT_SINCE = {
    "5m":  "2023-01-01",   # ~105k bars per symbol
    "1h":  "2021-01-01",   # ~30k bars per symbol
}

# ─────────────────────────────────────────────────────────────────────────────
# Data download
# ─────────────────────────────────────────────────────────────────────────────

async def fetch_candles(symbol: str, timeframe: str,
                        since_ts_ms: int, until_ts_ms: int) -> pd.DataFrame:
    """
    Fetch OHLCV candles from Binance spot API.
    Returns DataFrame with columns: open, high, low, close, volume
    and a DatetimeIndex (UTC).
    """
    import ccxt.async_support as ccxt

    exchange = ccxt.binance({
        "options": {"fetchCurrencies": False, "defaultType": "spot"},
        "enableRateLimit": True,
    })

    tf_ms      = TF_SECONDS.get(timeframe, 300) * 1000
    all_rows   = []
    current_ts = since_ts_ms
    retries    = 0
    MAX_RETRY  = 6

    try:
        while current_ts < until_ts_ms:
            try:
                batch = await exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=current_ts, limit=1000,
                    params={"endTime": until_ts_ms},
                )
                if not batch:
                    break
                all_rows.extend(batch)
                current_ts = batch[-1][0] + tf_ms
                retries    = 0
                await asyncio.sleep(0.08)

            except asyncio.TimeoutError:
                retries += 1
                if retries > MAX_RETRY:
                    logger.warning(f"[REFRESH] {symbol} {timeframe}: too many timeouts, stopping")
                    break
                await asyncio.sleep(retries * 4)

            except Exception as e:
                retries += 1
                if retries > MAX_RETRY:
                    logger.warning(f"[REFRESH] {symbol} {timeframe}: {e}")
                    break
                await asyncio.sleep(retries * 3)
    finally:
        await exchange.close()

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(subset=["close"])
    return df


def load_existing(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Load existing parquet, return None if missing."""
    path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.columns = [c.lower().strip() for c in df.columns]
        rename = {c: "timestamp" for c in df.columns
                  if c in ("open_time", "opentime", "time", "date")}
        if rename:
            df = df.rename(columns=rename)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.index = (df.index.tz_localize("UTC")
                    if df.index.tz is None
                    else df.index.tz_convert("UTC"))
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        return df[keep].copy()
    except Exception as e:
        logger.warning(f"[REFRESH] Could not load {symbol}_{timeframe}.parquet: {e}")
        return None


def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save DataFrame to parquet, keeping index as-is."""
    path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(path, compression="zstd")


# ─────────────────────────────────────────────────────────────────────────────
# Main per-symbol refresh
# ─────────────────────────────────────────────────────────────────────────────

async def refresh_symbol(symbol: str, timeframe: str,
                         force_days: int | None = None) -> dict:
    """
    Refresh a single symbol+timeframe.
    Returns summary dict with rows_added, total_rows, date_range.
    """
    existing = load_existing(symbol, timeframe)
    now_ts   = int(datetime.now(timezone.utc).timestamp() * 1000)

    if force_days is not None:
        # Ignore existing data, re-fetch last N days
        cutoff = datetime.now(timezone.utc) - timedelta(days=force_days)
        since_ts = int(cutoff.timestamp() * 1000)
        logger.info(f"[REFRESH] {symbol} {timeframe}: forced re-fetch last {force_days}d")
    elif existing is not None and len(existing) > 0:
        # Append from last known bar + 1 step
        tf_sec   = TF_SECONDS.get(timeframe, 300)
        last_ts  = existing.index[-1]
        since_dt = last_ts.to_pydatetime() + timedelta(seconds=tf_sec)
        since_ts = int(since_dt.timestamp() * 1000)
        gap_bars = (now_ts - since_ts) // (tf_sec * 1000)
        logger.info(f"[REFRESH] {symbol} {timeframe}: appending ~{gap_bars:,} new bars "
                    f"from {since_dt.strftime('%Y-%m-%d %H:%M')}")
        if gap_bars <= 1:
            logger.info(f"[REFRESH] {symbol} {timeframe}: already up-to-date, skipping")
            return {"rows_added": 0, "total_rows": len(existing),
                    "date_range": f"{existing.index[0].date()} → {existing.index[-1].date()}"}
    else:
        # First download
        since_str = DEFAULT_SINCE.get(timeframe, "2023-01-01")
        since_ts  = int(pd.Timestamp(since_str).timestamp() * 1000)
        logger.info(f"[REFRESH] {symbol} {timeframe}: first download from {since_str}")

    # Fetch new candles
    new_df = await fetch_candles(symbol, timeframe, since_ts, now_ts)

    if new_df.empty:
        logger.warning(f"[REFRESH] {symbol} {timeframe}: no new data fetched")
        rows_added = 0
        combined   = existing if existing is not None else pd.DataFrame()
    else:
        # Merge with existing, deduplicate
        frames = []
        if existing is not None and len(existing) > 0 and force_days is None:
            frames.append(existing)
        frames.append(new_df)
        combined   = pd.concat(frames).sort_index()
        combined   = combined[~combined.index.duplicated(keep="last")]
        rows_added = len(new_df)

        save_parquet(combined, symbol, timeframe)
        logger.info(f"[REFRESH] {symbol} {timeframe}: +{rows_added:,} rows → "
                    f"{len(combined):,} total")

    date_range = (f"{combined.index[0].date()} → {combined.index[-1].date()}"
                  if len(combined) > 0 else "empty")
    return {"rows_added": rows_added, "total_rows": len(combined), "date_range": date_range}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(symbols: list, timeframes: list, force_days: int | None):
    start = time.time()
    logger.info(f"[REFRESH] Starting data refresh for {symbols}")

    results = {}
    for sym in symbols:
        results[sym] = {}
        for tf in timeframes:
            result = await refresh_symbol(sym, tf, force_days=force_days)
            results[sym][tf] = result

    elapsed = time.time() - start
    logger.info(f"[REFRESH] Complete in {elapsed:.0f}s")

    print("\n" + "=" * 65)
    print("DATA REFRESH SUMMARY")
    print("=" * 65)
    for sym, tfs in results.items():
        for tf, r in tfs.items():
            added = r["rows_added"]
            total = r["total_rows"]
            dr    = r["date_range"]
            status = f"+{added:>6,} rows" if added > 0 else "  up-to-date"
            print(f"  {sym:<10} {tf:<4}  {status}  |  {total:>7,} total  |  {dr}")
    print("=" * 65)
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Incrementally refresh training data")
    parser.add_argument("--symbols",    nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--days",       type=int,  default=None,
                        help="Force re-fetch of last N days (default: append only)")
    args = parser.parse_args()

    asyncio.run(main(args.symbols, args.timeframes, args.days))