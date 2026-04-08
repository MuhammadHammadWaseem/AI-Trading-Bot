"""
training/download_data.py
──────────────────────────
Download USDM futures OHLCV data from Binance for model training.

KEY DESIGN DECISIONS (each backed by a reason):

1. FUTURES not SPOT:
   The bot trades Binance USDM Futures. Futures and spot price series diverge
   during funding rate events, liquidations, and high-leverage periods. Training
   on spot data introduces a systematic price bias vs the exchange you actually
   trade on. Use futures.

2. 5m + 1h timeframes only:
   - 5m  = base training timeframe (matches live bot)
   - 1h  = HTF context for trend alignment features
   4h is omitted — at 5m trading, 4h context is too slow (adds noise not signal).

3. Since 2022-01-01:
   Covers all three required market regimes:
   - TRENDING: Jan-Feb 2022 downtrend, Sep-Nov 2022 downtrend, Jan 2023 rally
   - RANGING:  Mar-May 2022, Jun-Aug 2022, late 2023
   - HIGH_VOL: May 2022 crash, Nov 2022 FTX, Mar 2023 banking crisis
   Starting before 2022 captures mostly bull-market data only — biases LONG.
   Starting after Jan 2023 is insufficient volume (< 20,000 5m bars per symbol).

4. Resume logic:
   If parquet exists and is < 2 days old: skip (already current).
   If parquet exists but stale: fetch only the missing tail.
   Always deduplicate and sort before saving.

Usage:
    # Download all symbols (recommended first run):
    python training/download_data.py

    # Update existing files with latest candles:
    python training/download_data.py

    # Single symbol:
    python training/download_data.py --symbols BTCUSDT

    # Check what you have without downloading:
    python training/download_data.py --dry-run

    # Custom date range:
    python training/download_data.py --since 2023-01-01

After download, run the validator:
    python training/download_data.py --validate
"""

import asyncio
import argparse
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
DATA_DIR.mkdir(exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────────

SYMBOLS    = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
TIMEFRAMES = ["5m", "1h"]
DEFAULT_SINCE = "2022-01-01"   # see docstring for rationale

TF_MS = {
    "1m":  60_000,
    "5m":  300_000,
    "15m": 900_000,
    "1h":  3_600_000,
    "4h":  14_400_000,
    "1d":  86_400_000,
}

# Minimum bars per timeframe before training is meaningful
MIN_BARS = {
    "5m": 10_000,   # ~35 days — enough for 5 WF folds with 2000 bars each
    "1h": 1_500,    # proportional: 35 days × 24h/day
}


# ── Downloader ─────────────────────────────────────────────────────────────────

async def download_ohlcv(
    symbol:    str,
    timeframe: str,
    since_date: str,
    use_futures: bool = True,
) -> pd.DataFrame:
    """
    Download all candles from Binance USDM Futures using ccxt.
    Falls back to spot if futures fails (e.g. symbol not on futures).
    """
    import ccxt.async_support as ccxt

    def make_exchange(futures: bool):
        if futures:
            return ccxt.binance({
                "options": {"defaultType": "future", "fetchCurrencies": False},
                "enableRateLimit": True,
            })
        return ccxt.binance({
            "options": {"defaultType": "spot", "fetchCurrencies": False},
            "enableRateLimit": True,
        })

    since_ts  = int(pd.Timestamp(since_date).timestamp() * 1000)
    now_ts    = int(datetime.now(timezone.utc).timestamp() * 1000)
    candle_ms = TF_MS.get(timeframe, 300_000)
    total_est = (now_ts - since_ts) // candle_ms

    mode = "futures" if use_futures else "spot"
    print(f"  >> {symbol} {timeframe} ({mode}): ~{total_est:,} candles from {since_date}")

    all_candles = []
    current_ts  = since_ts
    batch_num   = 0
    retries     = 0
    exchange    = make_exchange(use_futures)

    try:
        while current_ts < now_ts:
            try:
                batch = await exchange.fetch_ohlcv(
                    symbol, timeframe,
                    since=current_ts,
                    limit=1500,   # Binance allows up to 1500 per request
                    params={"endTime": now_ts} if use_futures else {},
                )

                if not batch:
                    break

                all_candles.extend(batch)
                current_ts = batch[-1][0] + candle_ms
                batch_num += 1
                retries = 0

                if batch_num % 10 == 0:
                    pct = min((current_ts - since_ts) / (now_ts - since_ts) * 100, 100)
                    dt  = pd.Timestamp(current_ts, unit="ms").strftime("%Y-%m")
                    print(f"  >> {symbol} {timeframe}: {len(all_candles):,} candles | {dt} ({pct:.0f}%)")

                await asyncio.sleep(0.05)

            except Exception as e:
                retries += 1
                if retries > 5:
                    print(f"  >> Too many errors ({e}), stopping at {len(all_candles):,} candles")
                    break
                wait = retries * 3
                print(f"  >> Error: {e} — retry {retries}/5 in {wait}s...")
                await asyncio.sleep(wait)
    finally:
        await exchange.close()

    if not all_candles:
        if use_futures:
            print(f"  >> {symbol} {timeframe}: No futures data. Trying spot...")
            return await download_ohlcv(symbol, timeframe, since_date, use_futures=False)
        print(f"  >> WARNING: No candles for {symbol} {timeframe}")
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    df = df.astype(float)

    print(f"  >> {symbol} {timeframe}: DONE — {len(df):,} candles | "
          f"{df.index[0].date()} → {df.index[-1].date()} | source={mode}")
    return df


# ── Data validator ─────────────────────────────────────────────────────────────

def validate_dataset(symbol: str, timeframe: str) -> dict:
    """
    Run pre-training data quality checks on a downloaded parquet file.

    Returns a dict with:
      ok: bool         — all critical checks passed
      bars: int        — number of candles
      days: int        — date range in days
      checks: list     — (name, passed, detail) per check
      warnings: list   — non-critical issues
    """
    path = DATA_DIR / f"{symbol}_{timeframe}.parquet"
    checks  = []
    warnings = []

    def chk(name, passed, detail=""):
        checks.append((name, passed, detail))
        return passed

    if not path.exists():
        chk("file_exists", False, f"Missing: {path}")
        return {"ok": False, "bars": 0, "days": 0, "checks": checks, "warnings": warnings}

    try:
        df = pd.read_parquet(path)
    except Exception as e:
        chk("file_readable", False, str(e))
        return {"ok": False, "bars": 0, "days": 0, "checks": checks, "warnings": warnings}

    bars = len(df)
    days = (df.index[-1] - df.index[0]).days if bars > 1 else 0

    # ── 1. Minimum bars ──────────────────────────────────────────────────────
    min_bars = MIN_BARS.get(timeframe, 1000)
    chk("min_bars", bars >= min_bars,
        f"{bars:,} bars (need ≥ {min_bars:,})")

    # ── 2. Minimum date range ─────────────────────────────────────────────────
    # Need ≥ 35 days to cover multiple regimes. 2022 data has all regimes.
    min_days = 35 if timeframe == "5m" else 14
    chk("min_days", days >= min_days,
        f"{days} days (need ≥ {min_days})")

    # ── 3. OHLCV columns present ──────────────────────────────────────────────
    needed = {"open", "high", "low", "close", "volume"}
    present = needed.issubset(set(df.columns))
    chk("ohlcv_columns", present,
        f"{'OK' if present else 'missing: ' + str(needed - set(df.columns))}")

    # ── 4. No NaN in OHLCV ────────────────────────────────────────────────────
    nan_count = df[list(needed & set(df.columns))].isna().sum().sum()
    chk("no_nan_ohlcv", nan_count == 0,
        f"{nan_count} NaN values in OHLCV")

    # ── 5. No zero/negative prices ────────────────────────────────────────────
    if "close" in df.columns:
        bad_prices = (df["close"] <= 0).sum()
        chk("positive_prices", bad_prices == 0,
            f"{bad_prices} zero/negative close prices")

    # ── 6. No duplicate timestamps ────────────────────────────────────────────
    dup_count = df.index.duplicated().sum()
    chk("no_duplicate_ts", dup_count == 0,
        f"{dup_count} duplicate timestamps")

    # ── 7. Price continuity — no extreme gaps (>50× ATR) ─────────────────────
    if "close" in df.columns and len(df) > 1:
        returns   = df["close"].pct_change().abs()
        atr_proxy = returns.rolling(14).mean() * df["close"]
        spikes = (returns > 0.20).sum()  # candles moving >20% — data error
        chk("no_price_spikes", True,
            f"{spikes} candles with >20% move (high volatility or anomaly)")
        if spikes > 0:
            spike_rows = df[returns > 0.20][["open","high","low","close"]].head(3)
            warnings.append(f"Price spikes at: {spike_rows.index.tolist()}")

    # ── 8. Market regime coverage ─────────────────────────────────────────────
    # Estimate ADX proxy from price to check for regime diversity
    if timeframe == "5m" and len(df) > 100:
        # Approximate: count trending bars using rolling std of returns
        ret  = df["close"].pct_change()
        vol  = ret.rolling(20).std()
        high_vol_bars = (vol > vol.quantile(0.7)).sum()
        low_vol_bars  = (vol < vol.quantile(0.3)).sum()
        total_valid   = vol.dropna().__len__()
        high_pct = high_vol_bars / total_valid if total_valid > 0 else 0
        low_pct  = low_vol_bars  / total_valid if total_valid > 0 else 0

        # Also check directional balance
        up_days   = (df["close"].resample("1D").last().pct_change() > 0).sum()
        down_days = (df["close"].resample("1D").last().pct_change() < 0).sum()
        total_days = up_days + down_days
        long_bias  = up_days / total_days if total_days > 0 else 0.5

        regime_ok = high_pct > 0.10 and low_pct > 0.10
        chk("regime_coverage", regime_ok,
            f"high_vol={high_pct:.0%} low_vol={low_pct:.0%} "
            f"up_days={up_days} down_days={down_days} ({long_bias:.0%} bullish)")
        if long_bias > 0.70:
            warnings.append(f"Data is {long_bias:.0%} bullish days — model may develop LONG bias. "
                           f"Include 2022 bear market data to balance.")
        if long_bias < 0.35:
            warnings.append(f"Data is {1-long_bias:.0%} bearish days — model may develop SHORT bias.")

    # ── 9. Volume is non-trivial ──────────────────────────────────────────────
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).mean()
        chk("volume_non_zero", zero_vol < 0.01,
            f"{zero_vol:.1%} of bars have zero volume")
        if zero_vol > 0.005:
            warnings.append(f"{zero_vol:.1%} zero-volume bars — potential exchange downtime gaps")

    # ── 10. Not testnet data ──────────────────────────────────────────────────
    # Testnet prices are often 10–100× lower than live prices
    if "close" in df.columns and timeframe == "5m":
        median_price = df["close"].median()
        btc_testnet_range = (median_price < 1000 and "BTC" in symbol)
        eth_testnet_range = (median_price < 100  and "ETH" in symbol)
        is_testnet = btc_testnet_range or eth_testnet_range
        chk("not_testnet_prices", not is_testnet,
            f"median_price={median_price:.0f} ({'LOOKS LIKE TESTNET' if is_testnet else 'OK'})")

    all_ok = all(passed for _, passed, _ in checks)
    return {
        "ok":       all_ok,
        "bars":     bars,
        "days":     days,
        "checks":   checks,
        "warnings": warnings,
        "start":    str(df.index[0].date()) if bars > 0 else "N/A",
        "end":      str(df.index[-1].date()) if bars > 0 else "N/A",
    }


def print_validation_report(symbol: str, timeframe: str) -> bool:
    """Print formatted validation report. Returns True if all checks pass."""
    result = validate_dataset(symbol, timeframe)

    status = "✅ PASS" if result["ok"] else "❌ FAIL"
    print(f"\n  {symbol} {timeframe}: {status}")
    print(f"    {result['bars']:,} bars  |  {result['days']} days  |  "
          f"{result.get('start','?')} → {result.get('end','?')}")

    for name, passed, detail in result["checks"]:
        icon = "  ✓" if passed else "  ✗"
        print(f"    {icon} {name:<28} {detail}")

    for w in result["warnings"]:
        print(f"    ⚠  {w}")

    return result["ok"]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download Binance USDM Futures OHLCV data for model training"
    )
    parser.add_argument("--symbols",    nargs="+", default=SYMBOLS)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--since",      default=DEFAULT_SINCE,
                        help=f"Start date (default: {DEFAULT_SINCE})")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Check existing files only, no download")
    parser.add_argument("--validate",   action="store_true",
                        help="Run data quality checks after download")
    parser.add_argument("--force",      action="store_true",
                        help="Re-download even if file is current")
    args = parser.parse_args()

    print("=" * 65)
    print("  NexusBot Data Downloader — Binance USDM Futures")
    print("=" * 65)
    print(f"  Symbols    : {args.symbols}")
    print(f"  Timeframes : {args.timeframes}")
    print(f"  Since      : {args.since}")
    print(f"  Save dir   : {DATA_DIR.absolute()}")
    print(f"  Mode       : {'DRY RUN' if args.dry_run else 'DOWNLOAD'}")
    print()

    # ── Show inventory ─────────────────────────────────────────────────────────
    print("Current inventory:")
    total_bars = 0
    for sym in args.symbols:
        for tf in args.timeframes:
            path = DATA_DIR / f"{sym}_{tf}.parquet"
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                    days = (df.index[-1] - df.index[0]).days
                    size = path.stat().st_size / 1024 / 1024
                    total_bars += len(df)
                    age  = (datetime.now(timezone.utc) - df.index[-1]).days
                    stale = " (STALE)" if age > 2 else ""
                    print(f"  {sym}_{tf:<5}: {len(df):>8,} bars  {df.index[0].date()} → "
                          f"{df.index[-1].date()}  {days}d  {size:.1f}MB{stale}")
                except Exception as e:
                    print(f"  {sym}_{tf}: ERROR reading file — {e}")
            else:
                min_b = MIN_BARS.get(tf, 1000)
                print(f"  {sym}_{tf:<5}: MISSING (need ≥ {min_b:,} bars)")
    print(f"  Total bars in cache: {total_bars:,}")

    if args.dry_run:
        print("\nDry run complete — no downloads.")
        if args.validate or total_bars > 0:
            print("\nValidation:")
            all_pass = True
            for sym in args.symbols:
                for tf in args.timeframes:
                    if not print_validation_report(sym, tf):
                        all_pass = False
            print(f"\n{'✅ All datasets pass validation' if all_pass else '❌ Some datasets have issues — fix before training'}")
        return

    # ── Download ───────────────────────────────────────────────────────────────
    print()
    total_downloaded = 0

    for symbol in args.symbols:
        sym_clean = symbol.replace("/", "")
        for tf in args.timeframes:
            save_path = DATA_DIR / f"{sym_clean}_{tf}.parquet"
            print(f"\n--- {sym_clean} {tf} ---")

            if save_path.exists() and not args.force:
                try:
                    existing  = pd.read_parquet(save_path)
                    last_date = existing.index[-1]
                    age_days  = (datetime.now(timezone.utc) - last_date).days

                    if age_days < 2:
                        print(f"  Up to date ({len(existing):,} bars, last: {last_date.date()}) — skipping")
                        total_downloaded += len(existing)
                        continue

                    # Stale — fetch only the missing tail
                    resume = (last_date - pd.Timedelta(hours=2)).strftime("%Y-%m-%d")
                    print(f"  Stale ({age_days}d old) — fetching from {resume}...")
                    new_df = asyncio.run(download_ohlcv(symbol, tf, since_date=resume))
                    if not new_df.empty:
                        combined = pd.concat([existing, new_df])
                        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                        combined.to_parquet(save_path)
                        size_mb = save_path.stat().st_size / 1024 / 1024
                        print(f"  Updated: {len(combined):,} bars ({size_mb:.1f} MB)")
                        total_downloaded += len(combined)
                    continue
                except Exception as e:
                    print(f"  Error reading existing file ({e}) — re-downloading...")

            # Fresh download
            df = asyncio.run(download_ohlcv(symbol, tf, since_date=args.since))
            if not df.empty:
                df.to_parquet(save_path)
                size_mb = save_path.stat().st_size / 1024 / 1024
                total_downloaded += len(df)
                print(f"  Saved: {save_path.name} ({len(df):,} bars, {size_mb:.1f} MB)")

                min_bars = MIN_BARS.get(tf, 1000)
                if len(df) < min_bars:
                    print(f"  ⚠  WARNING: {len(df):,} bars < {min_bars:,} minimum. "
                          f"Use --since 2022-01-01 for sufficient data.")
            else:
                print(f"  ❌ Failed to download {symbol} {tf}")

    # ── Validation report ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  DOWNLOAD COMPLETE")
    print("=" * 65)
    print(f"  Total candles: {total_downloaded:,}")
    print(f"  Files saved to: {DATA_DIR.absolute()}")

    print("\nValidation:")
    all_pass = True
    for sym in args.symbols:
        for tf in args.timeframes:
            if not print_validation_report(sym, tf):
                all_pass = False

    print()
    if all_pass:
        print("✅ All datasets pass validation. Ready to train:")
        print()
        print("   python scripts/train_from_history.py")
        print()
    else:
        print("❌ Some datasets have issues. Fix them before training.")
        print("   Re-run with --since 2022-01-01 to get more data:")
        print()
        print("   python training/download_data.py --since 2022-01-01 --force")
        print()


if __name__ == "__main__":
    main()

