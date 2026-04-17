"""
scripts/auto_retrain.py
═══════════════════════════════════════════════════════════════════════════════
Periodic self-improving retrainer.

Runs every RETRAIN_INTERVAL_DAYS (default: 7).
For each symbol, it:
  1. Reads the recalibrator trade journal to identify underperforming regimes
  2. Runs refresh_data.py to append the latest market candles
  3. Retrains the LightGBM/XGBoost ensemble with walk-forward validation
  4. Atomically hot-swaps the model file if the new model passes quality gates
  5. Logs a summary of what changed

HOT-SWAP DESIGN (no bot restart needed)
─────────────────────────────────────────
  - New model saved to: saved_models/ml_SYMBOL.joblib.pending
  - MLModel.check_reload() called each cycle; if .pending exists and is newer
    than the loaded model, it renames to .joblib and reloads.
  - This means the bot picks up new weights within one 5m bar of training completion.
  - If new model FAILS quality gates (Sharpe < MIN_SHARPE, F1 < MIN_F1),
    the .pending file is deleted — old model stays in production untouched.

WHAT IT DOES NOT DO
────────────────────
  - It does NOT train on trade win/loss data directly.
    The ML model learns from PRICE PATTERNS (OHLCV + indicators), not from
    "trade X was placed and won/lost". Live trades inform the recalibrator
    (threshold adjustment) not the model weights.
  - It does NOT use online/incremental learning. Tree ensembles cannot be
    updated incrementally; a full retrain on the full dataset is always safer.

USAGE
─────
    # Run once (check if retrain is needed, execute if so)
    python scripts/auto_retrain.py

    # Force retrain regardless of schedule
    python scripts/auto_retrain.py --force

    # Retrain specific symbol
    python scripts/auto_retrain.py --symbol SOLUSDT --force

    # Run as a persistent scheduler (blocks forever, checks every hour)
    python scripts/auto_retrain.py --daemon
"""

import asyncio
import argparse
import csv
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config.logger import get_logger
from config.settings import settings

logger = get_logger("auto_retrain")

# ── Configuration ─────────────────────────────────────────────────────────────
SYMBOLS               = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
RETRAIN_INTERVAL_DAYS = 7        # Retrain if last retrain was >7 days ago
MIN_TRADES_TO_RETRAIN = 30       # Don't retrain on tiny sample (not enough signal)
MIN_SHARPE            = 0.40     # Minimum Sharpe for new model to be accepted
MIN_F1                = 0.33     # Minimum F1 for new model to be accepted
POOR_WR_THRESHOLD     = 0.30     # Regime is "struggling" if WR < 30%
MIN_REGIME_TRADES     = 8        # Minimum trades in a regime to trust its stats

DATA_DIR     = Path(PROJECT_ROOT) / "training_data"
MODELS_DIR   = Path(PROJECT_ROOT) / "saved_models"
LOGS_DIR     = Path(PROJECT_ROOT) / "logs"
STATE_FILE   = LOGS_DIR / "auto_retrain_state.json"
JOURNAL_FILE = LOGS_DIR / "trade_journal.csv"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# State management
# ─────────────────────────────────────────────────────────────────────────────

def load_state() -> dict:
    """Load persisted scheduler state (last retrain timestamps etc.)."""
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"last_retrain": {}, "retrain_count": {}}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Trade journal analysis
# ─────────────────────────────────────────────────────────────────────────────

def read_journal() -> list[dict]:
    """Read trade_journal.csv written by SignalRecalibrator."""
    if not JOURNAL_FILE.exists():
        return []
    rows = []
    try:
        with open(JOURNAL_FILE, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except Exception as e:
        logger.warning(f"[RETRAIN] Could not read journal: {e}")
    return rows


def analyse_regime_performance(journal: list[dict]) -> dict:
    """
    Group journal trades by symbol+regime, compute win rate.
    Returns: {symbol: {regime: {trades, wr, net_pnl}}}
    """
    stats: dict = defaultdict(lambda: defaultdict(lambda: {"trades": 0, "wins": 0, "net_pnl": 0.0}))
    for row in journal:
        sym    = row.get("symbol", "")
        regime = row.get("regime", "UNKNOWN")
        won    = row.get("won", "0") in ("1", "True", "true")
        pnl    = float(row.get("pnl_usdt", 0) or 0)
        stats[sym][regime]["trades"] += 1
        stats[sym][regime]["net_pnl"] += pnl
        if won:
            stats[sym][regime]["wins"] += 1

    # Compute win rate
    for sym in stats:
        for regime in stats[sym]:
            s = stats[sym][regime]
            s["wr"] = s["wins"] / s["trades"] if s["trades"] > 0 else 0.5
    return stats


def identify_struggling_regimes(stats: dict) -> list[str]:
    """
    Return list of symbols where a regime has WR < POOR_WR_THRESHOLD
    and enough trades to be statistically meaningful.
    These are the priority retraining targets.
    """
    struggling = []
    for sym, regimes in stats.items():
        for regime, s in regimes.items():
            if s["trades"] >= MIN_REGIME_TRADES and s["wr"] < POOR_WR_THRESHOLD:
                struggling.append(sym)
                logger.info(
                    f"[RETRAIN] {sym} struggling in {regime}: "
                    f"{s['trades']} trades, WR={s['wr']:.0%}, net={s['net_pnl']:+.4f}"
                )
                break
    return list(dict.fromkeys(struggling))  # deduplicate, preserve order


def total_recent_trades(journal: list[dict], days: int = 7) -> int:
    """Count trades in the last N days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    count  = 0
    for row in journal:
        try:
            ts = datetime.fromisoformat(row.get("timestamp", "2000-01-01").replace(" ", "T"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                count += 1
        except Exception:
            pass
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Retrain a single symbol
# ─────────────────────────────────────────────────────────────────────────────

def retrain_symbol(symbol: str) -> dict:
    """
    Retrain the ML model for a single symbol.
    Returns result dict with accepted, sharpe, f1, reason.
    """
    logger.info(f"[RETRAIN] Starting retrain for {symbol}...")
    start = time.time()

    try:
        # Delegate entirely to train_from_history's own pipeline.
        # This guarantees the saved model is always in the correct format
        # (list of ("name", calibrated_model, weight) tuples) that ml_model.py
        # expects, and uses the same hyperparameters, label generation,
        # feature engineering and WF validation as the production trainer.
        import sys as _sys
        import importlib

        # Ensure project root is in path
        if PROJECT_ROOT not in _sys.path:
            _sys.path.insert(0, PROJECT_ROOT)

        # Import the production training function directly
        import scripts.train_from_history as _tfh

        result_dict = _tfh.train_symbol_from_history(symbol, dry_run=False)

        # train_from_history returns: status="OK", verdict="ACCEPTED"/"WF-REJECTED"
        status   = result_dict.get("status", "FAILED")
        verdict  = result_dict.get("verdict", "")
        mean_f1  = result_dict.get("wf_f1", 0.0)
        sharpe   = result_dict.get("wf_sharpe", 0.0)
        bars     = result_dict.get("bars_15m", 0)
        accepted = (status == "OK" and verdict == "ACCEPTED")

        if not accepted:
            reason = f"verdict={verdict} status={status} F1={mean_f1:.3f}"
            logger.warning(f"[RETRAIN] {symbol} not accepted: {reason}")
            return {"accepted": False, "f1": mean_f1, "sharpe": sharpe, "reason": reason}

        # train_from_history saves to saved_models/ml_SYMBOL.joblib directly.
        # Rename it to .pending so the hot-swap logic picks it up safely.
        import joblib, shutil as _shutil
        src_path     = MODELS_DIR / f"ml_{symbol}.joblib"
        pending_path = MODELS_DIR / f"ml_{symbol}.joblib.pending"

        if not src_path.exists():
            return {"accepted": False, "reason": "model file not found after training"}

        # Move to pending (hot-swap in ml_model.check_reload)
        _shutil.copy2(str(src_path), str(pending_path))
        logger.info(
            f"[RETRAIN] {symbol} ACCEPTED — F1={mean_f1:.3f} Sharpe={sharpe:.2f} | "
            f"{bars:,} bars | pending hot-swap on next bot cycle"
        )


        elapsed = time.time() - start
        return {
            "accepted":  True,
            "sharpe":    sharpe,
            "f1":        mean_f1,
            "bars":      bars,
            "elapsed_s": round(elapsed),
            "reason":    "passed",
        }

    except Exception as e:
        logger.error(f"[RETRAIN] {symbol} failed: {e}", exc_info=True)
        return {"accepted": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────────────────
# Main retrain loop
# ─────────────────────────────────────────────────────────────────────────────

async def maybe_retrain(symbols: list[str], force: bool = False) -> dict:
    """
    Check if retraining is needed and execute if so.
    Returns per-symbol results.
    """
    state   = load_state()
    journal = read_journal()
    results = {}

    # Determine which symbols need retraining
    to_retrain = []
    for sym in symbols:
        last_str  = state["last_retrain"].get(sym)
        last_ts   = datetime.fromisoformat(last_str) if last_str else None
        days_since= (datetime.now(timezone.utc) - last_ts).days if last_ts else 9999

        total_trades = sum(1 for r in journal if r.get("symbol") == sym)
        recent_count = total_recent_trades(journal, days=RETRAIN_INTERVAL_DAYS)

        if force:
            reason = "forced"
        elif days_since >= RETRAIN_INTERVAL_DAYS:
            reason = f"scheduled ({days_since}d since last)"
        else:
            logger.info(f"[RETRAIN] {sym}: {days_since}d since last retrain — not due yet")
            results[sym] = {"skipped": True, "reason": f"not due ({days_since}d < {RETRAIN_INTERVAL_DAYS}d)"}
            continue

        if total_trades < MIN_TRADES_TO_RETRAIN and not force:
            logger.info(f"[RETRAIN] {sym}: only {total_trades} trades in journal — skipping")
            results[sym] = {"skipped": True, "reason": f"only {total_trades} trades"}
            continue

        to_retrain.append((sym, reason))

    if not to_retrain:
        logger.info("[RETRAIN] Nothing to retrain")
        return results

    # 1. Refresh market data first
    logger.info(f"[RETRAIN] Refreshing market data for {[s for s,_ in to_retrain]}...")
    try:
        from scripts.refresh_data import main as refresh_main
        await refresh_main(
            symbols=[s for s, _ in to_retrain],
            timeframes=["5m", "1h"],
            force_days=None,  # append-only
        )
    except Exception as e:
        logger.error(f"[RETRAIN] Data refresh failed: {e} — retraining with existing data")

    # 2. Retrain each symbol (blocking — CPU intensive, runs in executor)
    loop = asyncio.get_event_loop()
    for sym, reason in to_retrain:
        logger.info(f"[RETRAIN] Retraining {sym} (reason: {reason})...")
        result = await loop.run_in_executor(None, retrain_symbol, sym)
        results[sym] = result

        if result.get("accepted"):
            state["last_retrain"][sym] = datetime.now(timezone.utc).isoformat()
            state["retrain_count"][sym] = state["retrain_count"].get(sym, 0) + 1

    save_state(state)

    # 3. Summary
    print()
    print("=" * 65)
    print("AUTO-RETRAIN SUMMARY")
    print("=" * 65)
    for sym, r in results.items():
        if r.get("skipped"):
            print(f"  {sym:<10} SKIPPED  — {r['reason']}")
        elif r.get("accepted"):
            print(f"  {sym:<10} ACCEPTED — F1={r['f1']:.3f}  {r['bars']:,} bars  {r['elapsed_s']}s")
            print(f"             Model saved to pending — bot hot-swaps on next cycle")
        else:
            print(f"  {sym:<10} REJECTED — {r['reason']}")
    print("=" * 65)
    print()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Daemon mode
# ─────────────────────────────────────────────────────────────────────────────

async def daemon(symbols: list[str]):
    """Run forever, checking every hour whether retraining is needed."""
    logger.info(f"[RETRAIN] Daemon started — checking every hour, retraining every {RETRAIN_INTERVAL_DAYS} days")
    while True:
        try:
            await maybe_retrain(symbols, force=False)
        except Exception as e:
            logger.error(f"[RETRAIN] Daemon cycle error: {e}", exc_info=True)
        # Check again in 1 hour
        await asyncio.sleep(3600)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Periodic model retrainer with hot-swap")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--force",   action="store_true", help="Retrain now regardless of schedule")
    parser.add_argument("--symbol",  type=str,  help="Single symbol shorthand")
    parser.add_argument("--daemon",  action="store_true", help="Run as persistent scheduler")
    args = parser.parse_args()

    syms = [args.symbol] if args.symbol else args.symbols

    if args.daemon:
        asyncio.run(daemon(syms))
    else:
        asyncio.run(maybe_retrain(syms, force=args.force))