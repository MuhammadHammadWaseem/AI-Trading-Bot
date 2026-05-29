"""
scripts/run_bot_managed.py
──────────────────────────
Entry point when launched by Laravel's BotService.

ROOT CAUSE FIXES (2026-03-26):
  1. APP_URL is now read from .env so it matches 'php artisan serve' URL
  2. Startup connectivity test — if Laravel unreachable, bot refuses to start
     and reports error status clearly so dashboard shows the real problem
  3. reporter.start() is ONLY called here (not in futures_trader.run()) to
     avoid duplicate background tasks corrupting log flush
  4. _final_status/_final_message guarantee status always posted on exit

Usage:
    python run_bot_managed.py --config /path/to/bot_N.json
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
from pathlib import Path

# Shared hosts often expose many CPU cores but enforce strict process/thread
# limits. Cap BLAS/OpenMP before importing numpy/scipy/sklearn through the bot.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.integrations.laravel_reporter import LaravelReporter
from core.trader.futures_trader import FuturesTrader
from core.exchange.exchange_factory import create_exchange_from_config, validate_binance_keys
from core.risk.risk_manager import RiskManager
from core.strategy.recovery_strategy import RecoveryStrategy
from core.models.signal_recalibrator import SignalRecalibrator
from config.settings import settings, RiskSettings
from config.logger import get_logger

logger = get_logger(__name__)

# Track final status so the finally block always reports accurately
_final_status  = "stopped"
_final_message = ""


def load_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_app_url_from_env() -> str:
    """
    Read APP_URL from the project's .env file.
    Falls back to http://127.0.0.1:8000 for 'php artisan serve'.
    """
    env_path = settings.base_dir / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("APP_URL="):
                url = line.split("=", 1)[1].strip().strip('"').strip("'")
                return url.rstrip("/")
    return "http://127.0.0.1:8000"


async def main():
    global _final_status, _final_message

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config    = load_config(args.config)
    bot_id    = config["bot_id"]
    bot_token = config["bot_token"]

    # ── Determine correct API URL ────────────────────────────────────────
    # Priority: config file > .env APP_URL > fallback
    # The config file should have been written by BotService with the correct URL.
    # If not present, read APP_URL from .env (works for both artisan serve and Apache).
    default_api_base = get_app_url_from_env()
    # Normalize URL — remove any double-slashes from APP_URL trailing slash
    raw = config.get("laravel_api_url") or f"{default_api_base}/api/bot"
    api_url = raw.replace("//api/", "/api/").rstrip("/")

    # ── Reporter setup FIRST (before any bot code) ────────────────────────
    reporter = LaravelReporter(api_url=api_url, bot_token=bot_token, bot_id=bot_id)
    reporter.attach_to_logging()  # routes ALL Python logs to dashboard

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        force=True,
    )

    # Start reporter background tasks immediately
    await reporter.start()

    logger.info(f"[MANAGED] Bot #{bot_id} starting | symbol={config['symbol']} | "
                f"leverage={config['leverage']}x | API={api_url}")

    # ── CONNECTIVITY TEST ─────────────────────────────────────────────────
    # If Laravel is unreachable, refuse to start and show a clear error.
    # Without this, the bot runs silently with no dashboard visibility at all.
    connected = await reporter.startup_test()
    if not connected:
        _final_status  = "error"
        _final_message = (
            f"Cannot reach Laravel API at {api_url}. "
            f"Ensure 'php artisan serve' is running on port 8000 "
            f"and APP_URL in .env matches. Check bot_{bot_id}.log for details."
        )
        logger.critical(f"[MANAGED] {_final_message}")
        # Try to post error status (may fail if truly unreachable, that's OK)
        try:
            await reporter.report_status("error", message=_final_message)
            await reporter.flush()
        except Exception:
            pass
        await reporter.close()
        os.remove(args.config)
        return

    # ── Graceful shutdown via OS signal ───────────────────────────────────
    stop_event = asyncio.Event()

    def handle_signal(signum, frame):
        logger.info("[MANAGED] Shutdown signal received.")
        stop_event.set()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT,  handle_signal)
    try:
        signal.signal(signal.SIGBREAK, handle_signal)   # Windows Ctrl+Break
    except (AttributeError, OSError):
        pass

    try:
        # ── Exchange connection ─────────────────────────────────────────────
        # ── Mode: paper vs live ─────────────────────────────────────────
        trading_mode  = config.get("trading_mode", "paper")
        paper_balance = float(config.get("paper_balance", 10_000.0))
        configured_timeframe = config.get("timeframe", "5m")
        if configured_timeframe != "5m":
            _final_status = "error"
            _final_message = (
                "Unsupported timeframe. Current production model is trained only "
                "on 5m candles with 1h context. Reconfigure this bot to 5m."
            )
            logger.critical(f"[MANAGED] {_final_message}")
            return
        if trading_mode == "live":
            logger.warning(
                f"[MANAGED] ⚠️  LIVE mode — real funds on the line. "
                f"Double-check your risk settings before proceeding."
            )
        exchange = await create_exchange_from_config(
            exchange_name = config.get("exchange", "binance"),
            api_key       = config["api_key"],
            api_secret    = config["api_secret"],
            trading_mode  = trading_mode,
            paper_balance = paper_balance,
        )

        ok = await exchange.connect()
        if not ok:
            _final_status  = "error"
            _final_message = "Exchange connection failed — check API keys in dashboard."
            logger.critical(f"[MANAGED] {_final_message}")
            return

        # ── Risk manager ─────────────────────────────────────────────────
        rs = RiskSettings(
            leverage             = config["leverage"],
            take_profit_pct      = config.get("take_profit_pct", settings.risk.take_profit_pct),
            stop_loss_pct        = config.get("stop_loss_pct",   settings.risk.stop_loss_pct),
            risk_per_trade_pct   = config["risk_per_trade"],
            max_open_trades      = int(config.get("max_open_trades", settings.risk.max_open_trades)),
            max_daily_loss_pct   = settings.risk.max_daily_loss_pct,
            max_daily_loss_usdt  = config.get("daily_loss_limit", 0.0),
        )
        risk_manager = RiskManager(rs)

        mode_label = "📄 PAPER" if trading_mode == "paper" else "💰 LIVE"
        logger.info(f"[MANAGED] Trading mode: {mode_label}")
        if trading_mode == "paper":
            logger.info(f"[MANAGED] Paper balance: ${paper_balance:,.2f} USDT (virtual)")

        balance = await exchange.get_balance()
        risk_manager.set_session_balance(balance.total_balance)
        logger.info(f"[MANAGED] Account balance: {balance.total_balance:.2f} USDT")

        # ── Build trader ─────────────────────────────────────────────────
        # ── Pass user settings to trader ───────────────────────────────
        # FIX: base_confidence_threshold was never passed here, so the bot
        # always used the class-level default (0.55 = 55%) regardless of what
        # the user configured in the dashboard. Now it reads from config JSON.
        base_threshold = config.get("base_confidence_threshold", None)

        trader = FuturesTrader(
            exchange          = exchange,
            symbol            = config["symbol"],
            risk_manager      = risk_manager,
            recovery_strategy = RecoveryStrategy(),
            recalibrator      = SignalRecalibrator(log_dir=settings.logs_dir),
            reporter          = reporter,
            stop_event        = stop_event,
            base_threshold    = base_threshold,   # ← FIX: user-defined min confidence
            timeframe         = configured_timeframe,
            max_trades_per_day = int(config.get("max_trades_per_day", 0) or 0),
        )

        # Log the effective threshold so the user can see it in the dashboard
        if base_threshold is not None:
            eff = base_threshold if base_threshold <= 1 else base_threshold / 100.0
            logger.info(
                f"[MANAGED] Min confidence threshold set to {eff:.0%} "                f"(from dashboard config). Trades below this level will be skipped."
            )

        # ── Report running THEN start main loop ──────────────────────────
        # NOTE: reporter.start() was already called above — do NOT call it again.
        # FuturesTrader.run() will NOT call reporter.start() (patched separately).
        await reporter.report_status("running")
        await trader.run(stop_event=stop_event)

        # ── Post-run: close positions if user requested it ────────────────
        if reporter.should_close_trades():
            logger.info("[MANAGED] Closing all open positions as requested by user...")
            reporter.queue_log(
                "info", "🔴 Closing all open positions on exchange as requested...", channel="bot"
            )
            try:
                await trader.close_all_positions_for_shutdown()
            except Exception as e:
                logger.warning(f"[MANAGED] Error during position close: {e}")
                reporter.queue_log(
                    "warning",
                    f"⚠️ Could not auto-close all positions: {e}. Please check exchange manually.",
                    channel="bot"
                )
            await reporter.flush()

        _final_status  = "stopped"
        _final_message = ""

    except Exception as e:
        _final_status  = "error"
        _final_message = str(e)[:500]
        logger.critical(f"[MANAGED] Fatal error: {e}", exc_info=True)

    finally:
        # ── ALWAYS report final status to Laravel ─────────────────────────
        try:
            await reporter.flush()
            await reporter.report_status(_final_status, message=_final_message)
            await reporter.close()
        except Exception as cleanup_err:
            logger.warning(f"[MANAGED] Cleanup error (non-fatal): {cleanup_err}")

        # Remove config so credentials don't linger on disk
        try:
            os.remove(args.config)
        except Exception:
            pass

        logger.info(f"[MANAGED] Bot process finished with status: {_final_status}")


if __name__ == "__main__":
    asyncio.run(main())
