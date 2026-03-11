"""
scripts/run_bot.py
───────────────────
Main entry point — run from terminal.

Usage:
    # Run with default settings from .env
    python scripts/run_bot.py

    # Train ML models first, then trade
    python scripts/run_bot.py --train

    # Trade specific pairs only
    python scripts/run_bot.py --pairs BTCUSDT ETHUSDT

    # Custom TP/SL
    python scripts/run_bot.py --tp 3.0 --sl 1.5

    # Set leverage
    python scripts/run_bot.py --leverage 10
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from typing import List
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exchange.exchange_factory import create_exchange
from core.risk.risk_manager import RiskManager
from core.strategy.recovery_strategy import RecoveryStrategy
from core.trader.futures_trader import FuturesTrader
from config.settings import settings, RiskSettings
from config.logger import get_logger
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich import box

logger  = get_logger(__name__)
console = Console()

# ── Global trader registry ─────────────────────────────────────────────────────
_traders: List[FuturesTrader] = []
_running = True


def signal_handler(sig, frame):
    """Graceful shutdown on Ctrl+C."""
    global _running
    console.print("\n[bold yellow]⚠️  Shutdown signal received — stopping gracefully...[/]")
    _running = False
    for t in _traders:
        t.stop()


signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def print_dashboard(exchange, traders: List[FuturesTrader]):
    """Print a live status table."""
    try:
        balance   = await exchange.get_balance()
        positions = await exchange.get_open_positions()

        table = Table(
            title=f"🤖 AI Trading Bot — {datetime.now().strftime('%H:%M:%S')}",
            box=box.ROUNDED,
            border_style="cyan",
        )
        table.add_column("Symbol",     style="bold white")
        table.add_column("Cycles",     justify="right")
        table.add_column("ML Trained", justify="center")
        table.add_column("Recovering", justify="center")
        table.add_column("Status")

        for t in traders:
            stats = t.get_stats()
            pos   = next((p for p in positions if p.symbol == t.symbol), None)
            if pos:
                status = (
                    f"[green]LONG[/]  @ {pos.entry_price} | "
                    f"PnL: {pos.unrealized_pnl:+.4f}"
                ) if pos.side.value == "long" else (
                    f"[red]SHORT[/] @ {pos.entry_price} | "
                    f"PnL: {pos.unrealized_pnl:+.4f}"
                )
            else:
                status = "[dim]Watching...[/]"

            table.add_row(
                stats["symbol"],
                str(stats["cycles"]),
                "✅" if stats["ml_trained"] else "⏳",
                "🔄" if stats["in_recovery"] else "—",
                status,
            )

        console.print(table)
        console.print(
            f"  💰 Balance: [bold green]{balance.total_balance:.2f} USDT[/]  |  "
            f"Available: [cyan]{balance.available_balance:.2f} USDT[/]  |  "
            f"PnL: [{'green' if balance.unrealized_pnl >= 0 else 'red'}]"
            f"{balance.unrealized_pnl:+.4f} USDT[/]"
        )

    except Exception as e:
        logger.warning(f"Dashboard error: {e}")


async def run_bot(
    pairs:     List[str],
    train:     bool,
    leverage:  int,
    tp_pct:    float,
    sl_pct:    float,
    risk_pct:  float,
    interval:  int,
):
    """Main bot loop."""
    global _traders, _running

    console.rule("[bold cyan]🤖 AI Futures Trading Bot — Starting[/]")
    console.print(f"  Environment : [bold]{settings.environment.upper()}[/]")
    console.print(f"  Exchange    : Binance")
    console.print(f"  Pairs       : {', '.join(pairs)}")
    console.print(f"  Leverage    : {leverage}x")
    console.print(f"  TP / SL     : {tp_pct}% / {sl_pct}%")
    console.print(f"  Interval    : {interval}s\n")

    # ── Connect to exchange ────────────────────────────────────────────────
    exchange = create_exchange("binance")
    connected = await exchange.connect()

    if not connected:
        console.print("[bold red]❌ Could not connect to exchange — check .env API keys[/]")
        return

    # ── Setup risk & recovery ──────────────────────────────────────────────
    risk_settings = RiskSettings(
        leverage=leverage,
        take_profit_pct=tp_pct,
        stop_loss_pct=sl_pct,
        risk_per_trade_pct=risk_pct,
        max_open_trades=settings.risk.max_open_trades,
        max_daily_loss_pct=settings.risk.max_daily_loss_pct,
    )

    risk_manager = RiskManager(risk_settings)
    recovery     = RecoveryStrategy()

    # Set session starting balance
    balance = await exchange.get_balance()
    risk_manager.set_session_balance(balance.total_balance)
    console.print(
        f"  💰 Starting balance: [bold green]{balance.total_balance:.2f} USDT[/]\n"
    )

    # ── Create traders (one per symbol) ───────────────────────────────────
    for symbol in pairs:
        trader = FuturesTrader(
            exchange=exchange,
            symbol=symbol,
            risk_manager=risk_manager,
            recovery_strategy=recovery,
        )
        _traders.append(trader)
        logger.info(f"✅ Trader created: {symbol}")

    # ── Optionally train ML models ────────────────────────────────────────
    if train:
        console.print("\n[bold yellow]🧠 Training ML models...[/]")
        for trader in _traders:
            try:
                await trader.train_ml_model(candle_limit=1000)
            except Exception as e:
                logger.error(f"Training failed for {trader.symbol}: {e}")

    # ── Main trading loop ─────────────────────────────────────────────────
    console.print("\n[bold green]▶️  Bot is running — press Ctrl+C to stop[/]\n")

    while _running:
        try:
            # Run all traders in parallel
            await asyncio.gather(
                *[trader.run_cycle() for trader in _traders],
                return_exceptions=True
            )

            # Print dashboard every cycle
            await print_dashboard(exchange, _traders)

            # Wait for next cycle
            for _ in range(interval):
                if not _running:
                    break
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            await asyncio.sleep(5)

    # ── Cleanup ────────────────────────────────────────────────────────────
    console.print("\n[bold yellow]Closing exchange connection...[/]")
    await exchange._exchange.close()
    console.print("[bold green]✅ Bot stopped cleanly.[/]")


def parse_args():
    parser = argparse.ArgumentParser(description="AI Futures Trading Bot")
    parser.add_argument(
        "--pairs", nargs="+", default=settings.trading_pairs,
        help="Trading pairs (e.g. BTCUSDT ETHUSDT)"
    )
    parser.add_argument("--train", action="store_true", help="Train ML models before trading")
    parser.add_argument("--leverage", type=int, default=settings.risk.leverage)
    parser.add_argument("--tp",       type=float, default=settings.risk.take_profit_pct, help="Take profit %")
    parser.add_argument("--sl",       type=float, default=settings.risk.stop_loss_pct,   help="Stop loss %")
    parser.add_argument("--risk",     type=float, default=settings.risk.risk_per_trade_pct, help="Risk per trade %")
    parser.add_argument("--interval", type=int, default=60, help="Cycle interval in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_bot(
        pairs=args.pairs,
        train=args.train,
        leverage=args.leverage,
        tp_pct=args.tp,
        sl_pct=args.sl,
        risk_pct=args.risk,
        interval=args.interval,
    ))
