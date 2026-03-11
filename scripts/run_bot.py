"""
scripts/run_bot.py — Main entry point.

Usage:
    python scripts/run_bot.py
    python scripts/run_bot.py --pairs BTCUSDT ETHUSDT --leverage 10 --tp 2.0 --sl 1.0
    python scripts/run_bot.py --train
"""

import asyncio
import argparse
import sys
import signal
from pathlib import Path
from typing import List
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exchange.exchange_factory import create_exchange
from core.risk.risk_manager import RiskManager
from core.strategy.recovery_strategy import RecoveryStrategy
from core.trader.futures_trader import FuturesTrader, _normalize_symbol
from config.settings import settings, RiskSettings
from config.logger import get_logger
from rich.console import Console
from rich.table import Table
from rich import box

logger  = get_logger(__name__)
console = Console()

_traders: List[FuturesTrader] = []
_running = True


def signal_handler(sig, frame):
    global _running
    console.print("\n[bold yellow]Stopping bot...[/]")
    _running = False
    for t in _traders:
        t.stop()

signal.signal(signal.SIGINT,  signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


async def print_dashboard(exchange, traders: List[FuturesTrader]):
    try:
        balance   = await exchange.get_balance()
        positions = await exchange.get_open_positions()

        # Build position lookup by normalized symbol
        pos_map = {}
        for p in positions:
            pos_map[_normalize_symbol(p.symbol)] = p

        table = Table(
            title=f"AI Trading Bot — {datetime.now().strftime('%H:%M:%S')}",
            box=box.ROUNDED, border_style="cyan",
        )
        table.add_column("Symbol",   style="bold white")
        table.add_column("Cycles",   justify="right")
        table.add_column("ML",       justify="center")
        table.add_column("Side",     justify="center")
        table.add_column("Entry",    justify="right")
        table.add_column("TP / SL",  justify="center")
        table.add_column("PnL",      justify="right")

        for t in traders:
            stats = t.get_stats()
            pos   = pos_map.get(_normalize_symbol(t.symbol))

            if pos:
                side_str  = "[green]LONG[/]"  if pos.side.value == "long" else "[red]SHORT[/]"
                entry_str = f"{pos.entry_price:.2f}"
                pnl_val   = pos.unrealized_pnl
                pnl_str   = f"[green]+{pnl_val:.4f}[/]" if pnl_val >= 0 else f"[red]{pnl_val:.4f}[/]"
                tp_sl_str = (f"{stats['tp']:.2f} / {stats['sl']:.2f}"
                             if stats["tp"] else "monitoring...")
            else:
                side_str  = "[dim]Watching[/]"
                entry_str = "—"
                pnl_str   = "—"
                tp_sl_str = "—"

            table.add_row(
                stats["symbol"],
                str(stats["cycles"]),
                "[green]YES[/]" if stats["ml_trained"] else "[yellow]NO[/]",
                side_str,
                entry_str,
                tp_sl_str,
                pnl_str,
            )

        console.print(table)

        pnl_color = "green" if balance.unrealized_pnl >= 0 else "red"
        start_bal = 5000.0
        total_pnl = balance.total_balance - start_bal
        total_color = "green" if total_pnl >= 0 else "red"

        console.print(
            f"  Balance: [bold green]{balance.total_balance:.2f} USDT[/]  |  "
            f"Available: [cyan]{balance.available_balance:.2f} USDT[/]  |  "
            f"Floating: [{pnl_color}]{balance.unrealized_pnl:+.4f} USDT[/]  |  "
            f"Session PnL: [{total_color}]{total_pnl:+.2f} USDT[/]"
        )

    except Exception as e:
        logger.warning(f"Dashboard error: {e}")


async def run_bot(pairs, train, leverage, tp_pct, sl_pct, risk_pct, interval):
    global _traders, _running

    console.rule("[bold cyan]AI Futures Trading Bot — Starting[/]")
    console.print(f"  Environment : [bold]{settings.environment.upper()}[/]")
    console.print(f"  Pairs       : {', '.join(pairs)}")
    console.print(f"  Leverage    : {leverage}x  |  TP: {tp_pct}%  |  SL: {sl_pct}%")
    console.print(f"  Cycle every : {interval}s\n")

    exchange  = create_exchange("binance")
    connected = await exchange.connect()
    if not connected:
        console.print("[bold red]Connection failed — check .env[/]")
        return

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

    balance = await exchange.get_balance()
    risk_manager.set_session_balance(balance.total_balance)
    console.print(f"  Starting balance: [bold green]{balance.total_balance:.2f} USDT[/]\n")

    for symbol in pairs:
        trader = FuturesTrader(
            exchange=exchange, symbol=symbol,
            risk_manager=risk_manager, recovery_strategy=recovery,
        )
        _traders.append(trader)
        logger.info(f"Trader ready: {symbol}")

    if train:
        for trader in _traders:
            try:
                await trader.train_ml_model(candle_limit=1000)
            except Exception as e:
                logger.error(f"Training failed {trader.symbol}: {e}")

    console.print("[bold green]Bot running — Ctrl+C to stop[/]\n")

    while _running:
        try:
            await asyncio.gather(
                *[t.run_cycle() for t in _traders],
                return_exceptions=True
            )
            await print_dashboard(exchange, _traders)

            for _ in range(interval):
                if not _running:
                    break
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Main loop error: {e}", exc_info=True)
            await asyncio.sleep(5)

    console.print("[bold yellow]Closing...[/]")
    await exchange._exchange.close()
    console.print("[bold green]Bot stopped.[/]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs",    nargs="+", default=settings.trading_pairs)
    parser.add_argument("--train",    action="store_true")
    parser.add_argument("--leverage", type=int,   default=settings.risk.leverage)
    parser.add_argument("--tp",       type=float, default=settings.risk.take_profit_pct)
    parser.add_argument("--sl",       type=float, default=settings.risk.stop_loss_pct)
    parser.add_argument("--risk",     type=float, default=settings.risk.risk_per_trade_pct)
    parser.add_argument("--interval", type=int,   default=60)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_bot(
        pairs=args.pairs, train=args.train,
        leverage=args.leverage, tp_pct=args.tp,
        sl_pct=args.sl, risk_pct=args.risk,
        interval=args.interval,
    ))