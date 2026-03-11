"""
training/backtester.py
───────────────────────
Backtesting engine — test strategy on historical data without live trading.
Shows win rate, profit factor, max drawdown, Sharpe ratio.

Usage:
    python training/backtester.py --symbol BTCUSDT --candles 500
"""

import asyncio
import argparse
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models.hybrid_model import HybridModel
from core.models.base_model import Signal
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from core.exchange.exchange_factory import create_exchange
from config.settings import settings
from config.logger import get_logger
from rich.console import Console
from rich.table import Table
from rich import box

logger  = get_logger(__name__)
console = Console()


@dataclass
class BacktestTrade:
    entry_candle: int
    side:         str
    entry_price:  float
    exit_price:   float  = 0.0
    exit_candle:  int    = 0
    pnl_pct:      float  = 0.0
    outcome:      str    = ""    # "TP" | "SL" | "HOLD"


@dataclass
class BacktestResult:
    symbol:          str
    total_trades:    int    = 0
    winning_trades:  int    = 0
    losing_trades:   int    = 0
    win_rate:        float  = 0.0
    total_pnl_pct:   float  = 0.0
    avg_win_pct:     float  = 0.0
    avg_loss_pct:    float  = 0.0
    profit_factor:   float  = 0.0
    max_drawdown:    float  = 0.0
    trades:          List[BacktestTrade] = field(default_factory=list)


def run_backtest(
    df:        pd.DataFrame,
    symbol:    str,
    tp_pct:    float = 2.0,
    sl_pct:    float = 1.0,
    lookback:  int   = 50,
) -> BacktestResult:
    """
    Simulate trades on historical data using the Hybrid model signals.
    Processes candle by candle (no look-ahead bias).
    """
    model  = HybridModel(symbol=symbol)
    result = BacktestResult(symbol=symbol)

    in_trade   = False
    trade_side = None
    entry_price = 0.0
    entry_i     = 0
    balance_curve = [100.0]   # Start at 100 units
    current_balance = 100.0

    logger.info(f"Backtesting {symbol} | {len(df)} candles | TP={tp_pct}% SL={sl_pct}%")

    for i in range(lookback + 50, len(df)):
        window = df.iloc[:i]
        current_price = float(df["close"].iloc[i])

        # ── Manage open trade ──────────────────────────────────────────────
        if in_trade:
            if trade_side == "LONG":
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
                high = float(df["high"].iloc[i])
                low  = float(df["low"].iloc[i])

                if high >= tp_price:
                    pnl = tp_pct
                    outcome = "TP"
                elif low <= sl_price:
                    pnl = -sl_pct
                    outcome = "SL"
                else:
                    continue

            else:  # SHORT
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
                low  = float(df["low"].iloc[i])
                high = float(df["high"].iloc[i])

                if low <= tp_price:
                    pnl = tp_pct
                    outcome = "TP"
                elif high >= sl_price:
                    pnl = -sl_pct
                    outcome = "SL"
                else:
                    continue

            # Close trade
            current_balance *= (1 + pnl / 100)
            balance_curve.append(current_balance)

            trade = BacktestTrade(
                entry_candle=entry_i,
                side=trade_side,
                entry_price=entry_price,
                exit_price=current_price,
                exit_candle=i,
                pnl_pct=pnl,
                outcome=outcome,
            )
            result.trades.append(trade)
            result.total_trades += 1

            if pnl > 0:
                result.winning_trades += 1
            else:
                result.losing_trades += 1

            in_trade = False
            continue

        # ── Check for new trade ────────────────────────────────────────────
        prediction = model.predict(window)

        if prediction.signal == Signal.LONG and not in_trade:
            in_trade    = True
            trade_side  = "LONG"
            entry_price = current_price
            entry_i     = i

        elif prediction.signal == Signal.SHORT and not in_trade:
            in_trade    = True
            trade_side  = "SHORT"
            entry_price = current_price
            entry_i     = i

    # ── Calculate metrics ──────────────────────────────────────────────────
    if result.total_trades > 0:
        result.win_rate = result.winning_trades / result.total_trades

        wins   = [t.pnl_pct for t in result.trades if t.pnl_pct > 0]
        losses = [abs(t.pnl_pct) for t in result.trades if t.pnl_pct < 0]

        result.avg_win_pct  = sum(wins) / len(wins) if wins else 0
        result.avg_loss_pct = sum(losses) / len(losses) if losses else 0
        result.total_pnl_pct = current_balance - 100.0

        total_win  = sum(wins)
        total_loss = sum(losses)
        result.profit_factor = total_win / total_loss if total_loss > 0 else 999.0

        # Max drawdown
        peak = balance_curve[0]
        max_dd = 0.0
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b) / peak * 100
            max_dd = max(max_dd, dd)
        result.max_drawdown = max_dd

    return result


def print_backtest_results(result: BacktestResult):
    """Display results as a rich table."""
    table = Table(
        title=f"📊 Backtest Results — {result.symbol}",
        box=box.ROUNDED, border_style="cyan"
    )
    table.add_column("Metric", style="bold")
    table.add_column("Value",  justify="right")

    color_wr = "green" if result.win_rate > 0.5 else "red"
    color_pf = "green" if result.profit_factor > 1.2 else "red"
    color_pnl = "green" if result.total_pnl_pct > 0 else "red"

    table.add_row("Total Trades",   str(result.total_trades))
    table.add_row("Win Rate",       f"[{color_wr}]{result.win_rate:.1%}[/]")
    table.add_row("Wins / Losses",  f"{result.winning_trades} / {result.losing_trades}")
    table.add_row("Avg Win",        f"[green]+{result.avg_win_pct:.2f}%[/]")
    table.add_row("Avg Loss",       f"[red]-{result.avg_loss_pct:.2f}%[/]")
    table.add_row("Profit Factor",  f"[{color_pf}]{result.profit_factor:.2f}[/]")
    table.add_row("Total PnL",      f"[{color_pnl}]{result.total_pnl_pct:+.2f}%[/]")
    table.add_row("Max Drawdown",   f"[red]-{result.max_drawdown:.2f}%[/]")

    console.print(table)


async def main(symbol: str, candles: int, tp: float, sl: float):
    exchange = create_exchange("binance")
    await exchange.connect()

    console.print(f"📥 Fetching {candles} candles for {symbol}...")
    raw = await exchange.get_ohlcv(symbol, "15m", limit=candles)
    df  = ohlcv_to_dataframe(raw)
    df  = add_all_indicators(df)

    await exchange._exchange.close()

    result = run_backtest(df, symbol, tp_pct=tp, sl_pct=sl)
    print_backtest_results(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",  default="BTCUSDT")
    parser.add_argument("--candles", type=int, default=500)
    parser.add_argument("--tp",      type=float, default=2.0)
    parser.add_argument("--sl",      type=float, default=1.0)
    args = parser.parse_args()

    asyncio.run(main(args.symbol, args.candles, args.tp, args.sl))
