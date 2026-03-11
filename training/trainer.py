"""
training/trainer.py
────────────────────
Standalone ML training script.
Fetches historical data and trains LSTM models for all configured symbols.

Usage:
    python training/trainer.py
    python training/trainer.py --symbols BTCUSDT ETHUSDT --candles 2000
"""

import asyncio
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.exchange.exchange_factory import create_exchange
from core.models.ml_model import MLModel
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.settings import settings
from config.logger import get_logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

logger  = get_logger(__name__)
console = Console()


async def train_symbol(
    exchange,
    symbol:       str,
    candle_limit: int,
    timeframe:    str = "15m",
):
    """Fetch data and train LSTM for one symbol."""
    console.print(f"\n  📊 Fetching {candle_limit} candles for [cyan]{symbol}[/]...")

    candles = await exchange.get_ohlcv(symbol, timeframe, limit=candle_limit)
    if not candles or len(candles) < 200:
        logger.error(f"Not enough candles for {symbol}: {len(candles) if candles else 0}")
        return False

    df = ohlcv_to_dataframe(candles)
    df = add_all_indicators(df)

    console.print(
        f"  ✅ Data ready: {len(df)} rows | "
        f"{df.index[0].strftime('%Y-%m-%d')} → {df.index[-1].strftime('%Y-%m-%d')}"
    )

    model = MLModel(symbol=symbol)

    console.print(f"  🧠 Training LSTM for [cyan]{symbol}[/]...")
    history = model.train(df, epochs=50)

    final_acc  = history.history.get("val_accuracy", [0])[-1]
    final_loss = history.history.get("val_loss", [999])[-1]

    console.print(
        f"  ✅ [bold green]{symbol}[/] trained | "
        f"val_acc=[bold]{final_acc:.2%}[/] | val_loss={final_loss:.4f}"
    )
    return True


async def train_all(symbols: list, candle_limit: int):
    """Train all symbols sequentially."""
    console.rule("[bold cyan]🧠 AI Model Training[/]")

    exchange = create_exchange("binance")
    connected = await exchange.connect()
    if not connected:
        console.print("[red]❌ Exchange connection failed[/]")
        return

    results = {}
    for symbol in symbols:
        success = await train_symbol(exchange, symbol, candle_limit)
        results[symbol] = "✅ Success" if success else "❌ Failed"

    await exchange._exchange.close()

    console.rule("[bold]Training Results[/]")
    for sym, status in results.items():
        console.print(f"  {sym}: {status}")

    console.print(
        f"\n  Models saved to: [cyan]{settings.model.saved_models_dir}[/]"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train AI trading models")
    parser.add_argument(
        "--symbols", nargs="+",
        default=settings.trading_pairs,
        help="Symbols to train"
    )
    parser.add_argument(
        "--candles", type=int, default=1500,
        help="Historical candles to use for training"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(train_all(args.symbols, args.candles))
