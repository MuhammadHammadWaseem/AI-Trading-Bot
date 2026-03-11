"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator for a single symbol.
Coordinates: data → indicators → prediction → risk → execution → monitoring.

One FuturesTrader instance per symbol per user.
"""

import asyncio
from datetime import datetime
from typing import Optional

from core.exchange.base_exchange import BaseExchange, OrderSide, PositionInfo
from core.models.hybrid_model import HybridModel
from core.risk.risk_manager import RiskManager, TradeParameters
from core.strategy.recovery_strategy import RecoveryStrategy
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


class FuturesTrader:
    """
    Manages the full lifecycle of trading a single symbol.
    
    Usage:
        trader = FuturesTrader(exchange, "BTCUSDT", risk_manager, recovery)
        await trader.run_cycle()   # Call on every candle close
    """

    TIMEFRAME = "15m"   # Primary analysis timeframe

    def __init__(
        self,
        exchange:          BaseExchange,
        symbol:            str,
        risk_manager:      RiskManager,
        recovery_strategy: RecoveryStrategy,
    ):
        self.exchange   = exchange
        self.symbol     = symbol
        self.risk       = risk_manager
        self.recovery   = recovery_strategy
        self.model      = HybridModel(symbol=symbol)

        self._current_position: Optional[PositionInfo] = None
        self._is_active = True
        self._cycles    = 0
        self._trades_opened = 0
        self._trades_closed = 0

    async def run_cycle(self):
        """
        Execute one trading cycle:
        1. Fetch data
        2. Calculate indicators
        3. Predict signal
        4. Manage existing position (exit if needed)
        5. Open new position (if signal + risk approved)
        """
        if not self._is_active:
            return

        self._cycles += 1
        logger.info(f"🔄 Cycle #{self._cycles} — {self.symbol}")

        try:
            # ── Step 1: Fetch & prepare data ──────────────────────────────
            candles = await self.exchange.get_ohlcv(
                self.symbol, self.TIMEFRAME, limit=300
            )
            if not candles or len(candles) < 50:
                logger.warning(f"Insufficient candles for {self.symbol}")
                return

            df = ohlcv_to_dataframe(candles)
            df = add_all_indicators(df)
            current_price = float(df["close"].iloc[-1])

            # ── Step 2: Check existing position ───────────────────────────
            await self._check_existing_position(current_price)

            # ── Step 3: Get prediction ─────────────────────────────────────
            prediction = self.model.predict(df)
            logger.info(
                f"🎯 {self.symbol} | Signal: {prediction.signal.value} | "
                f"Conf: {prediction.confidence:.0%} | "
                f"Source: {prediction.source}"
            )

            # ── Step 4: Skip if already in position ────────────────────────
            if self._current_position:
                logger.debug(f"{self.symbol} — position already open, skipping entry")
                return

            # ── Step 5: Determine trade direction ──────────────────────────
            from core.models.base_model import Signal
            if prediction.signal == Signal.HOLD:
                logger.info(f"⏸️  {self.symbol} — HOLD signal, no trade")
                return

            side = (
                OrderSide.LONG
                if prediction.signal == Signal.LONG
                else OrderSide.SHORT
            )

            # ── Step 6: Risk management ────────────────────────────────────
            balance = await self.exchange.get_balance()

            # Check if recovery trade and apply multiplier
            size_multiplier = 1.0
            if self.recovery.is_in_recovery(self.symbol):
                if self.recovery.should_open_recovery(self.symbol, prediction):
                    size_multiplier = self.recovery.get_recovery_size_multiplier(
                        self.symbol
                    )
                    logger.info(
                        f"🔄 Recovery trade: {self.symbol} | "
                        f"multiplier={size_multiplier:.2f}x"
                    )
                else:
                    logger.info(f"⏸️  {self.symbol} — recovery conditions not met")
                    return

            trade_params = self.risk.calculate_trade(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                balance=balance,
                open_trades=await self._count_open_trades(),
            )

            if not trade_params.approved:
                logger.warning(
                    f"🚫 Trade not approved: {self.symbol} — {trade_params.reject_reason}"
                )
                return

            # Apply recovery multiplier to quantity
            if size_multiplier > 1.0:
                trade_params.quantity = round(
                    trade_params.quantity * size_multiplier, 3
                )

            # ── Step 7: Execute trade ──────────────────────────────────────
            await self._execute_trade(trade_params)

        except Exception as e:
            logger.error(f"❌ Cycle error {self.symbol}: {e}", exc_info=True)

    async def _execute_trade(self, params: TradeParameters):
        """Place the order and update state."""
        if params.side == OrderSide.LONG:
            result = await self.exchange.open_long(
                symbol=params.symbol,
                quantity=params.quantity,
                leverage=params.leverage,
                take_profit=params.take_profit,
                stop_loss=params.stop_loss,
            )
        else:
            result = await self.exchange.open_short(
                symbol=params.symbol,
                quantity=params.quantity,
                leverage=params.leverage,
                take_profit=params.take_profit,
                stop_loss=params.stop_loss,
            )

        if result.success:
            self._trades_opened += 1
            logger.info(
                f"✅ Trade #{self._trades_opened} opened | "
                f"{params.symbol} {params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price} | "
                f"TP={params.take_profit} | SL={params.stop_loss}"
            )
        else:
            logger.error(f"❌ Trade failed: {result.message}")

    async def _check_existing_position(self, current_price: float):
        """Check if current position needs manual closure or has been closed by TP/SL."""
        positions = await self.exchange.get_open_positions()
        self._current_position = next(
            (p for p in positions if p.symbol == self.symbol), None
        )

        if self._current_position:
            pnl = self._current_position.unrealized_pnl
            entry = self._current_position.entry_price
            pnl_pct = ((current_price - entry) / entry) * 100
            side = self._current_position.side.value

            logger.info(
                f"📊 Open position: {self.symbol} {side.upper()} | "
                f"entry={entry} | current={current_price} | "
                f"PnL={pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
            )

            # Record if position was closed externally (TP/SL hit)
        else:
            # Position was closed (TP/SL hit or manual close)
            # We can't easily detect this here without tracking previous state
            pass

    async def _count_open_trades(self) -> int:
        """Count how many positions are currently open across all symbols."""
        positions = await self.exchange.get_open_positions()
        return len(positions)

    async def train_ml_model(self, candle_limit: int = 1000):
        """Fetch historical data and train the ML model for this symbol."""
        logger.info(f"🧠 Starting ML training for {self.symbol}...")

        candles = await self.exchange.get_ohlcv(
            self.symbol, self.TIMEFRAME, limit=candle_limit
        )
        df = ohlcv_to_dataframe(candles)
        df = add_all_indicators(df)

        self.model.train_ml(df, epochs=50)
        logger.info(f"✅ ML training complete for {self.symbol}")

    def stop(self):
        """Stop this trader gracefully."""
        self._is_active = False
        logger.info(f"🛑 FuturesTrader stopped: {self.symbol}")

    def get_stats(self) -> dict:
        return {
            "symbol":         self.symbol,
            "cycles":         self._cycles,
            "trades_opened":  self._trades_opened,
            "in_recovery":    self.recovery.is_in_recovery(self.symbol),
            "ml_trained":     self.model.ml_is_trained,
        }
