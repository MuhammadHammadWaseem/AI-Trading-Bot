"""
core/trader/futures_trader.py
------------------------------
Main trading orchestrator. 
Fixed: symbol format matching, TP/SL monitoring, position display.
"""

import asyncio
from typing import Optional

from core.exchange.base_exchange import BaseExchange, OrderSide, PositionInfo
from core.models.hybrid_model import HybridModel
from core.models.base_model import Signal
from core.risk.risk_manager import RiskManager, TradeParameters
from core.strategy.recovery_strategy import RecoveryStrategy
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.logger import get_logger

logger = get_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    """Normalize symbol — remove /: separators for comparison."""
    return symbol.replace("/", "").replace(":USDT", "").replace(":BTC", "").upper()


class FuturesTrader:

    TIMEFRAME = "15m"

    def __init__(self, exchange, symbol, risk_manager, recovery_strategy):
        self.exchange   = exchange
        self.symbol     = symbol
        self.risk       = risk_manager
        self.recovery   = recovery_strategy
        self.model      = HybridModel(symbol=symbol)

        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0
        self._in_position   = False
        self._tp_price: Optional[float] = None
        self._sl_price: Optional[float] = None
        self._position_side: Optional[OrderSide] = None
        self._entry_price: Optional[float] = None

    def _find_my_position(self, positions):
        """Find position matching this symbol — handles different format variants."""
        my_sym = _normalize_symbol(self.symbol)
        for p in positions:
            if _normalize_symbol(p.symbol) == my_sym:
                return p
        return None

    async def run_cycle(self):
        if not self._is_active:
            return

        self._cycles += 1
        logger.info(f"Cycle #{self._cycles} — {self.symbol}")

        try:
            # Fetch market data
            candles = await self.exchange.get_ohlcv(self.symbol, self.TIMEFRAME, limit=300)
            if not candles or len(candles) < 50:
                logger.warning(f"Insufficient candles for {self.symbol}")
                return

            df            = ohlcv_to_dataframe(candles)
            df            = add_all_indicators(df)
            current_price = float(df["close"].iloc[-1])

            # Check existing position
            positions    = await self.exchange.get_open_positions()
            current_pos  = self._find_my_position(positions)

            if current_pos:
                self._in_position = True
                await self._monitor_tpsl(current_pos, current_price)
                return

            # Position was closed
            if self._in_position:
                logger.info(f"[CLOSED] {self.symbol} position closed — ready for next trade")
                self._reset_position_state()

            # Get AI prediction
            prediction = self.model.predict(df)
            logger.info(
                f"[SIGNAL] {self.symbol} -> {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol} — skipping cycle")
                return

            if prediction.confidence < 0.65:
                logger.info(f"[SKIP] {self.symbol} — conf {prediction.confidence:.0%} < 65%")
                return

            side = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT

            balance    = await self.exchange.get_balance()
            total_open = len(positions)

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — trading paused")
                return

            trade_params = self.risk.calculate_trade(
                symbol=self.symbol, side=side,
                entry_price=current_price,
                balance=balance, open_trades=total_open,
            )

            if not trade_params.approved:
                logger.warning(f"[REJECTED] {self.symbol} — {trade_params.reject_reason}")
                return

            if self.recovery.is_in_recovery(self.symbol):
                if self.recovery.should_open_recovery(self.symbol, prediction):
                    mult = self.recovery.get_recovery_size_multiplier(self.symbol)
                    trade_params.quantity = round(trade_params.quantity * mult, 3)
                    logger.info(f"[RECOVERY] {self.symbol} size x{mult:.1f}")
                else:
                    logger.info(f"[RECOVERY WAIT] {self.symbol}")
                    return

            await self._execute_trade(trade_params)

        except Exception as e:
            logger.error(f"Cycle error {self.symbol}: {e}", exc_info=True)

    async def _monitor_tpsl(self, pos: PositionInfo, current_price: float):
        """Monitor TP/SL manually each cycle."""
        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        if pos.side == OrderSide.SHORT:
            pnl_pct = -pnl_pct

        if self._tp_price and self._sl_price:
            tp_hit = (current_price >= self._tp_price) if pos.side == OrderSide.LONG else (current_price <= self._tp_price)
            sl_hit = (current_price <= self._sl_price) if pos.side == OrderSide.LONG else (current_price >= self._sl_price)

            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.2f} now={current_price:.2f} | "
                f"TP={self._tp_price:.2f} SL={self._sl_price:.2f} | "
                f"PnL={pos.unrealized_pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
            )

            if tp_hit:
                logger.info(f"[TP HIT] {self.symbol} — closing for PROFIT!")
                result = await self.exchange.close_position(self.symbol)
                if result.success:
                    self.recovery.record_profit(self.symbol, pos.unrealized_pnl)
                    self.risk.record_profit(pos.unrealized_pnl)
                self._reset_position_state()
                return

            if sl_hit:
                logger.info(f"[SL HIT] {self.symbol} — closing to limit LOSS")
                result = await self.exchange.close_position(self.symbol)
                if result.success:
                    self.recovery.record_loss(self.symbol, pos.unrealized_pnl)
                    self.risk.record_loss(pos.unrealized_pnl)
                self._reset_position_state()
                return
        else:
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.2f} now={current_price:.2f} | "
                f"PnL={pos.unrealized_pnl:+.4f} USDT ({pnl_pct:+.2f}%)"
            )

    async def _execute_trade(self, params: TradeParameters):
        if params.side == OrderSide.LONG:
            result = await self.exchange.open_long(
                symbol=params.symbol, quantity=params.quantity,
                leverage=params.leverage,
                take_profit=params.take_profit, stop_loss=params.stop_loss,
            )
        else:
            result = await self.exchange.open_short(
                symbol=params.symbol, quantity=params.quantity,
                leverage=params.leverage,
                take_profit=params.take_profit, stop_loss=params.stop_loss,
            )

        if result.success:
            self._trades_opened += 1
            self._in_position   = True
            self._tp_price      = params.take_profit
            self._sl_price      = params.stop_loss
            self._position_side = params.side
            self._entry_price   = result.price
            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price:.2f} | "
                f"TP={params.take_profit:.2f} | SL={params.stop_loss:.2f}"
            )
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")

    def _reset_position_state(self):
        self._in_position   = False
        self._tp_price      = None
        self._sl_price      = None
        self._position_side = None
        self._entry_price   = None

    async def train_ml_model(self, candle_limit: int = 1000):
        candles = await self.exchange.get_ohlcv(self.symbol, self.TIMEFRAME, limit=candle_limit)
        df = ohlcv_to_dataframe(candles)
        df = add_all_indicators(df)
        self.model.train_ml(df, epochs=50)

    def stop(self):
        self._is_active = False
        logger.info(f"Trader stopped: {self.symbol}")

    def get_stats(self) -> dict:
        return {
            "symbol":        self.symbol,
            "cycles":        self._cycles,
            "trades_opened": self._trades_opened,
            "in_position":   self._in_position,
            "tp":            self._tp_price,
            "sl":            self._sl_price,
            "entry":         self._entry_price,
            "in_recovery":   self.recovery.is_in_recovery(self.symbol),
            "ml_trained":    self.model.ml_is_trained,
        }