"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator — one instance per symbol.

Changes in this version
------------------------
FIX #1  Regime detector integrated (RegimeDetector.detect() each cycle).
FIX #2  HybridModel now has internal EMA smoothing — no change here.
FIX #3  Early-profit threshold raised: RANGE=0.35R  TRENDING=0.60R
        (regime-aware, pulled from RegimeParams.early_profit_r).
FIX #4  ATR passed to risk_manager.calculate_trade() → ATR-based TP/SL.
FIX #5  Portfolio direction tracker in RiskManager rejects correlated entries.
NEW     Trailing stop: once price moves 1R in profit, SL moves to breakeven.
NEW     Dynamic timeout: TRENDING regime uses 48-bar timeout (more room to run),
        RANGING uses 24 bars (mean-reverts faster).
NEW     Cooldown counter: after a position closes, symbol waits COOLDOWN_BARS
        before new entries — fixes the immediate re-entry bug from March 18.
NEW     Reversal skip: if model flips direction while we're in profit, only
        exit if the new signal is AGREE (not SPLIT).
"""

from __future__ import annotations

import asyncio
from typing import Optional

from core.exchange.base_exchange import BaseExchange, OrderSide, PositionInfo
from core.models.hybrid_model import HybridModel
from core.models.base_model import Signal
from core.risk.risk_manager import RiskManager, TradeParameters
from core.market.regime_detector import RegimeDetector, Regime, RegimeParams
from core.strategy.recovery_strategy import RecoveryStrategy
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "").replace(":BTC", "").upper()


class FuturesTrader:

    TIMEFRAME = "15m"

    # ── Timeout by regime ──────────────────────────────────────────────────
    TIMEOUT_BARS: dict[Regime, int] = {
        Regime.TRENDING:  48,   # trending: give trade more time
        Regime.RANGING:   24,   # ranging: mean-reverts quickly
        Regime.HIGH_VOL:  20,   # high vol: close sooner
    }

    # ── Cooldown after close (FIX for immediate re-entry bug) ─────────────
    COOLDOWN_BARS = 3

    # ── Trailing stop: move SL to breakeven after this many R ─────────────
    BREAKEVEN_TRIGGER_R = 1.0

    # ── Minimum bars held before early-profit exit is eligible ────────────
    MIN_HOLD_BARS = 3

    def __init__(self, exchange, symbol, risk_manager, recovery_strategy):
        self.exchange  = exchange
        self.symbol    = symbol
        self.risk      = risk_manager
        self.recovery  = recovery_strategy
        self.model     = HybridModel(symbol=symbol)
        self.regime    = RegimeDetector(symbol=symbol)

        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0

        # Position state
        self._in_position      = False
        self._tp_price:   Optional[float]     = None
        self._sl_price:   Optional[float]     = None
        self._position_side: Optional[OrderSide] = None
        self._entry_price: Optional[float]    = None
        self._bars_held        = 0
        self._breakeven_moved  = False

        # Cooldown after close
        self._cooldown_bars_remaining = 0

        # Last regime params (for dynamic timeout, early-exit threshold)
        self._last_regime_params: Optional[RegimeParams] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_my_position(self, positions):
        my_sym = _normalize_symbol(self.symbol)
        for p in positions:
            if _normalize_symbol(p.symbol) == my_sym:
                return p
        return None

    # ── Main cycle ────────────────────────────────────────────────────────

    async def run_cycle(self):
        if not self._is_active:
            return

        self._cycles += 1
        logger.info(f"Cycle #{self._cycles} — {self.symbol}")

        try:
            # ── Market data ───────────────────────────────────────────────
            candles = await self.exchange.get_ohlcv(self.symbol, self.TIMEFRAME, limit=350)
            if not candles or len(candles) < 50:
                logger.warning(f"Insufficient candles for {self.symbol}")
                return

            df            = ohlcv_to_dataframe(candles)
            df            = add_all_indicators(df)
            current_price = float(df["close"].iloc[-1])

            # ── Regime detection (FIX #1) ─────────────────────────────────
            regime_params = self.regime.detect(df)
            self._last_regime_params = regime_params

            # ── Check existing position ───────────────────────────────────
            positions   = await self.exchange.get_open_positions()
            current_pos = self._find_my_position(positions)

            if current_pos:
                self._in_position = True
                self._bars_held  += 1
                await self._monitor_position(current_pos, current_price, df, regime_params)
                return

            # Position was closed externally
            if self._in_position:
                logger.info(f"[CLOSED] {self.symbol} — ready for next trade")
                self._reset_position_state()

            # ── Cooldown (FIX for immediate re-entry bug) ─────────────────
            if self._cooldown_bars_remaining > 0:
                self._cooldown_bars_remaining -= 1
                logger.info(f"[COOLDOWN] {self.symbol} — {self._cooldown_bars_remaining} bars remaining")
                return

            # ── Signal ────────────────────────────────────────────────────
            prediction = self.model.predict(df)
            logger.info(
                f"[SIGNAL] {self.symbol} → {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol}")
                return

            # Effective confidence threshold (regime-adjusted)
            eff_threshold = min(
                0.95,
                settings.model.confidence_threshold + regime_params.conf_thr_delta,
            )
            if prediction.confidence < eff_threshold:
                logger.info(
                    f"[SKIP] {self.symbol} — conf {prediction.confidence:.0%} "
                    f"< threshold {eff_threshold:.0%}"
                )
                return

            # AGREE-only filter when regime requires it
            agreement = prediction.reasoning.split("]")[0].replace("[HYBRID ", "") if "]" in prediction.reasoning else ""
            if regime_params.require_agree and "AGREE" not in agreement:
                logger.info(
                    f"[SKIP:SPLIT] {self.symbol} — regime requires AGREE. "
                    f"Signal={prediction.signal.value} conf={prediction.confidence:.0%}"
                )
                return

            side = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT

            # ── Risk checks ───────────────────────────────────────────────
            balance    = await self.exchange.get_balance()
            total_open = len(positions)

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — trading paused")
                return

            # Get ATR for dynamic TP/SL (FIX #4)
            atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else None

            trade_params = self.risk.calculate_trade(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                balance=balance,
                open_trades=total_open,
                atr=atr,
                sl_atr_mult=regime_params.sl_mult,
                tp_atr_mult=regime_params.tp_mult * 1.5,   # TP further than SL
                size_scale=regime_params.size_scale,
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

    # ── Position monitoring ───────────────────────────────────────────────

    async def _monitor_position(
        self,
        pos: PositionInfo,
        current_price: float,
        df,
        regime_params: RegimeParams,
    ):
        """Monitor TP/SL, trailing stop, early profit, timeout, reversals."""
        if not (self._tp_price and self._sl_price and self._entry_price):
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} | bars={self._bars_held}"
            )
            return

        entry = self._entry_price
        side  = pos.side

        # ── Compute R ────────────────────────────────────────────────────
        sl_dist   = abs(entry - self._sl_price)
        if sl_dist == 0:
            sl_dist = 1.0  # guard against divide-by-zero

        if side == OrderSide.LONG:
            current_r = (current_price - entry) / sl_dist
        else:
            current_r = (entry - current_price) / sl_dist

        logger.info(
            f"[WATCHING] {self.symbol} {side.value.upper()} | "
            f"entry={entry:.4f} now={current_price:.4f} | "
            f"TP={self._tp_price:.4f}  SL={self._sl_price:.4f} | "
            f"R={current_r:+.2f}  PnL={pos.unrealized_pnl:+.4f} | "
            f"bars={self._bars_held}/{self._get_timeout(regime_params)}"
        )

        # ── TP hit ────────────────────────────────────────────────────────
        tp_hit = (
            current_price >= self._tp_price if side == OrderSide.LONG
            else current_price <= self._tp_price
        )
        if tp_hit:
            logger.info(f"[TP HIT] {self.symbol} — closing for PROFIT")
            await self._close_position("TP", pos.unrealized_pnl)
            return

        # ── SL hit ────────────────────────────────────────────────────────
        sl_hit = (
            current_price <= self._sl_price if side == OrderSide.LONG
            else current_price >= self._sl_price
        )
        if sl_hit:
            logger.info(f"[SL HIT] {self.symbol} — limiting loss")
            await self._close_position("SL", pos.unrealized_pnl)
            return

        # ── Breakeven trailing stop (NEW) ─────────────────────────────────
        # Once trade reaches BREAKEVEN_TRIGGER_R, move SL to entry
        if not self._breakeven_moved and current_r >= self.BREAKEVEN_TRIGGER_R:
            if side == OrderSide.LONG:
                new_sl = entry + 0.0001   # just above entry
            else:
                new_sl = entry - 0.0001
            if abs(new_sl - self._sl_price) > 0.0001:
                logger.info(
                    f"[BREAKEVEN] {self.symbol} — moving SL from "
                    f"{self._sl_price:.4f} to {new_sl:.4f} (R={current_r:+.2f})"
                )
                self._sl_price        = new_sl
                self._breakeven_moved = True

        # ── Early profit exit (FIX #3) ────────────────────────────────────
        early_r = regime_params.early_profit_r
        if (
            self._bars_held >= self.MIN_HOLD_BARS
            and current_r >= early_r
        ):
            logger.info(
                f"[EARLY PROFIT] {self.symbol} — R={current_r:+.2f} >= {early_r:.2f}R | "
                f"PnL={pos.unrealized_pnl:+.4f} USDT. Closing."
            )
            await self._close_position("EARLY_PROFIT", pos.unrealized_pnl)
            return

        # ── Reversal signal check (signal-degradation guard) ─────────────
        # If model flips to OPPOSITE direction, only exit if AGREE (not SPLIT)
        prediction = self.model.predict(df)
        opposite = (
            prediction.signal == Signal.SHORT if side == OrderSide.LONG
            else prediction.signal == Signal.LONG
        )
        if opposite and self._bars_held >= self.MIN_HOLD_BARS:
            agree_in_reasoning = "AGREE" in prediction.reasoning
            if agree_in_reasoning and current_r > 0:
                logger.info(
                    f"[REVERSAL EXIT] {self.symbol} — model flipped to "
                    f"{prediction.signal.value} AGREE while in profit (R={current_r:+.2f})"
                )
                await self._close_position("REVERSAL", pos.unrealized_pnl)
                return
            else:
                logger.info(
                    f"[REVERSAL SKIPPED] {self.symbol} — model flipped to "
                    f"{prediction.signal.value} but SPLIT (not AGREE). Holding."
                )

        # ── Dynamic timeout (NEW) ─────────────────────────────────────────
        max_bars = self._get_timeout(regime_params)
        if self._bars_held >= max_bars:
            logger.info(
                f"[TIMEOUT] {self.symbol} — {max_bars} bars, closing. "
                f"Regime={regime_params.regime.value}"
            )
            await self._close_position("TIMEOUT", pos.unrealized_pnl)
            return

    def _get_timeout(self, regime_params: RegimeParams) -> int:
        return self.TIMEOUT_BARS.get(regime_params.regime, 32)

    # ── Trade execution ───────────────────────────────────────────────────

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
            self._in_position     = True
            self._tp_price        = params.take_profit
            self._sl_price        = params.stop_loss
            self._position_side   = params.side
            self._entry_price     = result.price
            self._bars_held       = 0
            self._breakeven_moved = False

            # Register in portfolio direction tracker
            self.risk.portfolio.register_open(self.symbol, params.side)

            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price:.4f} | "
                f"TP={params.take_profit:.4f} | SL={params.stop_loss:.4f}"
            )
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")

    async def _close_position(self, reason: str, pnl: float):
        result = await self.exchange.close_position(self.symbol)
        if result.success:
            if pnl < 0:
                self.recovery.record_loss(self.symbol, pnl)
                self.risk.record_loss(pnl)
            else:
                self.recovery.record_profit(self.symbol, pnl)
                self.risk.record_profit(pnl)
            logger.info(f"[CLOSED:{reason}] {self.symbol} | PnL={pnl:+.4f} USDT | bars={self._bars_held}")
        else:
            logger.error(f"[CLOSE FAILED] {self.symbol}: {result.message}")

        # Deregister from portfolio tracker
        self.risk.portfolio.register_close(self.symbol)

        self._reset_position_state()
        # Start cooldown to prevent immediate re-entry
        self._cooldown_bars_remaining = self.COOLDOWN_BARS

    # ── State management ──────────────────────────────────────────────────

    def _reset_position_state(self):
        self._in_position     = False
        self._tp_price        = None
        self._sl_price        = None
        self._position_side   = None
        self._entry_price     = None
        self._bars_held       = 0
        self._breakeven_moved = False

    # ── Utilities ─────────────────────────────────────────────────────────

    async def train_ml_model(self, candle_limit: int = 1000):
        candles = await self.exchange.get_ohlcv(self.symbol, self.TIMEFRAME, limit=candle_limit)
        df = ohlcv_to_dataframe(candles)
        df = add_all_indicators(df)
        self.model.train_ml(df, epochs=50)

    def stop(self):
        self._is_active = False
        logger.info(f"Trader stopped: {self.symbol}")

    def get_stats(self) -> dict:
        regime_name = (
            self._last_regime_params.regime.value
            if self._last_regime_params else "UNKNOWN"
        )
        return {
            "symbol":        self.symbol,
            "cycles":        self._cycles,
            "trades_opened": self._trades_opened,
            "in_position":   self._in_position,
            "tp":            self._tp_price,
            "sl":            self._sl_price,
            "entry":         self._entry_price,
            "bars_held":     self._bars_held,
            "regime":        regime_name,
            "in_recovery":   self.recovery.is_in_recovery(self.symbol),
            "ml_trained":    self.model.ml_is_trained,
        }
