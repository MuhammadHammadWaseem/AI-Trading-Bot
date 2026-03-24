"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator — one instance per symbol.

Changes in this version (v4)
------------------------------
TIMEFRAME   : Changed from 15m to 5m — more responsive data, new candle
              every 5 minutes instead of 15.
CYCLE       : 60-second cycles. The bot checks every 60s whether a new
              5m candle has closed and if the signal has changed.
CANDLE GATE : Only re-evaluates signal when a NEW 5m candle has closed
              (tracked by last_candle_ts). Prevents re-reading same partial
              candle 5 times per cycle.
VOLATILITY  : ATR_ratio gate still active — skips entry if ATR below average.
PARTIAL TP  : Close 50% at 1.0R, move SL to breakeven on remainder.
RECALIBRATOR: SignalRecalibrator tracks per-direction win rates and adjusts
              confidence thresholds dynamically. No model retraining needed.
              Trade outcomes feed into the recalibrator on every close.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from core.exchange.base_exchange import BaseExchange, OrderSide, PositionInfo
from core.models.hybrid_model import HybridModel
from core.models.base_model import Signal
from core.models.signal_recalibrator import SignalRecalibrator
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

    # Changed: 15m -> 5m for more responsive signals with 60s cycles
    TIMEFRAME = "5m"

    # ── Timeout by regime (in 5m bars) ────────────────────────────────────
    # 5m bars: TRENDING=24 (2h), RANGING=12 (1h), HIGH_VOL=8 (40min)
    TIMEOUT_BARS: dict[Regime, int] = {
        Regime.TRENDING:  24,
        Regime.RANGING:   12,
        Regime.HIGH_VOL:   8,
    }

    # ── Cooldown after close (in 5m bars) ─────────────────────────────────
    COOLDOWN_BARS = 3

    # ── Trailing stop / partial TP ─────────────────────────────────────────
    BREAKEVEN_TRIGGER_R  = 1.0
    PARTIAL_TP_R         = 1.0
    PARTIAL_TP_FRACTION  = 0.5
    MIN_HOLD_BARS        = 2    # lower for 5m (was 3 for 15m)

    # ── Volatility gate ────────────────────────────────────────────────────
    ATR_RATIO_MIN = 1.0

    def __init__(self, exchange, symbol, risk_manager, recovery_strategy,
                 recalibrator: Optional[SignalRecalibrator] = None):
        self.exchange      = exchange
        self.symbol        = symbol
        self.risk          = risk_manager
        self.recovery      = recovery_strategy
        self.model         = HybridModel(symbol=symbol)
        self.regime        = RegimeDetector(symbol=symbol)
        self.recalibrator  = recalibrator or SignalRecalibrator()

        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0

        # Position state
        self._in_position        = False
        self._tp_price:   Optional[float]        = None
        self._sl_price:   Optional[float]        = None
        self._position_side: Optional[OrderSide] = None
        self._entry_price: Optional[float]       = None
        self._entry_confidence: float            = 0.0
        self._bars_held          = 0
        self._breakeven_moved    = False
        self._partial_exit_taken = False

        # Cooldown after close
        self._cooldown_bars_remaining = 0

        # Candle gate: only re-evaluate when a new 5m candle has closed
        # Stored as the timestamp (ms) of the last candle open we evaluated
        self._last_candle_ts: Optional[int] = None

        # Last regime for stats / recalibrator
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
            candles = await self.exchange.get_ohlcv(self.symbol, self.TIMEFRAME, limit=200)
            if not candles or len(candles) < 50:
                logger.warning(f"Insufficient candles for {self.symbol}")
                return

            df            = ohlcv_to_dataframe(candles)
            df            = add_all_indicators(df)
            current_price = float(df["close"].iloc[-1])

            # ── Regime detection ──────────────────────────────────────────
            regime_params = self.regime.detect(df)
            self._last_regime_params = regime_params

            # ── Check existing position ───────────────────────────────────
            positions   = await self.exchange.get_open_positions()
            current_pos = self._find_my_position(positions)

            if current_pos:
                self._in_position = True
                # Count bars using candle closes, not wall-clock cycles
                current_candle_ts = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], 'timestamp') else self._cycles
                if self._last_candle_ts is None or current_candle_ts != self._last_candle_ts:
                    self._bars_held      += 1
                    self._last_candle_ts  = current_candle_ts
                await self._monitor_position(current_pos, current_price, df, regime_params)
                return

            # Position was closed externally
            if self._in_position:
                logger.info(f"[CLOSED] {self.symbol} — ready for next trade")
                self._reset_position_state()

            # ── Cooldown ──────────────────────────────────────────────────
            if self._cooldown_bars_remaining > 0:
                # Only decrement on new candle
                current_candle_ts = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], 'timestamp') else self._cycles
                if self._last_candle_ts is None or current_candle_ts != self._last_candle_ts:
                    self._cooldown_bars_remaining -= 1
                    self._last_candle_ts = current_candle_ts
                logger.info(f"[COOLDOWN] {self.symbol} — {self._cooldown_bars_remaining} bars remaining")
                return

            # ── Candle gate ───────────────────────────────────────────────
            # Only evaluate a new signal when a new 5m candle has closed.
            # Between candle closes, the signal cannot meaningfully change.
            current_candle_ts = int(df.index[-1].timestamp() * 1000) if hasattr(df.index[-1], 'timestamp') else None
            if current_candle_ts is not None and current_candle_ts == self._last_candle_ts:
                logger.info(f"[CANDLE GATE] {self.symbol} — no new 5m candle, skipping signal eval")
                return
            self._last_candle_ts = current_candle_ts

            # ── Volatility gate ───────────────────────────────────────────
            if "atr" in df.columns:
                atr_series = df["atr"].dropna()
                if len(atr_series) >= 20:
                    atr_current = float(atr_series.iloc[-1])
                    atr_mean    = float(atr_series.rolling(20).mean().iloc[-1])
                    atr_ratio   = atr_current / atr_mean if atr_mean > 0 else 1.0
                    if atr_ratio < self.ATR_RATIO_MIN:
                        logger.info(
                            f"[SKIP:LOW_VOL] {self.symbol} — "
                            f"ATR_ratio={atr_ratio:.2f} < {self.ATR_RATIO_MIN:.1f}"
                        )
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

            side = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT

            # ── Effective threshold (regime + recalibrator adjustments) ───
            # Regime raises/lowers threshold based on market structure.
            # Recalibrator raises/lowers further based on LIVE win rate.
            base_threshold    = settings.model.confidence_threshold
            regime_adj        = regime_params.conf_thr_delta
            recalib_adj       = self.recalibrator.get_threshold_adjustment(
                                    self.symbol, side.value) / 100.0  # convert pp to decimal
            eff_threshold     = min(0.95, base_threshold + regime_adj + recalib_adj)

            if prediction.confidence < eff_threshold:
                logger.info(
                    f"[SKIP] {self.symbol} — conf {prediction.confidence:.0%} "
                    f"< threshold {eff_threshold:.0%} "
                    f"(base={base_threshold:.0%} regime={regime_adj:+.0%} recalib={recalib_adj:+.0%})"
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

            # ── Risk checks ───────────────────────────────────────────────
            balance    = await self.exchange.get_balance()
            total_open = len(positions)

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — trading paused")
                return

            atr = float(df["atr"].iloc[-1]) if "atr" in df.columns else None

            trade_params = self.risk.calculate_trade(
                symbol=self.symbol,
                side=side,
                entry_price=current_price,
                balance=balance,
                open_trades=total_open,
                atr=atr,
                sl_atr_mult=regime_params.sl_mult,
                tp_atr_mult=regime_params.tp_mult * 1.5,
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

            self._entry_confidence = prediction.confidence
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
        if not (self._tp_price and self._sl_price and self._entry_price):
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} | bars={self._bars_held}"
            )
            return

        entry = self._entry_price
        side  = pos.side

        sl_dist = abs(entry - self._sl_price)
        if sl_dist == 0:
            sl_dist = 1.0

        current_r = (
            (current_price - entry) / sl_dist if side == OrderSide.LONG
            else (entry - current_price) / sl_dist
        )

        logger.info(
            f"[WATCHING] {self.symbol} {side.value.upper()} | "
            f"entry={entry:.4f} now={current_price:.4f} | "
            f"TP={self._tp_price:.4f}  SL={self._sl_price:.4f} | "
            f"R={current_r:+.2f}  PnL={pos.unrealized_pnl:+.4f} | "
            f"bars={self._bars_held}/{self._get_timeout(regime_params)}"
        )

        # ── TP hit ────────────────────────────────────────────────────────
        tp_hit = (current_price >= self._tp_price if side == OrderSide.LONG
                  else current_price <= self._tp_price)
        if tp_hit:
            logger.info(f"[TP HIT] {self.symbol}")
            await self._close_position("TP", pos.unrealized_pnl, current_r, regime_params)
            return

        # ── SL hit ────────────────────────────────────────────────────────
        sl_hit = (current_price <= self._sl_price if side == OrderSide.LONG
                  else current_price >= self._sl_price)
        if sl_hit:
            logger.info(f"[SL HIT] {self.symbol}")
            await self._close_position("SL", pos.unrealized_pnl, current_r, regime_params)
            return

        # ── Partial TP ────────────────────────────────────────────────────
        if (
            not self._partial_exit_taken
            and self._bars_held >= self.MIN_HOLD_BARS
            and current_r >= self.PARTIAL_TP_R
        ):
            partial_qty = round(pos.quantity * self.PARTIAL_TP_FRACTION, 3)
            if partial_qty > 0:
                logger.info(
                    f"[PARTIAL TP] {self.symbol} — R={current_r:+.2f} | "
                    f"closing {partial_qty}/{pos.quantity} | "
                    f"PnL={pos.unrealized_pnl:+.4f}"
                )
                result = await self.exchange.close_position_partial(
                    symbol=self.symbol, quantity=partial_qty
                )
                if result and result.success:
                    self._sl_price           = entry + 0.0001 if side == OrderSide.LONG else entry - 0.0001
                    self._partial_exit_taken = True
                    self._breakeven_moved    = True
                    logger.info(f"[PARTIAL TP OK] {self.symbol} SL → breakeven {self._sl_price:.4f}")
                else:
                    logger.info(f"[PARTIAL TP FALLBACK] {self.symbol} — full close at R={current_r:+.2f}")
                    await self._close_position("PARTIAL_TP_FULL", pos.unrealized_pnl, current_r, regime_params)
                    return

        # ── Breakeven trailing stop ───────────────────────────────────────
        if not self._breakeven_moved and current_r >= self.BREAKEVEN_TRIGGER_R:
            new_sl = entry + 0.0001 if side == OrderSide.LONG else entry - 0.0001
            if abs(new_sl - self._sl_price) > 0.0001:
                logger.info(f"[BREAKEVEN] {self.symbol} SL {self._sl_price:.4f} → {new_sl:.4f}")
                self._sl_price        = new_sl
                self._breakeven_moved = True

        # ── Early profit exit ─────────────────────────────────────────────
        early_r = regime_params.early_profit_r
        if (
            self._bars_held >= self.MIN_HOLD_BARS
            and current_r >= early_r
            and not self._partial_exit_taken
        ):
            logger.info(f"[EARLY PROFIT] {self.symbol} R={current_r:+.2f} >= {early_r:.2f}R")
            await self._close_position("EARLY_PROFIT", pos.unrealized_pnl, current_r, regime_params)
            return

        # ── Reversal check ────────────────────────────────────────────────
        prediction = self.model.predict(df)
        opposite = (prediction.signal == Signal.SHORT if side == OrderSide.LONG
                    else prediction.signal == Signal.LONG)
        if opposite and self._bars_held >= self.MIN_HOLD_BARS:
            if "AGREE" in prediction.reasoning and current_r > 0:
                logger.info(f"[REVERSAL EXIT] {self.symbol} R={current_r:+.2f}")
                await self._close_position("REVERSAL", pos.unrealized_pnl, current_r, regime_params)
                return
            else:
                logger.info(f"[REVERSAL SKIPPED] {self.symbol} SPLIT or not in profit")

        # ── Timeout ───────────────────────────────────────────────────────
        max_bars = self._get_timeout(regime_params)
        if self._bars_held >= max_bars:
            logger.info(f"[TIMEOUT] {self.symbol} {max_bars} bars")
            await self._close_position("TIMEOUT", pos.unrealized_pnl, current_r, regime_params)

    def _get_timeout(self, regime_params: RegimeParams) -> int:
        return self.TIMEOUT_BARS.get(regime_params.regime, 16)

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
            self._trades_opened      += 1
            self._in_position         = True
            self._tp_price            = params.take_profit
            self._sl_price            = params.stop_loss
            self._position_side       = params.side
            self._entry_price         = result.price
            self._bars_held           = 0
            self._breakeven_moved     = False
            self._partial_exit_taken  = False
            self.risk.portfolio.register_open(self.symbol, params.side)
            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price:.4f} | "
                f"TP={params.take_profit:.4f} | SL={params.stop_loss:.4f}"
            )
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")

    async def _close_position(
        self,
        reason:        str,
        pnl:           float,
        pnl_r:         float = 0.0,
        regime_params: Optional[RegimeParams] = None,
    ):
        result = await self.exchange.close_position(self.symbol)
        if result.success:
            if pnl < 0:
                self.recovery.record_loss(self.symbol, pnl)
                self.risk.record_loss(pnl)
            else:
                self.recovery.record_profit(self.symbol, pnl)
                self.risk.record_profit(pnl)
            logger.info(
                f"[CLOSED:{reason}] {self.symbol} | "
                f"PnL={pnl:+.4f} USDT | R={pnl_r:+.2f} | bars={self._bars_held}"
            )

            # Feed outcome into recalibrator
            regime_name = (regime_params.regime.value
                           if regime_params else "UNKNOWN")
            self.recalibrator.record_trade(
                symbol=self.symbol,
                side=self._position_side.value if self._position_side else "unknown",
                won=(pnl >= 0),
                confidence=self._entry_confidence,
                pnl_usdt=pnl,
                pnl_r=pnl_r,
                bars_held=self._bars_held,
                exit_reason=reason,
                regime=regime_name,
            )
        else:
            logger.error(f"[CLOSE FAILED] {self.symbol}: {result.message}")

        self.risk.portfolio.register_close(self.symbol, profit=(pnl >= 0))
        self._reset_position_state()
        self._cooldown_bars_remaining = self.COOLDOWN_BARS

    # ── State management ──────────────────────────────────────────────────

    def _reset_position_state(self):
        self._in_position        = False
        self._tp_price           = None
        self._sl_price           = None
        self._position_side      = None
        self._entry_price        = None
        self._entry_confidence   = 0.0
        self._bars_held          = 0
        self._breakeven_moved    = False
        self._partial_exit_taken = False
        self._last_candle_ts     = None   # reset so first cycle after cooldown re-evaluates

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
        recalib = self.recalibrator.get_summary(self.symbol)
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
            "partial_taken": self._partial_exit_taken,
            "recalib":       recalib,
        }
