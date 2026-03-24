"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator — one instance per symbol.

v7 Changes (production-grade state reconciliation)
────────────────────────────────────────────────────
RECONCILIATION : Every cycle, before monitoring, the bot fetches live
                 positions from Binance and compares against internal state.

                 Three cases handled:
                  A. Bot=open, Exchange=open  → normal monitoring.
                  B. Bot=open, Exchange=none  → external close detected
                     (SL/TP hit or liquidation). Estimates PnL from
                     last-known unrealized_pnl, records trade, resets
                     state cleanly, starts cooldown.
                  C. Bot=closed, Exchange=open → rogue position. Warns
                     loudly but does NOT manage it to avoid double-management.
                  D. Both=closed → normal idle, continue to signal eval.

PERIODIC SYNC  : Every FULL_SYNC_CYCLES (default 60 ≈ 5 min), a deep sync
                 re-validates internal entry_price against exchange data and
                 confirms SL/TP are still set.

SL/TP GUARD    : After order placement, SL/TP are verified non-None. Missing
                 values are logged as WARNING — position still lives, managed
                 by timeout only.

v6 Changes (filter calibration + execution frequency)
───────────────────────────────────────────────────────
CANDLE GATE   : Hybrid 1m/60s timing gate replaces pure 5m gate.
ATR GATE      : Relaxed from 1.0 to 0.90.
AGREE FILTER  : TRENDING regime only.
THRESHOLD     : Base 0.55. Effective range 0.52–0.65 by regime.
SPLIT TRADES  : Allowed >= SPLIT_MIN_CONF (0.52) in non-TRENDING regimes.
RECALIBRATOR  : Live win-rate tracker, capped at ±0.08pp.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import pandas as pd

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

    # ── Timeframes ────────────────────────────────────────────────────────
    TIMEFRAME     = "5m"
    SIG_TIMEFRAME = "1m"

    # ── Execution timing ──────────────────────────────────────────────────
    SIGNAL_EVAL_SECONDS = 60
    FAST_ENTRY_CONF     = 0.72

    # ── Position monitoring ────────────────────────────────────────────────
    TIMEOUT_BARS: dict = {
        Regime.TRENDING:  24,
        Regime.RANGING:   12,
        Regime.HIGH_VOL:   8,
    }
    COOLDOWN_BARS = 3

    # ── Partial TP / trailing ─────────────────────────────────────────────
    BREAKEVEN_TRIGGER_R  = 1.0
    PARTIAL_TP_R         = 1.0
    PARTIAL_TP_FRACTION  = 0.5
    MIN_HOLD_BARS        = 2

    # ── Volatility gate ───────────────────────────────────────────────────
    ATR_RATIO_MIN = 0.90

    # ── Confidence ────────────────────────────────────────────────────────
    BASE_THRESHOLD = 0.55
    SPLIT_MIN_CONF = 0.52
    RECALIB_CAP    = 0.08

    # ── Reconciliation ────────────────────────────────────────────────────
    FULL_SYNC_CYCLES = 60   # deep sync every ~5 min at 5s cycle

    def __init__(self, exchange, symbol, risk_manager, recovery_strategy,
                 recalibrator: Optional[SignalRecalibrator] = None):
        self.exchange     = exchange
        self.symbol       = symbol
        self.risk         = risk_manager
        self.recovery     = recovery_strategy
        self.model        = HybridModel(symbol=symbol)
        self.regime       = RegimeDetector(symbol=symbol)
        self.recalibrator = recalibrator or SignalRecalibrator()

        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0

        # Position state
        self._in_position        = False
        self._tp_price:   Optional[float]        = None
        self._sl_price:   Optional[float]        = None
        self._position_side: Optional[OrderSide] = None
        self._entry_price:  Optional[float]      = None
        self._entry_confidence: float            = 0.0
        self._bars_held          = 0
        self._breakeven_moved    = False
        self._partial_exit_taken = False

        # Last-seen unrealized PnL — used to estimate PnL on external close
        self._last_known_pnl: float = 0.0

        # Cooldown
        self._cooldown_bars_remaining = 0

        # Signal timing
        self._last_1m_candle_ts: Optional[int]   = None
        self._last_5m_candle_ts: Optional[int]   = None
        self._last_eval_time:    float            = 0.0
        self._last_regime_params: Optional[RegimeParams] = None

    # ── Helpers ───────────────────────────────────────────────────────────

    def _find_my_position(self, positions):
        my_sym = _normalize_symbol(self.symbol)
        for p in positions:
            if _normalize_symbol(p.symbol) == my_sym:
                return p
        return None

    def _effective_threshold(self, regime_params: RegimeParams, side: OrderSide) -> float:
        recalib_adj = self.recalibrator.get_threshold_adjustment(
            self.symbol, side.value) / 100.0
        recalib_adj = max(-self.RECALIB_CAP, min(self.RECALIB_CAP, recalib_adj))
        eff = self.BASE_THRESHOLD + regime_params.conf_thr_delta + recalib_adj
        return min(0.85, max(0.45, eff))

    def _should_evaluate_signal(self, df_1m, force: bool = False) -> bool:
        if force:
            return True
        now = time.monotonic()
        if now - self._last_eval_time >= self.SIGNAL_EVAL_SECONDS:
            return True
        if df_1m is not None and len(df_1m) > 0:
            ts_1m = (int(df_1m.index[-1].timestamp() * 1000)
                     if hasattr(df_1m.index[-1], 'timestamp') else None)
            if ts_1m is not None and ts_1m != self._last_1m_candle_ts:
                self._last_1m_candle_ts = ts_1m
                return True
        return False

    # ── Reconciliation ────────────────────────────────────────────────────

    async def _reconcile_position_state(
        self,
        current_pos:  Optional[PositionInfo],
        current_price: float,
        regime_params: Optional[RegimeParams],
        new_5m_candle: bool,
    ) -> bool:
        """
        Single source of truth for position state. Called every cycle
        before any monitoring or signal logic.

        Returns True  → position is confirmed open on exchange.
        Returns False → no open position; proceed to signal evaluation.
        """
        exchange_has_pos = current_pos is not None
        bot_thinks_open  = self._in_position

        # ── Case A: both agree position is open ───────────────────────────
        if bot_thinks_open and exchange_has_pos:
            if new_5m_candle:
                self._bars_held += 1
            self._last_known_pnl = current_pos.unrealized_pnl
            return True

        # ── Case B: bot thinks open, exchange says closed ─────────────────
        if bot_thinks_open and not exchange_has_pos:
            estimated_pnl = self._last_known_pnl

            # Best-guess exit reason
            exit_reason = "EXTERNALLY_CLOSED"
            if self._sl_price and self._tp_price and self._entry_price:
                if self._position_side == OrderSide.LONG:
                    if current_price >= self._tp_price:
                        exit_reason = "TP_EXTERNAL"
                    elif current_price <= self._sl_price:
                        exit_reason = "SL_EXTERNAL"
                else:
                    if current_price <= self._tp_price:
                        exit_reason = "TP_EXTERNAL"
                    elif current_price >= self._sl_price:
                        exit_reason = "SL_EXTERNAL"

            logger.warning(
                f"[RECONCILE:{exit_reason}] {self.symbol} — position vanished on exchange. "
                f"estimated_pnl={estimated_pnl:+.4f} USDT | "
                f"last_price={current_price:.4f} | bars_held={self._bars_held}"
            )

            # Record to recalibrator and risk manager
            regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
            if estimated_pnl < 0:
                self.recovery.record_loss(self.symbol, estimated_pnl)
                self.risk.record_loss(estimated_pnl)
            else:
                self.recovery.record_profit(self.symbol, estimated_pnl)
                self.risk.record_profit(estimated_pnl)

            self.recalibrator.record_trade(
                symbol=self.symbol,
                side=self._position_side.value if self._position_side else "unknown",
                won=(estimated_pnl >= 0),
                confidence=self._entry_confidence,
                pnl_usdt=estimated_pnl,
                pnl_r=0.0,
                bars_held=self._bars_held,
                exit_reason=exit_reason,
                regime=regime_name,
            )

            self.risk.portfolio.register_close(self.symbol, profit=(estimated_pnl >= 0))
            self._reset_position_state()
            self._cooldown_bars_remaining = self.COOLDOWN_BARS
            return False

        # ── Case C: bot thinks closed, exchange has rogue position ────────
        if not bot_thinks_open and exchange_has_pos:
            logger.warning(
                f"[RECONCILE:ROGUE] {self.symbol} — unknown open position on exchange "
                f"(side={current_pos.side.value}, qty={current_pos.quantity}, "
                f"entry={current_pos.entry_price:.4f}). "
                f"Bot will NOT manage this. Close it manually if needed."
            )
            return False  # Do not take over management

        # ── Case D: both agree no position ────────────────────────────────
        return False

    async def _deep_sync(self, positions):
        """
        Periodic deep validation of internal state vs exchange.
        Re-anchors entry price and checks SL/TP validity.
        Runs every FULL_SYNC_CYCLES cycles.
        """
        current_pos = self._find_my_position(positions)

        if self._in_position and current_pos:
            # Re-anchor entry price if it diverged
            if (self._entry_price is None
                    or abs(self._entry_price - current_pos.entry_price) > 1.0):
                old_entry = self._entry_price
                self._entry_price = current_pos.entry_price
                logger.info(
                    f"[DEEP_SYNC] {self.symbol} — entry_price reanchored "
                    f"{old_entry} → {current_pos.entry_price:.4f}"
                )

            # Validate SL/TP are set
            if not self._sl_price or not self._tp_price:
                logger.warning(
                    f"[DEEP_SYNC] {self.symbol} — active position is missing SL/TP "
                    f"(sl={self._sl_price}, tp={self._tp_price}). "
                    f"Position will rely on timeout only."
                )

        logger.debug(
            f"[DEEP_SYNC] {self.symbol} complete | "
            f"bot_open={self._in_position} | "
            f"exchange_pos={'YES' if current_pos else 'NO'} | "
            f"bars={self._bars_held}"
        )

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

            # 1h HTF features (best-effort)
            df_1h = None
            try:
                c1h = await self.exchange.get_ohlcv(self.symbol, "1h", limit=100)
                if c1h and len(c1h) >= 20:
                    df_1h = ohlcv_to_dataframe(c1h)
            except Exception:
                pass

            # 1m signal timing gate (best-effort)
            df_1m = None
            try:
                c1m = await self.exchange.get_ohlcv(self.symbol, "1m", limit=5)
                if c1m and len(c1m) >= 2:
                    df_1m = ohlcv_to_dataframe(c1m)
            except Exception:
                pass

            # ── Regime detection ──────────────────────────────────────────
            regime_params = self.regime.detect(df)
            self._last_regime_params = regime_params

            # ── Track 5m bars ─────────────────────────────────────────────
            ts_5m = (int(df.index[-1].timestamp() * 1000)
                     if hasattr(df.index[-1], 'timestamp') else self._cycles)
            new_5m_candle = (ts_5m != self._last_5m_candle_ts)
            if new_5m_candle:
                self._last_5m_candle_ts = ts_5m

            # ── Fetch live positions (single API call per cycle) ──────────
            positions   = await self.exchange.get_open_positions()
            current_pos = self._find_my_position(positions)

            # ── Periodic deep sync ────────────────────────────────────────
            if self._cycles % self.FULL_SYNC_CYCLES == 0:
                await self._deep_sync(positions)

            # ── Position reconciliation ───────────────────────────────────
            # This is the single authoritative check. It compares bot state
            # with exchange state and heals any divergence before we proceed.
            position_is_open = await self._reconcile_position_state(
                current_pos, current_price, regime_params, new_5m_candle
            )

            if position_is_open:
                # current_pos is guaranteed non-None here (Case A)
                await self._monitor_position(current_pos, current_price, df, regime_params)
                return

            # ── No position open — check cooldown then signals ────────────

            if self._cooldown_bars_remaining > 0:
                if new_5m_candle:
                    self._cooldown_bars_remaining -= 1
                logger.info(f"[COOLDOWN] {self.symbol} — {self._cooldown_bars_remaining} bars remaining")
                return

            # ── Signal timing gate ────────────────────────────────────────
            first_cycle = (self._last_eval_time == 0.0)
            if not self._should_evaluate_signal(df_1m, force=first_cycle):
                logger.info(f"[GATE] {self.symbol} — waiting for next eval window")
                return

            # ── Volatility gate ───────────────────────────────────────────
            if "atr" in df.columns:
                atr_s = df["atr"].dropna()
                if len(atr_s) >= 20:
                    atr_cur   = float(atr_s.iloc[-1])
                    atr_mean  = float(atr_s.rolling(20).mean().iloc[-1])
                    atr_ratio = atr_cur / atr_mean if atr_mean > 0 else 1.0
                    if atr_ratio < self.ATR_RATIO_MIN:
                        logger.info(
                            f"[SKIP:LOW_VOL] {self.symbol} — "
                            f"ATR_ratio={atr_ratio:.2f} < {self.ATR_RATIO_MIN:.2f}"
                        )
                        self._last_eval_time = time.monotonic()
                        return

            # ── Signal evaluation ─────────────────────────────────────────
            prediction = self.model.predict(df, df_1h=df_1h)
            self._last_eval_time = time.monotonic()

            logger.info(
                f"[SIGNAL] {self.symbol} → {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol}")
                return

            side = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT

            # ── Effective threshold ───────────────────────────────────────
            eff_threshold = self._effective_threshold(regime_params, side)

            if prediction.confidence < eff_threshold:
                logger.info(
                    f"[SKIP:CONF] {self.symbol} — "
                    f"conf={prediction.confidence:.0%} < threshold={eff_threshold:.0%} "
                    f"(base={self.BASE_THRESHOLD:.0%} "
                    f"regime={regime_params.conf_thr_delta:+.0%})"
                )
                return

            # ── AGREE filter (TRENDING regime only) ───────────────────────
            agreement = ""
            if "]" in prediction.reasoning:
                agreement = prediction.reasoning.split("]")[0].replace("[HYBRID ", "")

            if regime_params.require_agree and "AGREE" not in agreement:
                if prediction.confidence >= eff_threshold + 0.08:
                    logger.info(
                        f"[SPLIT:STRONG] {self.symbol} — TRENDING but high conf "
                        f"{prediction.confidence:.0%}, proceeding"
                    )
                else:
                    logger.info(
                        f"[SKIP:SPLIT] {self.symbol} — TRENDING requires AGREE. "
                        f"conf={prediction.confidence:.0%}"
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
            # Missing SL/TP — still log position status but skip SL/TP logic
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} | bars={self._bars_held} "
                f"[NO SL/TP — timeout only]"
            )
            # Still enforce timeout even without SL/TP
            max_bars = self._get_timeout(regime_params)
            if self._bars_held >= max_bars:
                logger.info(f"[TIMEOUT] {self.symbol} {max_bars} bars (no SL/TP mode)")
                await self._close_position("TIMEOUT", pos.unrealized_pnl, 0.0, regime_params)
            return

        entry = self._entry_price
        side  = pos.side

        sl_dist   = abs(entry - self._sl_price) or 1.0
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
        if (current_price >= self._tp_price if side == OrderSide.LONG
                else current_price <= self._tp_price):
            logger.info(f"[TP HIT] {self.symbol}")
            await self._close_position("TP", pos.unrealized_pnl, current_r, regime_params)
            return

        # ── SL hit ────────────────────────────────────────────────────────
        if (current_price <= self._sl_price if side == OrderSide.LONG
                else current_price >= self._sl_price):
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
                    f"closing {partial_qty}/{pos.quantity}"
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
                    logger.info(f"[PARTIAL TP FALLBACK] {self.symbol} — full close")
                    await self._close_position("PARTIAL_TP_FULL", pos.unrealized_pnl, current_r, regime_params)
                    return

        # ── Breakeven trailing ────────────────────────────────────────────
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
            self._last_known_pnl      = 0.0

            # SL/TP guard — verify values are set after order placement
            if not self._sl_price or not self._tp_price:
                logger.warning(
                    f"[SL/TP GUARD] {params.symbol} — SL or TP is None after "
                    f"order placement (sl={self._sl_price}, tp={self._tp_price}). "
                    f"Position will be managed by timeout only."
                )

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
            regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
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
            # Exchange already closed it (race condition with SL/TP)
            logger.warning(
                f"[CLOSE FAILED] {self.symbol}: {result.message} — "
                f"treating as externally closed, pnl={pnl:+.4f}"
            )
            if pnl < 0:
                self.recovery.record_loss(self.symbol, pnl)
                self.risk.record_loss(pnl)
            else:
                self.recovery.record_profit(self.symbol, pnl)
                self.risk.record_profit(pnl)
            regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
            self.recalibrator.record_trade(
                symbol=self.symbol,
                side=self._position_side.value if self._position_side else "unknown",
                won=(pnl >= 0),
                confidence=self._entry_confidence,
                pnl_usdt=pnl,
                pnl_r=pnl_r,
                bars_held=self._bars_held,
                exit_reason=f"{reason}_EXTERNAL",
                regime=regime_name,
            )

        self.risk.portfolio.register_close(self.symbol, profit=(pnl >= 0))
        self._reset_position_state()
        self._cooldown_bars_remaining = self.COOLDOWN_BARS

    # ── State reset ───────────────────────────────────────────────────────

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
        self._last_known_pnl     = 0.0
        self._last_eval_time     = 0.0   # force immediate eval after cooldown ends

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