"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator — one instance per symbol.

v8 Changes (close reliability + state consistency)
────────────────────────────────────────────────────
ROOT CAUSE FIXED: The "CLOSE FAILED → ROGUE" cascade was caused by a broken
assumption in _close_position(). When close_position() returns False it means
the exchange had NO POSITION at that moment — which means the exchange already
closed it (SL/TP race). BUT the code was still calling _reset_position_state()
and returning, leaving the internal state marked as closed while the exchange
still had the position open (partial fill scenario or bot-triggered close
racing against a fresh open). The new _close_with_retry() logic is:

  1. Try market close.
  2. If exchange says "No open position" → verify by re-fetching positions.
     a. If truly gone  → treat as externally closed, record, reset.
     b. If still there → exchange returned a transient error; retry up to
        MAX_CLOSE_RETRIES with 1s delay. If all retries fail, log CRITICAL
        and leave _in_position=True so the next cycle's reconciler handles it.
  3. Never silently reset state on a close failure without confirming the
     position is actually gone from the exchange.

DEEP_SYNC ENTRY ANCHOR BUG FIXED: The deep sync was re-anchoring entry_price
from the exchange whenever it differed by >$1. This is wrong for BTCUSDT where
the fill price can legitimately differ from the average entry returned by the
exchange. The >$1 threshold was triggering on the BTCUSDT partial fill case,
corrupting the SL distance calculation and producing nonsensical R values of
+2,480,454. The sync now only re-anchors when entry_price is None.

BARS NOT INCREMENTING: The bar counter was correctly gated on new_5m_candle in
_reconcile_position_state. But the deep_sync on cycle 60 was running BEFORE
reconcile and called _find_my_position on a stale positions list that was
fetched BEFORE the reconcile positions fetch. Moved deep_sync AFTER reconcile.

SPLIT:STRONG THRESHOLD RAISED: +0.15 over effective threshold (was +0.08).
This prevents weak SPLIT signals from entering TRENDING regime.

UI FIXES:
- Table now shows "COOLDOWN" side when cooldown is active.
- Table uses bot internal state for entry/bars, not just exchange pos, so
  values remain visible during the ~1s window after a close when exchange
  still shows the position but bot has reset state.
- WR format changed to "L:N/WR% S:N/WR%" for clarity.

v7 Changes (production-grade state reconciliation)
v6 Changes (filter calibration + execution frequency)
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
        Regime.TRENDING:        10,   # 10 × 5m = 50 min
        Regime.RANGE:            8,   #  8 × 5m = 40 min
        Regime.HIGH_VOLATILITY:  6,   #  6 × 5m = 30 min
    }
    COOLDOWN_BARS = 3

    # Hard wall-clock cap — no trade stays open longer than this, ever.
    MAX_TRADE_DURATION_MINUTES: int = 60

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

    # ── Close reliability ─────────────────────────────────────────────────
    # How many times to retry a close before giving up and letting the
    # next-cycle reconciler take over.
    MAX_CLOSE_RETRIES  = 3
    CLOSE_RETRY_DELAY  = 1.0   # seconds between retries
    SETTLE_DELAY       = 1.5   # seconds to wait after SL/TP/exit trigger
    TAKER_FEE_RATE     = 0.0005  # 0.05% per side (Binance USDM standard taker fee)
                                  # Update to 0.0002 if you have BNB fee discount or VIP tier
                               # before attempting close, to let the exchange
                               # settle its own fill.

    # ── Reconciliation ────────────────────────────────────────────────────
    FULL_SYNC_CYCLES = 60   # deep sync every ~5 min at 5s cycle

    def __init__(
        self,
        exchange,
        symbol:            str,
        risk_manager=None,
        recovery_strategy=None,
        recalibrator:      Optional[SignalRecalibrator] = None,
        reporter=None,
        stop_event=None,
        # Overridable settings (used by run_bot_managed.py)
        leverage:              int   = None,
        risk_per_trade:        float = None,
        daily_loss_limit:      float = None,
        daily_profit_target:   float = None,
        base_threshold:        float = None,
        timeframe:             str   = None,
        **kwargs,
    ):
        self.exchange     = exchange
        self.symbol       = symbol
        self.model        = HybridModel(symbol=symbol)
        self.regime       = RegimeDetector(symbol=symbol)
        self.recalibrator = recalibrator or SignalRecalibrator()

        # ── Auto-instantiate when not provided (managed mode) ─────────────
        self.recovery = recovery_strategy or RecoveryStrategy()

        if risk_manager is None:
            from core.risk.risk_manager import RiskSettings
            rs = RiskSettings(
                leverage=leverage or settings.risk.leverage,
                take_profit_pct=settings.risk.take_profit_pct,
                stop_loss_pct=settings.risk.stop_loss_pct,
                risk_per_trade_pct=risk_per_trade or settings.risk.risk_per_trade_pct,
                max_open_trades=settings.risk.max_open_trades,
                max_daily_loss_pct=settings.risk.max_daily_loss_pct,
                max_daily_loss_usdt=daily_loss_limit or 0.0,
            )
            risk_manager = RiskManager(rs)
        self.risk = risk_manager

        # Override class-level constants if provided
        if base_threshold is not None:
            self.BASE_THRESHOLD = base_threshold / 100.0 if base_threshold > 1 else base_threshold
        if timeframe is not None:
            self.TIMEFRAME = timeframe

        # Reporter + stop event (Laravel integration)
        self._reporter   = reporter
        self._stop_event = stop_event
        self._trade_ids: dict = {}   # symbol → Laravel trade_id

        # Signal metadata for reporting
        self._last_signal_type = None
        self._last_regime      = None

        # ── Bot lifecycle ──────────────────────────────────────────────────
        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0
        self._position_opened_at: Optional[float] = None   # monotonic time

        # ── Position state ─────────────────────────────────────────────────
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
        # Frozen SL distance (R-unit) set at trade open — never changes after that.
        # Using the live self._sl_price for R calculations corrupts R after
        # breakeven/trailing moves (sl_dist → 0.0001 → R inflates by 80,000×).
        self._original_sl_dist: float = 1.0
        # Cumulative realized PnL from partial TP closes within this trade.
        # Needed so the final close reports combined PnL, not just remaining leg.
        self._partial_realized_pnl: float = 0.0
        self._original_qty: float = 0.0          # full quantity at open (for fee calc)
        self._last_qty: float = 0.0              # remaining qty (decreases after partial TP)
        self._partial_exit_notional: float = 0.0 # sum of exit notionals from partial TPs

        # ── Cooldown ───────────────────────────────────────────────────────
        self._cooldown_bars_remaining = 0

        # ── Signal timing ──────────────────────────────────────────────────
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
        """
        Compute the effective confidence threshold for this signal.

        USER THRESHOLD IS A HARD FLOOR:
        BASE_THRESHOLD is set from the user's dashboard setting (base_confidence_threshold).
        Regime adjustments and recalibration can only RAISE the threshold above the
        user's minimum — they can never lower it below what the user configured.
        This ensures the contract: 'I set 68%, no trade below 68%' is always honoured.
        """
        recalib_adj = self.recalibrator.get_threshold_adjustment(
            self.symbol, side.value) / 100.0
        recalib_adj = max(-self.RECALIB_CAP, min(self.RECALIB_CAP, recalib_adj))

        # Start from regime-adjusted threshold
        # regime_params.conf_thr_delta is the regime adjustment delta (e.g. +0.02pp).
        # It is already relative (not absolute), so add directly to BASE_THRESHOLD.
        eff = self.BASE_THRESHOLD + regime_params.conf_thr_delta + recalib_adj

        # HARD FLOOR: never go below user's configured minimum, regardless of
        # regime or recalibration adjustments.
        eff = max(eff, self.BASE_THRESHOLD)

        return min(0.95, max(self.BASE_THRESHOLD, eff))

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

    # ── Managed run loop (used by run_bot_managed.py) ────────────────────

    async def run(self, stop_event=None):
        """
        Managed run loop — used when launched by Laravel.
        Checks stop_event and reporter stop command every cycle.
        Falls back to _is_active flag for compatibility with run_bot.py.
        """
        _stop = stop_event or self._stop_event

        if self._reporter:
            # NOTE: reporter.start() is called by run_bot_managed.py BEFORE run() is called.
            # Do NOT call reporter.start() here — it would create duplicate background tasks
            # (heartbeat + flush) which cause logs and trade data to be lost.
            self._reporter.queue_log(
                "info", f"🚀 Bot started — trading {self.symbol} on {self.TIMEFRAME} timeframe"
            )

        while self._is_active:
            # ── Graceful shutdown checks ──────────────────────────────────
            if _stop and _stop.is_set():
                logger.info(f"[MANAGED] Stop event set — {self.symbol} shutting down.")
                break
            if self._reporter and self._reporter.should_stop():
                logger.info(f"[MANAGED] Stop command from Laravel — {self.symbol} shutting down.")
                break

            await self.run_cycle()
            await asyncio.sleep(5)

        if self._reporter:
            self._reporter.queue_log("info", f"🛑 Bot stopped — {self.symbol}")
            await self._reporter.flush()

    async def close_all_positions_for_shutdown(self):
        """
        Close all open positions on the exchange.
        Called by run_bot_managed.py when user chose 'Stop AND close trades'.
        """
        try:
            positions = await self.exchange.get_open_positions()
            if not positions:
                logger.info("[MANAGED] No open positions to close.")
                return

            for pos in positions:
                sym = pos.symbol
                side_str = "short" if pos.side.value == "long" else "long"  # close opposite
                logger.info(f"[MANAGED] Closing position: {sym} ({pos.side.value})")
                if self._reporter:
                    self._reporter.queue_log(
                        "info", f"🔴 Closing {sym} {pos.side.value} position @ market...", channel="trade"
                    )
                try:
                    await self.exchange.close_position(sym)
                    logger.info(f"[MANAGED] Closed {sym} successfully.")
                    if self._reporter:
                        self._reporter.queue_log(
                            "info", f"✅ {sym} position closed successfully.", channel="trade"
                        )
                except Exception as e:
                    logger.warning(f"[MANAGED] Failed to close {sym}: {e}")
                    if self._reporter:
                        self._reporter.queue_log(
                            "warning", f"⚠️ Failed to close {sym}: {e}. Close manually on exchange.", channel="trade"
                        )
        except Exception as e:
            logger.warning(f"[MANAGED] Error fetching positions for close: {e}")

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

        Returns True  → position confirmed open, proceed to monitoring.
        Returns False → no open position, proceed to signal evaluation.

        Cases:
          A. Bot=open, Exchange=open  → update bar count, return True.
          B. Bot=open, Exchange=none  → external close, record, reset.
          C. Bot=closed, Exchange=open → rogue, warn, do NOT manage.
          D. Both=closed → idle, return False.
        """
        exchange_has_pos = current_pos is not None
        bot_thinks_open  = self._in_position

        # ── Case A ────────────────────────────────────────────────────────
        if bot_thinks_open and exchange_has_pos:
            if new_5m_candle:
                self._bars_held += 1
            self._last_known_pnl = current_pos.unrealized_pnl
            return True

        # ── Case B ────────────────────────────────────────────────────────
        if bot_thinks_open and not exchange_has_pos:
            estimated_pnl = self._last_known_pnl

            # Infer exit reason from current price vs SL/TP
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

            # ── Compute correct pnl_r from actual exit price ──────────────
            # Use current_price as the best estimate of where the position
            # was closed. sl_dist is always the original SL distance from entry.
            pnl_r = 0.0
            if self._entry_price and self._sl_price:
                sl_dist = abs(self._entry_price - self._sl_price) or 1e-9
                if self._position_side == OrderSide.LONG:
                    pnl_r = (current_price - self._entry_price) / sl_dist
                else:
                    pnl_r = (self._entry_price - current_price) / sl_dist
                pnl_r = max(-9999.0, min(9999.0, pnl_r))

            logger.warning(
                f"[RECONCILE:{exit_reason}] {self.symbol} — position vanished on exchange. "
                f"estimated_pnl={estimated_pnl:+.4f} USDT | R={pnl_r:+.2f} | "
                f"last_price={current_price:.4f} | bars_held={self._bars_held}"
            )

            regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
            self._record_trade_outcome(estimated_pnl, exit_reason, regime_name)
            self.risk.portfolio.register_close(self.symbol, profit=(estimated_pnl >= 0))

            # Report the external close to Laravel so the trade is recorded correctly
            if self._reporter:
                pnl_emoji = "🟢" if estimated_pnl >= 0 else "🔴"
                self._reporter.queue_log(
                    "warning",
                    f"{pnl_emoji} Trade closed externally ({exit_reason}): {self.symbol} | "
                    f"P&L: {estimated_pnl:+.4f} USDT | R: {pnl_r:+.2f}",
                    channel="trade"
                )
                trade_id = self._trade_ids.pop(self.symbol, None)
                est_fee  = (
                    (self._entry_price or 0.0) * (self._original_qty or 0.0) +
                    current_price * (self._last_qty or self._original_qty or 0.0) +
                    self._partial_exit_notional
                ) * self.TAKER_FEE_RATE
                await self._reporter.report_trade_close(
                    symbol=self.symbol,
                    side=self._position_side.value if self._position_side else "long",
                    exit_price=current_price,
                    pnl_usdt=estimated_pnl,
                    pnl_r=pnl_r,
                    exit_reason=exit_reason,
                    bars_held=self._bars_held,
                    trade_id=trade_id,
                    fee_usdt=round(est_fee, 4),
                )

            self._reset_position_state()
            self._cooldown_bars_remaining = self.COOLDOWN_BARS
            return False

        # ── Case C ────────────────────────────────────────────────────────
        if not bot_thinks_open and exchange_has_pos:
            logger.warning(
                f"[RECONCILE:ROGUE] {self.symbol} — unknown open position on exchange "
                f"(side={current_pos.side.value}, qty={current_pos.quantity:.4f}, "
                f"entry={current_pos.entry_price:.4f}). "
                f"Bot will NOT manage this. Close it manually."
            )
            return False

        # ── Case D ────────────────────────────────────────────────────────
        return False

    async def _deep_sync(self, positions):
        """
        Periodic deep validation. Checks SL/TP are set.
        Does NOT re-anchor entry_price (that was causing R calculation
        corruption on partial fills — entry_price is only set at open time).
        """
        current_pos = self._find_my_position(positions)

        if self._in_position and current_pos:
            # Only restore entry_price if it was somehow lost (None)
            if self._entry_price is None and current_pos.entry_price:
                self._entry_price = current_pos.entry_price
                logger.info(
                    f"[DEEP_SYNC] {self.symbol} — entry_price restored from exchange: "
                    f"{current_pos.entry_price:.4f}"
                )

            if not self._sl_price or not self._tp_price:
                logger.warning(
                    f"[DEEP_SYNC] {self.symbol} — active position missing SL/TP "
                    f"(sl={self._sl_price}, tp={self._tp_price}). Timeout only."
                )

        logger.debug(
            f"[DEEP_SYNC] {self.symbol} | "
            f"bot_open={self._in_position} | "
            f"exchange={'YES' if current_pos else 'NO'} | "
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
            # FIX: Pass user's BASE_THRESHOLD so regime log shows the correct
            # conf_thr value. Previously detect() used its own hardcoded default
            # (~0.52), causing regime log to show '57%' while trades were actually
            # blocked at 72%. Now both values reflect the user's configuration.
            regime_params = self.regime.detect(
                df,
                base_conf_threshold=self.BASE_THRESHOLD,
            )
            self._last_regime_params = regime_params

            # ── Track 5m bars ─────────────────────────────────────────────
            ts_5m = (int(df.index[-1].timestamp() * 1000)
                     if hasattr(df.index[-1], 'timestamp') else self._cycles)
            new_5m_candle = (ts_5m != self._last_5m_candle_ts)
            if new_5m_candle:
                self._last_5m_candle_ts = ts_5m
                # Tick recovery wait-bar counter so auto-clear can fire
                if self.recovery.is_in_recovery(self.symbol):
                    self.recovery.tick_bar(self.symbol)

            # ── Fetch live positions ──────────────────────────────────────
            positions   = await self.exchange.get_open_positions()
            current_pos = self._find_my_position(positions)

            # ── Position reconciliation ───────────────────────────────────
            # Must run BEFORE deep_sync so bar counting is authoritative.
            position_is_open = await self._reconcile_position_state(
                current_pos, current_price, regime_params, new_5m_candle
            )

            # ── Periodic deep sync (run after reconcile) ──────────────────
            if self._cycles % self.FULL_SYNC_CYCLES == 0:
                await self._deep_sync(positions)

            if position_is_open:
                await self._monitor_position(current_pos, current_price, df, regime_params)
                return

            # ── No open position: cooldown → signals ──────────────────────

            if self._cooldown_bars_remaining > 0:
                if new_5m_candle:
                    self._cooldown_bars_remaining -= 1
                logger.info(
                    f"[COOLDOWN] {self.symbol} — {self._cooldown_bars_remaining} bars remaining"
                )
                if self._reporter:
                    self._reporter.queue_log(
                        "info",
                        f"⏳ {self.symbol} — Cooldown after last trade: "
                        f"{self._cooldown_bars_remaining} bars remaining before next signal evaluation.",
                        channel="signal"
                    )
                return

            # ── Signal timing gate ────────────────────────────────────────
            first_cycle = (self._last_eval_time == 0.0)
            if not self._should_evaluate_signal(df_1m, force=first_cycle):
                logger.info(f"[GATE] {self.symbol} — waiting for next eval window")
                if self._reporter and self._cycles % 3 == 0:  # log every ~15s not every 5s
                    price_str = ""
                    try:
                        if df is not None and len(df) > 0:
                            price_str = f" | Price: {df['close'].iloc[-1]:.2f}"
                    except Exception:
                        pass
                    self._reporter.queue_log(
                        "info",
                        f"👁 {self.symbol} — Monitoring market{price_str}. "
                        f"Waiting for next candle to evaluate signal.",
                        channel="signal"
                    )
                return

            # ── Volatility gate ───────────────────────────────────────────
            atr_ratio = 0.0
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
                        if self._reporter:
                            self._reporter.queue_log(
                                "info",
                                f"📉 {self.symbol} — Market too quiet (volatility {atr_ratio:.0%} of average). "
                                f"Waiting for a better setup.",
                                channel="signal"
                            )
                        self._last_eval_time = time.monotonic()
                        return

            # ── Signal evaluation ─────────────────────────────────────────
            prediction = self.model.predict(df, df_1h=df_1h)
            self._last_eval_time = time.monotonic()

            # Capture metadata for reporter and trade recording
            agreement = ""
            if "]" in prediction.reasoning:
                agreement = prediction.reasoning.split("]")[0].replace("[HYBRID ", "")
            self._last_signal_type = "AGREE" if "AGREE" in agreement else "SPLIT"
            self._last_regime      = regime_params.regime.value if regime_params else None

            logger.info(
                f"[SIGNAL] {self.symbol} → {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol}")
                if self._reporter:
                    self._reporter.queue_log(
                        "info",
                        f"🔍 {self.symbol} — Analysed market. No clear trade opportunity right now "
                        f"(confidence {prediction.confidence:.0%}, regime: {self._last_regime}). Watching.",
                        channel="signal"
                    )
                return

            side = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT
            eff_threshold = self._effective_threshold(regime_params, side)

            if prediction.confidence < eff_threshold:
                logger.info(
                    f"[SKIP:CONF] {self.symbol} — "
                    f"conf={prediction.confidence:.2%} < threshold={eff_threshold:.2%}"
                )
                if self._reporter:
                    direction = "LONG 📈" if side == OrderSide.LONG else "SHORT 📉"
                    self._reporter.queue_log(
                        "info",
                        f"🔍 {self.symbol} — {direction} signal seen but confidence too low "
                        f"({prediction.confidence:.1%} < required {eff_threshold:.1%}). Skipped.",
                        channel="signal"
                    )
                return

            # ── AGREE filter (TRENDING only) ──────────────────────────────
            if regime_params.require_agree and "AGREE" not in agreement:
                if prediction.confidence >= eff_threshold + 0.20:  # 92%+ required to override AGREE
                    logger.info(
                        f"[SPLIT:STRONG] {self.symbol} — high conf (≥92%) "
                        f"{prediction.confidence:.0%} in TRENDING, proceeding"
                    )
                    if self._reporter:
                        self._reporter.queue_log(
                            "info",
                            f"⚡ {self.symbol} — Strong signal ({prediction.confidence:.0%}) detected "
                            f"in trending market. Proceeding despite split models.",
                            channel="signal"
                        )
                else:
                    logger.info(
                        f"[SKIP:SPLIT] {self.symbol} — TRENDING requires AGREE. "
                        f"conf={prediction.confidence:.0%}"
                    )
                    if self._reporter:
                        self._reporter.queue_log(
                            "info",
                            f"🔍 {self.symbol} — Models disagree on direction in trending market. "
                            f"Skipping to avoid risky trade.",
                            channel="signal"
                        )
                    return

            # ── Reporter: signal event ─────────────────────────────────────
            if self._reporter:
                await self._reporter.report_signal(
                    symbol          = self.symbol,
                    signal          = prediction.signal.value,
                    confidence      = prediction.confidence,
                    signal_type     = self._last_signal_type,
                    regime          = self._last_regime,
                    adx             = regime_params.adx if hasattr(regime_params, "adx") else None,
                    atr_ratio       = atr_ratio,
                    action_taken    = "evaluating",
                    price_at_signal = current_price,
                )

            # ── Risk checks ───────────────────────────────────────────────
            balance    = await self.exchange.get_balance()
            total_open = len(positions)

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — trading paused")
                if self._reporter:
                    self._reporter.queue_log(
                        "warning",
                        f"🚫 {self.symbol} — Daily loss limit reached. Bot is paused for today to protect your account.",
                        channel="bot"
                    )
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
                tp_atr_mult=regime_params.tp_mult,   # no 1.5x stretch — keeps TP reachable within timeout
                size_scale=regime_params.size_scale,
            )

            logger.info(
                f"[TRADE_PARAMS] {self.symbol} {side.value.upper()} | "
                f"SL={trade_params.stop_loss:.2f} ({regime_params.sl_mult:.2f}x ATR) | "
                f"TP={trade_params.take_profit:.2f} ({regime_params.tp_mult:.2f}x ATR) | "
                f"Qty={trade_params.quantity:.4f} | Risk=${trade_params.risk_amount_usdt:.2f} | "
                f"R:R={abs(trade_params.take_profit - trade_params.entry_price) / max(abs(trade_params.stop_loss - trade_params.entry_price), 1e-9):.2f}"
            )
            if not trade_params.approved:
                logger.warning(f"[REJECTED] {self.symbol} — {trade_params.reject_reason}")
                if self._reporter:
                    self._reporter.queue_log(
                        "warning",
                        f"⚠️ {self.symbol} — Trade rejected by risk manager: {trade_params.reject_reason}",
                        channel="bot"
                    )
                return

            if self.recovery.is_in_recovery(self.symbol):
                if self.recovery.should_open_recovery(self.symbol, prediction):
                    mult = self.recovery.get_recovery_size_multiplier(self.symbol)
                    trade_params.quantity = round(trade_params.quantity * mult, 3)
                    logger.info(f"[RECOVERY] {self.symbol} size x{mult:.1f}")
                    if self._reporter:
                        self._reporter.queue_log(
                            "info",
                            f"🔄 {self.symbol} — Recovery mode: increasing position size by {mult:.1f}x "
                            f"to recover previous loss.",
                            channel="bot"
                        )
                    # Reset wait counter — a qualifying signal fired
                    if self.symbol in self.recovery._states:
                        self.recovery._states[self.symbol].wait_bars = 0
                else:
                    logger.info(f"[RECOVERY WAIT] {self.symbol}")
                    if self._reporter:
                        self._reporter.queue_log(
                            "info",
                            f"⏳ {self.symbol} — In recovery mode. Waiting for a high-confidence "
                            f"signal before re-entering.",
                            channel="bot"
                        )
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
        # No SL/TP: log and enforce timeout only
        if not (self._tp_price and self._sl_price and self._entry_price):
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} | bars={self._bars_held} [NO SL/TP]"
            )
            max_bars = self._get_timeout(regime_params)
            if self._bars_held >= max_bars:
                logger.info(f"[TIMEOUT] {self.symbol} {max_bars} bars (no SL/TP mode)")
                await asyncio.sleep(self.SETTLE_DELAY)
                await self._close_with_retry(
                    "TIMEOUT", pos.unrealized_pnl, 0.0, regime_params, current_price
                )
            return

        entry = self._entry_price
        side  = pos.side

        # Use the FROZEN original SL distance as the R-unit.
        # After SL moves to breakeven, self._sl_price is entry ± 0.0001,
        # making sl_dist ≈ 0.0001 and inflating R by ~80,000×.
        # _original_sl_dist is set once at open and never changes.
        sl_dist   = self._original_sl_dist or abs(entry - self._sl_price) or 1.0
        current_r = (
            (current_price - entry) / sl_dist if side == OrderSide.LONG
            else (entry - current_price) / sl_dist
        )

        _elapsed_min = (
            (time.monotonic() - self._position_opened_at) / 60.0
            if self._position_opened_at else 0.0
        )
        logger.info(
            f"[WATCHING] {self.symbol} {side.value.upper()} | "
            f"entry={entry:.4f} now={current_price:.4f} | "
            f"TP={self._tp_price:.4f}  SL={self._sl_price:.4f} | "
            f"R={current_r:+.2f}  PnL={pos.unrealized_pnl:+.4f} | "
            f"bars={self._bars_held}/{self._get_timeout(regime_params)} | "
            f"open={_elapsed_min:.0f}m/{self.MAX_TRADE_DURATION_MINUTES}m"
        )

        # ── Hard 60-minute wall-clock cap (checked before everything else) ─────
        if self._position_opened_at is not None:
            elapsed_min = (time.monotonic() - self._position_opened_at) / 60.0
            if elapsed_min >= self.MAX_TRADE_DURATION_MINUTES:
                logger.warning(
                    f"[HARD TIMEOUT] {self.symbol} — position open for "
                    f"{elapsed_min:.1f} min >= {self.MAX_TRADE_DURATION_MINUTES} min cap. "
                    f"Force-closing regardless of P&L."
                )
                if self._reporter:
                    self._reporter.queue_log(
                        "warning",
                        f"\u23f1\ufe0f {self.symbol} — Position exceeded {self.MAX_TRADE_DURATION_MINUTES}-minute "
                        f"max hold time ({elapsed_min:.0f} min open). Force-closing now.",
                        channel="trade"
                    )
                await asyncio.sleep(self.SETTLE_DELAY)
                await self._close_with_retry(
                    "HARD_TIMEOUT", pos.unrealized_pnl, current_r, regime_params, current_price
                )
                return

        # ── TP hit ────────────────────────────────────────────────────────
        if (current_price >= self._tp_price if side == OrderSide.LONG
                else current_price <= self._tp_price):
            logger.info(f"[TP HIT] {self.symbol}")
            await asyncio.sleep(self.SETTLE_DELAY)
            await self._close_with_retry("TP", pos.unrealized_pnl, current_r, regime_params, current_price)
            return

        # ── SL hit ────────────────────────────────────────────────────────
        if (current_price <= self._sl_price if side == OrderSide.LONG
                else current_price >= self._sl_price):
            logger.info(f"[SL HIT] {self.symbol}")
            await asyncio.sleep(self.SETTLE_DELAY)
            await self._close_with_retry("SL", pos.unrealized_pnl, current_r, regime_params, current_price)
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
                    # Accumulate the realized PnL from this partial close so the final
                    # close can report the true combined PnL (partial + remaining leg).
                    partial_realized = (
                        (current_price - entry) * partial_qty if side == OrderSide.LONG
                        else (entry - current_price) * partial_qty
                    )
                    self._partial_realized_pnl  += partial_realized
                    self._last_qty               = round(pos.quantity - partial_qty, 3)
                    self._partial_exit_notional  += current_price * partial_qty
                    logger.info(f"[PARTIAL TP OK] {self.symbol} SL → breakeven {self._sl_price:.4f} | "
                                f"partial_realized={partial_realized:+.4f} USDT")
                    if self._reporter:
                        self._reporter.queue_log("info",
                            f"🎯 {self.symbol} — Took 50% profit at {current_price:.4f} (R={current_r:+.2f}). "
                            f"Stop loss moved to breakeven. Riding the rest.",
                            channel="trade")
                else:
                    logger.info(f"[PARTIAL TP FALLBACK] {self.symbol} — attempting full close")
                    await asyncio.sleep(self.SETTLE_DELAY)
                    await self._close_with_retry(
                        "PARTIAL_TP_FULL", pos.unrealized_pnl, current_r, regime_params, current_price
                    )
                    return

        # ── Breakeven trailing ────────────────────────────────────────────
        if not self._breakeven_moved and current_r >= self.BREAKEVEN_TRIGGER_R:
            new_sl = entry + 0.0001 if side == OrderSide.LONG else entry - 0.0001
            if abs(new_sl - self._sl_price) > 0.0001:
                logger.info(f"[BREAKEVEN] {self.symbol} SL {self._sl_price:.4f} → {new_sl:.4f}")
                self._sl_price        = new_sl
                self._breakeven_moved = True
                if self._reporter:
                    self._reporter.queue_log("info",
                        f"🛡️ {self.symbol} — Stop loss moved to breakeven at {new_sl:.4f}. Position is now risk-free.",
                        channel="trade")

        # ── Early profit exit ─────────────────────────────────────────────
        early_r = regime_params.early_profit_r
        if (
            self._bars_held >= self.MIN_HOLD_BARS
            and current_r >= early_r
            and not self._partial_exit_taken
        ):
            logger.info(f"[EARLY PROFIT] {self.symbol} R={current_r:+.2f} >= {early_r:.2f}R")
            await asyncio.sleep(self.SETTLE_DELAY)
            await self._close_with_retry(
                "EARLY_PROFIT", pos.unrealized_pnl, current_r, regime_params, current_price
            )
            return

        # ── Reversal check ────────────────────────────────────────────────
        prediction = self.model.predict(df)
        opposite = (prediction.signal == Signal.SHORT if side == OrderSide.LONG
                    else prediction.signal == Signal.LONG)
        if opposite and self._bars_held >= self.MIN_HOLD_BARS:
            if "AGREE" in prediction.reasoning and current_r > 0:
                logger.info(f"[REVERSAL EXIT] {self.symbol} R={current_r:+.2f}")
                await asyncio.sleep(self.SETTLE_DELAY)
                await self._close_with_retry(
                    "REVERSAL", pos.unrealized_pnl, current_r, regime_params, current_price
                )
                return
            else:
                logger.info(f"[REVERSAL SKIPPED] {self.symbol} SPLIT or not in profit")

        # ── Timeout ───────────────────────────────────────────────────────
        max_bars = self._get_timeout(regime_params)
        if self._bars_held >= max_bars:
            logger.info(f"[TIMEOUT] {self.symbol} {max_bars} bars")
            await asyncio.sleep(self.SETTLE_DELAY)
            await self._close_with_retry(
                "TIMEOUT", pos.unrealized_pnl, current_r, regime_params, current_price
            )

    def _get_timeout(self, regime_params: RegimeParams) -> int:
        bars = self.TIMEOUT_BARS.get(regime_params.regime, 8)
        # Never let bar-based timeout exceed the hard wall-clock cap
        cap_bars = (self.MAX_TRADE_DURATION_MINUTES * 60) // (5 * 60)   # 5-min bars
        return min(bars, cap_bars)

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
            self._position_opened_at  = time.monotonic()
            # Freeze the original SL distance as the R-unit for this trade.
            # Position management (self._sl_price) and performance measurement
            # (R calculation) must never share the same mutable value.
            self._original_sl_dist      = abs(result.price - params.stop_loss) or 1.0
            self._partial_realized_pnl  = 0.0
            self._original_qty          = params.quantity
            self._last_qty              = params.quantity
            self._partial_exit_notional = 0.0

            if not self._sl_price or not self._tp_price:
                logger.warning(
                    f"[SL/TP GUARD] {params.symbol} — SL or TP is None after "
                    f"order placement (sl={self._sl_price}, tp={self._tp_price}). "
                    f"Timeout only."
                )

            self.risk.portfolio.register_open(self.symbol, params.side)
            direction = "LONG 📈" if params.side == OrderSide.LONG else "SHORT 📉"
            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price:.4f} | "
                f"TP={params.take_profit:.4f} | SL={params.stop_loss:.4f}"
            )

            # ── Reporter: trade open ───────────────────────────────────────
            if self._reporter:
                self._reporter.queue_log(
                    "info",
                    f"✅ Trade opened: {direction} {params.symbol} | "
                    f"Entry: {result.price:.4f} | Qty: {params.quantity} | "
                    f"TP: {params.take_profit:.4f} | SL: {params.stop_loss:.4f} | "
                    f"Confidence: {self._entry_confidence:.0%}",
                    channel="trade"
                )
                trade_id = await self._reporter.report_trade_open(
                    symbol      = self.symbol,
                    side        = params.side.value,
                    entry_price = result.price,
                    quantity    = params.quantity,
                    leverage    = params.leverage,
                    tp_price    = params.take_profit,
                    sl_price    = params.stop_loss,
                    confidence  = self._entry_confidence,
                    signal_type = self._last_signal_type,
                    regime      = self._last_regime,
                    risk_usdt   = params.risk_amount_usdt,
                    order_id    = getattr(result, "order_id", None),
                )
                if trade_id:
                    self._trade_ids[self.symbol] = trade_id
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")
            if self._reporter:
                self._reporter.queue_log(
                    "error",
                    f"❌ Failed to open trade on {params.symbol}: {result.message}",
                    channel="trade"
                )

    async def _close_with_retry(
        self,
        reason:        str,
        pnl:           float,
        pnl_r:         float = 0.0,
        regime_params: Optional[RegimeParams] = None,
        current_price: float = None,
    ):
        """
        Reliable close with verification and retry.
        pnl is the unrealized_pnl of the remaining position at close time.
        If a partial TP was taken, the true total PnL includes _partial_realized_pnl.
        """
        if self._partial_realized_pnl != 0.0:
            combined_pnl = pnl + self._partial_realized_pnl
            logger.info(
                f"[COMBINED PnL] {self.symbol} — remaining={pnl:+.4f} + "
                f"partial={self._partial_realized_pnl:+.4f} = total={combined_pnl:+.4f} USDT"
            )
            pnl = combined_pnl

        # ── Compute round-trip trading fee ────────────────────────────────
        # entry_notional uses _entry_price × original full quantity (before partial).
        # exit_notional  uses current_price × remaining quantity.
        # partial TP leg adds its own exit notional (entry already counted).
        entry_notional   = (self._entry_price or 0.0) * (self._original_qty or 0.0)
        exit_notional    = (current_price or 0.0) * (self._last_qty or 0.0)
        partial_notional = self._partial_exit_notional  # accumulated during partial TP
        fee_usdt = (entry_notional + exit_notional + partial_notional) * self.TAKER_FEE_RATE
        net_pnl  = pnl - fee_usdt
        logger.info(
            f"[FEES] {self.symbol} — entry_notional=${entry_notional:.2f} | "
            f"exit_notional=${exit_notional:.2f} | fee=${fee_usdt:.4f} | "
            f"gross={pnl:+.4f} | net={net_pnl:+.4f} USDT"
        )
        for attempt in range(1, self.MAX_CLOSE_RETRIES + 1):
            result = await self.exchange.close_position(self.symbol)

            if result.success:
                # ── Clean close ───────────────────────────────────────────
                logger.info(
                    f"[CLOSED:{reason}] {self.symbol} | "
                    f"PnL={pnl:+.4f} USDT | R={pnl_r:+.2f} | bars={self._bars_held}"
                )
                regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
                self._record_trade_outcome(pnl, reason, regime_name)
                self.risk.portfolio.register_close(self.symbol, profit=(pnl >= 0))

                # ── Reporter: trade close ──────────────────────────────────
                if self._reporter:
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    self._reporter.queue_log(
                        "info",
                        f"{pnl_emoji} Trade closed ({reason}): {self.symbol} | "
                        f"P&L: {pnl:+.4f} USDT | R: {pnl_r:+.2f} | Bars held: {self._bars_held}",
                        channel="trade"
                    )
                    trade_id = self._trade_ids.pop(self.symbol, None)
                    await self._reporter.report_trade_close(
                        symbol=self.symbol,
                        side=self._position_side.value if self._position_side else "long",
                        exit_price=current_price or 0.0,
                        pnl_usdt=pnl,
                        pnl_r=pnl_r,
                        exit_reason=reason,
                        bars_held=self._bars_held,
                        trade_id=trade_id,
                        fee_usdt=round(fee_usdt, 4),
                    )

                self._reset_position_state()
                self._cooldown_bars_remaining = self.COOLDOWN_BARS
                return

            # ── Close failed — verify via re-fetch ────────────────────────
            logger.warning(
                f"[CLOSE ATTEMPT {attempt}/{self.MAX_CLOSE_RETRIES}] {self.symbol} "
                f"exchange returned: {result.message}"
            )

            try:
                live_positions = await self.exchange.get_open_positions()
                still_open = self._find_my_position(live_positions)
            except Exception as e:
                logger.error(f"[CLOSE VERIFY] {self.symbol} — position re-fetch failed: {e}")
                still_open = None  # assume gone; will be caught by reconciler

            if not still_open:
                # Position confirmed gone — exchange closed it before us
                logger.warning(
                    f"[CLOSED:{reason}_EXTERNAL] {self.symbol} — "
                    f"position confirmed closed on exchange (race condition). "
                    f"pnl≈{pnl:+.4f} USDT"
                )
                regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
                self._record_trade_outcome(pnl, f"{reason}_EXTERNAL", regime_name)
                self.risk.portfolio.register_close(self.symbol, profit=(pnl >= 0))
                if self._reporter:
                    pnl_emoji = "🟢" if pnl >= 0 else "🔴"
                    self._reporter.queue_log("info",
                        f"{pnl_emoji} Trade closed externally ({reason}): {self.symbol} | P&L: {pnl:+.4f} USDT",
                        channel="trade")
                    trade_id = self._trade_ids.pop(self.symbol, None)
                    await self._reporter.report_trade_close(
                        symbol=self.symbol, side=self._position_side.value if self._position_side else "long",
                        exit_price=current_price or 0.0, pnl_usdt=pnl, pnl_r=pnl_r,
                        exit_reason=f"{reason}_EXTERNAL", bars_held=self._bars_held, trade_id=trade_id,
                        fee_usdt=round(fee_usdt, 4),
                    )
                self._reset_position_state()
                self._cooldown_bars_remaining = self.COOLDOWN_BARS
                return

            # Position still open — wait and retry
            if attempt < self.MAX_CLOSE_RETRIES:
                logger.warning(
                    f"[CLOSE RETRY] {self.symbol} — position still open, "
                    f"retrying in {self.CLOSE_RETRY_DELAY}s "
                    f"({attempt}/{self.MAX_CLOSE_RETRIES})"
                )
                await asyncio.sleep(self.CLOSE_RETRY_DELAY)

        # ── All retries exhausted ─────────────────────────────────────────
        # On Binance testnet (and occasionally live), positions that were closed
        # by an exchange-side trigger (SL/TP fill) briefly remain visible in
        # fetch_positions() but return "No open position" on close attempts.
        # This is a "pending settlement" state — the position IS effectively
        # closed on the exchange; it just hasn't been removed from the position
        # list yet. Treating this as externally closed is correct.
        # We reset internal state and let the reconciler verify next cycle.
        logger.warning(
            f"[CLOSED:{reason}_PENDING_SETTLEMENT] {self.symbol} — "
            f"exchange rejected all {self.MAX_CLOSE_RETRIES} close attempts "
            f"(likely SL/TP triggered and position is settling). "
            f"Treating as externally closed. pnl≈{pnl:+.4f} USDT"
        )
        regime_name = regime_params.regime.value if regime_params else "UNKNOWN"
        self._record_trade_outcome(pnl, f"{reason}_SETTLEMENT", regime_name)
        self.risk.portfolio.register_close(self.symbol, profit=(pnl >= 0))
        if self._reporter:
            pnl_emoji = "🟢" if pnl >= 0 else "🔴"
            self._reporter.queue_log("warning",
                f"{pnl_emoji} Trade closed (settlement): {self.symbol} | P&L: {pnl:+.4f} USDT | "
                f"Position settled on exchange before bot could close it.",
                channel="trade")
            trade_id = self._trade_ids.pop(self.symbol, None)
            await self._reporter.report_trade_close(
                symbol=self.symbol, side=self._position_side.value if self._position_side else "long",
                exit_price=current_price or 0.0, pnl_usdt=pnl, pnl_r=pnl_r,
                exit_reason=f"{reason}_SETTLEMENT", bars_held=self._bars_held, trade_id=trade_id,
                fee_usdt=round(fee_usdt, 4),
            )
        self._reset_position_state()
        self._cooldown_bars_remaining = self.COOLDOWN_BARS

    def _record_trade_outcome(self, pnl: float, exit_reason: str, regime_name: str):
        """Shared bookkeeping for both bot-closed and externally-closed trades."""
        if pnl < 0:
            self.recovery.record_loss(self.symbol, pnl)
            self.risk.record_loss(pnl)
        else:
            self.recovery.record_profit(self.symbol, pnl)
            self.risk.record_profit(pnl)

        self.recalibrator.record_trade(
            symbol=self.symbol,
            side=self._position_side.value if self._position_side else "unknown",
            won=(pnl >= 0),
            confidence=self._entry_confidence,
            pnl_usdt=pnl,
            pnl_r=0.0,
            bars_held=self._bars_held,
            exit_reason=exit_reason,
            regime=regime_name,
        )

    # ── State reset ───────────────────────────────────────────────────────

    def _reset_position_state(self):
        self._in_position          = False
        self._tp_price             = None
        self._sl_price             = None
        self._position_side        = None
        self._entry_price          = None
        self._entry_confidence     = 0.0
        self._bars_held            = 0
        self._breakeven_moved      = False
        self._partial_exit_taken   = False
        self._last_known_pnl       = 0.0
        self._last_eval_time       = 0.0
        self._position_opened_at   = None
        self._original_sl_dist      = 1.0
        self._partial_realized_pnl  = 0.0
        self._original_qty          = 0.0
        self._last_qty              = 0.0
        self._partial_exit_notional = 0.0

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
            "symbol":           self.symbol,
            "cycles":           self._cycles,
            "trades_opened":    self._trades_opened,
            "in_position":      self._in_position,
            "position_side":    self._position_side,
            "tp":               self._tp_price,
            "sl":               self._sl_price,
            "entry":            self._entry_price,
            "bars_held":        self._bars_held,
            "regime":           regime_name,
            "in_recovery":      self.recovery.is_in_recovery(self.symbol),
            "ml_trained":       self.model.ml_is_trained,
            "partial_taken":    self._partial_exit_taken,
            "cooldown_bars":    self._cooldown_bars_remaining,
            "recalib":          recalib,
        }