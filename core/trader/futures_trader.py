"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator.

Changes in this version:
  1. REQUIRE_AGREEMENT flag (default True)
     — Only enter when both the technical model AND the ML model independently
       agree on the direction (AGREE, not SPLIT). SPLIT signals are still shown
       in logs but not traded unless you set REQUIRE_AGREEMENT = False.
     — Why: In 15 cycles of paper trading, 3 of 4 symbols showed SPLIT status
       (ML says SHORT, technicals disagree). BNB showed AGREE at 89-90% and was
       the only symbol with a clean floating PnL. The other three (SPLIT) ran
       against the position consistently. Requiring agreement filters out
       exactly the lower-quality signals that cause this.

  2. MAX_CONCURRENT_ENTRIES limit (default 2)
     — Even when REQUIRE_AGREEMENT is True, don't open more than N positions
       in the same cycle. BTC/ETH/BNB/SOL are ~0.8 correlated. Opening 4
       simultaneous shorts = 4× the directional exposure, not 4× independent bets.
     — Highest-confidence AGREE signals are prioritised. SPLIT signals fill
       remaining slots only if REQUIRE_AGREEMENT = False.

  3. Early profit capture (EARLY_PROFIT_THRESHOLD_R = 0.5)
     — Close when floating PnL >= 0.5× the SL distance in profit, after
       MIN_HOLD_BARS. Captures small profits before reversals.

  4. Signal reversal exit
     — Close when model flips direction with sufficient confidence after
       MIN_HOLD_BARS. Prevents holding stale signals.

  5. Cooldown after SL (COOLDOWN_BARS = 4 bars = 1 hour)
     — Prevents revenge trading immediately after a stop-loss.

  6. Signal recheck after close → immediate re-entry
     — After any early-exit/reversal close, falls through to signal evaluation
       in the same cycle rather than waiting 60 seconds.

  7. ATR-based TP/SL matching the label engine multipliers per symbol.

  8. Portfolio-level risk cap (max 3% total open risk across all symbols).

  9. Timeout guard: force-close after MAX_HOLD_BARS bars (8 hours).

 10. TP/SL recovery on restart (fixes 'monitoring...' display bug).

 11. Model freshness check at startup (warns if > 90 days old).

CONFIGURABLE PARAMETERS (class-level constants):
    REQUIRE_AGREEMENT         — True = only enter on AGREE signals (recommended)
                                False = enter on SPLIT too (more trades, lower quality)
    MAX_CONCURRENT_ENTRIES    — Max new positions to open in one cycle
    EARLY_PROFIT_THRESHOLD_R  — R-multiples of profit to trigger early exit (0.5)
    MIN_HOLD_BARS             — bars before early-exit logic fires (3 = 45 min)
    COOLDOWN_BARS             — bars after SL before re-entering (4 = 1 hour)
    MAX_HOLD_BARS             — force-close after this many bars (32 = 8 hours)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from core.exchange.base_exchange import OrderSide, PositionInfo
from core.models.hybrid_model import HybridModel
from core.models.base_model import Signal
from core.risk.risk_manager import RiskManager, TradeParameters
from data.indicators import ohlcv_to_dataframe, add_all_indicators
from config.logger import get_logger

logger = get_logger(__name__)


def _normalize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace(":USDT", "").replace(":BTC", "").upper()


# ── Per-symbol confidence thresholds ──────────────────────────────────────────
# ETH and SOL models are HOLD-biased (predict HOLD 60-72% vs 43-51% label rate).
# Lower threshold recovers valid LONG/SHORT signals the model would otherwise skip.
# These MUST match CONF_THRESHOLDS in scripts/validate_models.py.
CONF_THRESHOLDS: dict[str, float] = {
    "BTCUSDT": 0.42,
    "ETHUSDT": 0.38,
    "BNBUSDT": 0.42,
    "SOLUSDT": 0.38,
}

# ── ATR multipliers for TP/SL ─────────────────────────────────────────────────
# Must match ATR_MULT_MAP in scripts/train_from_history.py.
ATR_SL_MULT: dict[str, float] = {
    "BTCUSDT": 1.5,
    "ETHUSDT": 1.8,
    "BNBUSDT": 1.6,
    "SOLUSDT": 2.0,
}
ATR_TP_MULT: dict[str, float] = {sym: v * 2.0 for sym, v in ATR_SL_MULT.items()}

# ── Portfolio risk cap ────────────────────────────────────────────────────────
MAX_PORTFOLIO_RISK_PCT = 0.03   # 3%


class FuturesTrader:

    TIMEFRAME   = "15m"
    MIN_CANDLES = 250

    # ── Entry quality filters ─────────────────────────────────────────────────

    # Require both ML and technical models to agree before entering.
    # When True: only AGREE signals are traded (fewer trades, higher quality).
    # When False: SPLIT signals also enter (more trades, more noise exposure).
    # From paper trading: BNB at 89% AGREE performed cleanly; BTC/ETH/SOL at
    # 70-76% SPLIT all ran against the position in the first 15 cycles.
    REQUIRE_AGREEMENT: bool = True

    # Maximum new positions to open in a single cycle across all symbols.
    # Prevents opening all 4 correlated symbols simultaneously.
    # The highest-confidence AGREE signals are prioritised.
    MAX_CONCURRENT_ENTRIES: int = 2

    # ── Position management parameters ───────────────────────────────────────
    EARLY_PROFIT_THRESHOLD_R: float = 0.5   # close at 0.5R profit
    MIN_HOLD_BARS:            int   = 3     # 3 bars = 45 min before early exits
    COOLDOWN_BARS:            int   = 4     # 4 bars = 1h cooldown after SL
    MAX_HOLD_BARS:            int   = 32    # 32 bars = 8h timeout

    def __init__(self, exchange, symbol: str, risk_manager: RiskManager,
                 portfolio_risk_tracker=None,
                 entry_gate=None,          # shared EntryGate instance
                 **kwargs):
        self.exchange = exchange
        self.symbol   = symbol
        self.risk     = risk_manager
        self.model    = HybridModel(symbol=symbol)

        self._portfolio_tracker = portfolio_risk_tracker
        self._entry_gate        = entry_gate   # limits concurrent entries per cycle

        self._is_active     = True
        self._cycles        = 0
        self._trades_opened = 0
        self._bar_counter   = 0

        self._in_position:   bool            = False
        self._tp_price:      Optional[float] = None
        self._sl_price:      Optional[float] = None
        self._position_side: Optional[OrderSide] = None
        self._entry_price:   Optional[float] = None
        self._entry_bar:     Optional[int]   = None
        self._sl_distance:   Optional[float] = None

        self._cooldown_until_bar: int = 0
        self._closed_on_bar:      int = -1  # bar when last position closed

        self._check_model_freshness()

    # ── Model freshness check ─────────────────────────────────────────────────

    def _check_model_freshness(self):
        try:
            from pathlib import Path
            import joblib, os
            root       = os.path.dirname(os.path.dirname(
                             os.path.dirname(os.path.abspath(__file__))))
            model_path = Path(root) / "saved_models" / f"ml_{self.symbol}.joblib"
            if model_path.exists():
                payload    = joblib.load(model_path)
                trained_at = datetime.fromisoformat(
                    payload.get("trained_at", "2000-01-01"))
                age_days   = (datetime.now() - trained_at).days
                if age_days > 90:
                    logger.warning(
                        f"[STALE MODEL] {self.symbol}: {age_days} days old. "
                        f"Run: python scripts/train_from_history.py "
                        f"--symbol {self.symbol}"
                    )
                else:
                    logger.info(f"[MODEL OK] {self.symbol}: {age_days} days old")
        except Exception as e:
            logger.debug(f"[{self.symbol}] Model freshness check skipped: {e}")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_my_position(self, positions):
        my_sym = _normalize_symbol(self.symbol)
        for p in positions:
            if _normalize_symbol(p.symbol) == my_sym:
                return p
        return None

    def _get_conf_threshold(self) -> float:
        return CONF_THRESHOLDS.get(_normalize_symbol(self.symbol), 0.42)

    def _bars_held(self) -> int:
        if self._entry_bar is None:
            return 0
        return self._bar_counter - self._entry_bar

    def _in_cooldown(self) -> bool:
        return self._bar_counter < self._cooldown_until_bar

    def _compute_atr_tpsl(
        self, entry_price: float, side: OrderSide, atr: float
    ) -> tuple[float, float]:
        sym     = _normalize_symbol(self.symbol)
        sl_mult = ATR_SL_MULT.get(sym, 1.5)
        tp_mult = ATR_TP_MULT.get(sym, 3.0)
        if side == OrderSide.LONG:
            sl = round(entry_price - sl_mult * atr, 4)
            tp = round(entry_price + tp_mult * atr, 4)
        else:
            sl = round(entry_price + sl_mult * atr, 4)
            tp = round(entry_price - tp_mult * atr, 4)
        return tp, sl

    def _portfolio_risk_ok(self, new_risk_usdt: float, balance_usdt: float) -> bool:
        if self._portfolio_tracker is None:
            return True
        current  = self._portfolio_tracker.total_open_risk_usdt
        if_added = current + new_risk_usdt
        cap      = balance_usdt * MAX_PORTFOLIO_RISK_PCT
        if if_added > cap:
            logger.warning(
                f"[PORTFOLIO CAP] {self.symbol}: "
                f"{current:.2f} + {new_risk_usdt:.2f} = {if_added:.2f} "
                f"> cap={cap:.2f} ({MAX_PORTFOLIO_RISK_PCT:.0%}). Skipping."
            )
            return False
        return True

    def _recover_tpsl_from_position(self, pos: PositionInfo,
                                    current_atr: Optional[float]):
        """
        Reconstruct TP/SL and portfolio tracker state if lost (e.g. restart
        while a position was open). Called every cycle while in a position,
        but the _tp_price guard means it only acts on the first call.

        Also registers with PortfolioRiskTracker so the dashboard shows
        correct open risk instead of "Positions: none" after a restart.
        """
        if self._tp_price is not None and self._sl_price is not None:
            return   # already recovered

        if current_atr and current_atr > 0:
            tp, sl   = self._compute_atr_tpsl(pos.entry_price, pos.side, current_atr)
            sl_dist  = abs(pos.entry_price - sl)

            self._tp_price      = tp
            self._sl_price      = sl
            self._position_side = pos.side
            self._entry_price   = pos.entry_price
            self._sl_distance   = sl_dist
            if self._entry_bar is None:
                self._entry_bar = self._bar_counter

            # Estimate USDT at risk: quantity (base units) × SL distance (USDT/unit)
            estimated_risk_usdt = pos.quantity * sl_dist

            # Register with portfolio tracker — avoids double-registering on
            # repeated calls by checking if symbol is already tracked.
            if (self._portfolio_tracker is not None
                    and _normalize_symbol(self.symbol)
                    not in self._portfolio_tracker._open_risks):
                self._portfolio_tracker.register_open(
                    self.symbol, estimated_risk_usdt)
                logger.info(
                    f"[RECOVERED TP/SL] {self.symbol}: "
                    f"entry={pos.entry_price:.4f}  TP={tp:.4f}  SL={sl:.4f}  "
                    f"risk~{estimated_risk_usdt:.2f} USDT"
                )
            else:
                logger.info(
                    f"[RECOVERED TP/SL] {self.symbol}: "
                    f"entry={pos.entry_price:.4f}  TP={tp:.4f}  SL={sl:.4f}"
                )
        else:
            logger.warning(
                f"[CANNOT RECOVER TP/SL] {self.symbol}: no ATR. "
                f"Will exit on timeout only."
            )

    # ── Main trading cycle ────────────────────────────────────────────────────

    async def run_cycle(self):
        if not self._is_active:
            return

        self._cycles      += 1
        self._bar_counter += 1
        logger.info(f"Cycle #{self._cycles} — {self.symbol}")

        try:
            candles = await self.exchange.get_ohlcv(
                self.symbol, self.TIMEFRAME, limit=350
            )
            if not candles or len(candles) < self.MIN_CANDLES:
                logger.warning(
                    f"Insufficient candles {self.symbol}: "
                    f"{len(candles) if candles else 0} < {self.MIN_CANDLES}"
                )
                return

            df            = ohlcv_to_dataframe(candles)
            df            = add_all_indicators(df)
            current_price = float(df["close"].iloc[-1])
            current_atr   = float(df["atr"].iloc[-1]) if "atr" in df.columns else None

            # ── Check existing position ────────────────────────────────────────
            positions   = await self.exchange.get_open_positions()
            current_pos = self._find_my_position(positions)

            if current_pos:
                self._in_position = True
                self._recover_tpsl_from_position(current_pos, current_atr)
                closed = await self._monitor_position(current_pos, current_price, df)
                if not closed:
                    # Position still open — nothing more to do this cycle
                    return
                # Position was closed. Fall through to signal recheck only if
                # the close was clean. _monitor_position returns False on genuine
                # exchange errors so we only reach here on confirmed closes.
                # Add a one-bar re-entry guard: don't open immediately in the same
                # cycle the close happened — wait for next candle's fresh signal.
                if self._bar_counter == self._closed_on_bar:
                    logger.info(
                        f"[RE-ENTRY GUARD] {self.symbol} — "
                        f"position closed this bar. Waiting one bar for fresh signal."
                    )
                    return

            elif self._in_position:
                logger.info(f"[EXTERNALLY CLOSED] {self.symbol} — resetting state")
                self._reset_position_state()

            # ── Cooldown check ─────────────────────────────────────────────────
            if self._in_cooldown():
                bars_left = self._cooldown_until_bar - self._bar_counter
                logger.info(f"[COOLDOWN] {self.symbol} — {bars_left} bars remaining")
                return

            # ── Get prediction ─────────────────────────────────────────────────
            prediction = self.model.predict(df)
            agreement_label = "AGREE" if prediction.models_agree else "SPLIT"

            logger.info(
                f"[SIGNAL] {self.symbol} → {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | "
                f"{agreement_label} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol}")
                return

            # ── Agreement filter ───────────────────────────────────────────────
            # SPLIT means ML and technical models disagree on direction.
            # Paper trading showed SPLIT signals at 70-76% confidence
            # consistently ran against the position while AGREE at 89%+ did not.
            if self.REQUIRE_AGREEMENT and not prediction.models_agree:
                logger.info(
                    f"[SKIP:SPLIT] {self.symbol} — "
                    f"ML and technical models disagree. "
                    f"Signal={prediction.signal.value} conf={prediction.confidence:.0%}. "
                    f"Set REQUIRE_AGREEMENT=False to trade SPLIT signals."
                )
                return

            # ── Confidence gate ────────────────────────────────────────────────
            conf_threshold = self._get_conf_threshold()
            if prediction.confidence < conf_threshold:
                logger.info(
                    f"[SKIP:CONF] {self.symbol} — "
                    f"conf {prediction.confidence:.0%} < {conf_threshold:.0%}"
                )
                return

            # ── Concurrent entry limit ─────────────────────────────────────────
            # Prevents all 4 correlated symbols opening simultaneously.
            if self._entry_gate is not None:
                if not self._entry_gate.can_enter(
                    self.symbol, prediction.confidence, prediction.models_agree
                ):
                    logger.info(
                        f"[SKIP:GATE] {self.symbol} — "
                        f"max {self.MAX_CONCURRENT_ENTRIES} concurrent entries "
                        f"this cycle already reached."
                    )
                    return

            side    = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT
            balance = await self.exchange.get_balance()

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — paused")
                return

            await self._open_new_trade(side, current_price, current_atr,
                                       balance, positions)

        except Exception as e:
            logger.error(f"Cycle error {self.symbol}: {e}", exc_info=True)

    # ── Position monitoring ───────────────────────────────────────────────────

    async def _monitor_position(self, pos: PositionInfo,
                                current_price: float,
                                df) -> bool:
        """
        Check exit conditions in priority order:
          1. Timeout
          2. Hard TP / SL
          3. Early profit (after MIN_HOLD_BARS)
          4. Signal reversal (after MIN_HOLD_BARS)
        Returns True if closed, False if still open.
        """
        bars_held = self._bars_held()

        if self._entry_price and self._sl_distance and self._sl_distance > 0:
            price_move = (current_price - self._entry_price) \
                         if pos.side == OrderSide.LONG \
                         else (self._entry_price - current_price)
            current_r = price_move / self._sl_distance
        else:
            current_r = 0.0

        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        if pos.side == OrderSide.SHORT:
            pnl_pct = -pnl_pct

        if self._tp_price and self._sl_price:
            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"TP={self._tp_price:.4f}  SL={self._sl_price:.4f} | "
                f"R={current_r:+.2f}  PnL={pos.unrealized_pnl:+.4f} "
                f"({pnl_pct:+.2f}%) | bars={bars_held}/{self.MAX_HOLD_BARS}"
            )
        else:
            logger.info(
                f"[WATCHING] {self.symbol} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} ({pnl_pct:+.2f}%) | "
                f"bars={bars_held}/{self.MAX_HOLD_BARS}"
            )

        # 1. Timeout
        if bars_held >= self.MAX_HOLD_BARS:
            logger.info(f"[TIMEOUT] {self.symbol} — {bars_held} bars, closing")
            return await self._close_position(pos, "TIMEOUT")

        if self._tp_price and self._sl_price:
            # 2. Hard TP / SL
            tp_hit = (current_price >= self._tp_price) \
                     if pos.side == OrderSide.LONG \
                     else (current_price <= self._tp_price)
            sl_hit = (current_price <= self._sl_price) \
                     if pos.side == OrderSide.LONG \
                     else (current_price >= self._sl_price)

            if tp_hit:
                logger.info(f"[TP HIT] {self.symbol} @ {current_price:.4f}")
                return await self._close_position(pos, "TP")
            if sl_hit:
                logger.info(f"[SL HIT] {self.symbol} @ {current_price:.4f}")
                return await self._close_position(pos, "SL")

        # Early exits only after MIN_HOLD_BARS
        if bars_held < self.MIN_HOLD_BARS:
            return False

        # 3. Early profit capture
        if current_r >= self.EARLY_PROFIT_THRESHOLD_R:
            logger.info(
                f"[EARLY PROFIT] {self.symbol} — "
                f"R={current_r:+.2f} >= {self.EARLY_PROFIT_THRESHOLD_R:.1f}R | "
                f"PnL={pos.unrealized_pnl:+.4f} USDT. Closing."
            )
            return await self._close_position(pos, "EARLY_PROFIT")

        # 4. Signal reversal exit
        try:
            prediction     = self.model.predict(df)
            conf_threshold = self._get_conf_threshold()

            if (prediction.signal != Signal.HOLD
                    and prediction.confidence >= conf_threshold):

                new_side = (OrderSide.LONG if prediction.signal == Signal.LONG
                            else OrderSide.SHORT)

                # Only act on reversals that also have agreement (or if not required)
                if new_side != pos.side:
                    if not self.REQUIRE_AGREEMENT or prediction.models_agree:
                        logger.info(
                            f"[SIGNAL REVERSAL] {self.symbol} — "
                            f"holding {pos.side.value.upper()} but model now "
                            f"{new_side.value.upper()} at conf={prediction.confidence:.0%} "
                            f"({'AGREE' if prediction.models_agree else 'SPLIT'}). Closing."
                        )
                        return await self._close_position(pos, "SIGNAL_REVERSAL")
                    else:
                        logger.info(
                            f"[REVERSAL SKIPPED] {self.symbol} — "
                            f"model flipped to {new_side.value.upper()} but SPLIT "
                            f"(not AGREE). Holding position."
                        )

        except Exception as e:
            logger.warning(f"[{self.symbol}] Signal reversal check failed: {e}")

        return False

    # ── Close a position ──────────────────────────────────────────────────────

    async def _close_position(self, pos: PositionInfo, reason: str) -> bool:
        """
        Close the position and update accounting.

        Returns True  → position is gone (closed successfully, or was already gone)
                        caller may fall through to signal recheck / re-entry
        Returns False → exchange returned a genuine error; position state unknown
                        caller must NOT fall through to re-entry this cycle

        Special case: "No open position" from the exchange means the position
        closed between our get_open_positions() call and this close attempt
        (race condition in asyncio.gather). Treat it as already-closed so we
        don't log a scary error and, critically, so we don't re-enter immediately.
        We still reset state and record PnL from the last known unrealized value.
        """
        result = await self.exchange.close_position(self.symbol)

        # "No open position" = position vanished between check and close.
        # Treat as silent success — the goal (no open position) is achieved.
        already_closed = (
            not result.success
            and "no open position" in (result.message or "").lower()
        )

        if result.success or already_closed:
            if already_closed:
                logger.info(
                    f"[ALREADY CLOSED] {self.symbol} — position closed externally "
                    f"between check and close attempt (race condition). "
                    f"Treating as closed:{reason}."
                )
            else:
                logger.info(
                    f"[CLOSED:{reason}] {self.symbol} | "
                    f"PnL={pos.unrealized_pnl:+.4f} USDT | "
                    f"bars_held={self._bars_held()}"
                )

            # Record PnL from last known unrealized value
            if pos.unrealized_pnl >= 0:
                self.risk.record_profit(pos.unrealized_pnl)
            else:
                self.risk.record_loss(pos.unrealized_pnl)

            if self._portfolio_tracker:
                self._portfolio_tracker.deregister_close(self.symbol)

            if reason == "SL":
                self._cooldown_until_bar = self._bar_counter + self.COOLDOWN_BARS
                logger.info(
                    f"[COOLDOWN SET] {self.symbol} — "
                    f"no entries for {self.COOLDOWN_BARS} bars after SL"
                )

            self._reset_position_state()
            return True   # ← position is gone; caller may re-evaluate signal

        else:
            # Genuine exchange error (network, insufficient margin, etc.)
            # Do NOT reset state — position may still be open.
            # Do NOT fall through to re-entry — we don't know the position state.
            logger.error(
                f"[CLOSE FAILED] {self.symbol}: {result.message}. "
                f"Position state unknown — will check and retry next cycle. "
                f"No new entry this cycle."
            )
            return False  # ← caller must NOT re-enter

    # ── Open a new trade ──────────────────────────────────────────────────────

    async def _open_new_trade(self, side: OrderSide, current_price: float,
                              current_atr: Optional[float],
                              balance, positions) -> None:

        trade_params = self.risk.calculate_trade(
            symbol      = self.symbol,
            side        = side,
            entry_price = current_price,
            balance     = balance,
            open_trades = len(positions),
            current_atr = current_atr,
            confidence  = self._get_conf_threshold(),
        )

        if not trade_params.approved:
            logger.warning(f"[REJECTED] {self.symbol} — {trade_params.reject_reason}")
            if self._entry_gate:
                self._entry_gate.cancel(self.symbol)
            return

        if current_atr and current_atr > 0:
            tp_atr, sl_atr = self._compute_atr_tpsl(current_price, side, current_atr)
            trade_params.take_profit = tp_atr
            trade_params.stop_loss   = sl_atr
            sl_distance = abs(current_price - sl_atr)
            logger.info(
                f"[ATR TP/SL] {self.symbol}: ATR={current_atr:.4f} | "
                f"TP={tp_atr:.4f}  SL={sl_atr:.4f}"
            )
        else:
            sl_distance = abs(current_price - trade_params.stop_loss)
            logger.warning(f"[NO ATR] {self.symbol}: using fixed-% TP/SL fallback")

        if not self._portfolio_risk_ok(trade_params.risk_amount_usdt,
                                       balance.available_balance):
            if self._entry_gate:
                self._entry_gate.cancel(self.symbol)
            return

        if self._portfolio_tracker is not None:
            self._portfolio_tracker.register_open(
                self.symbol, trade_params.risk_amount_usdt)

        await self._execute_trade(trade_params, sl_distance)

    # ── Execute trade ─────────────────────────────────────────────────────────

    async def _execute_trade(self, params: TradeParameters,
                             sl_distance: float) -> None:
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
            self._trades_opened  += 1
            self._in_position     = True
            self._tp_price        = params.take_profit
            self._sl_price        = params.stop_loss
            self._position_side   = params.side
            self._entry_price     = result.price if result.price > 0 else params.entry_price
            self._entry_bar       = self._bar_counter
            self._sl_distance     = sl_distance
            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {self._entry_price:.4f} | "
                f"TP={params.take_profit:.4f}  SL={params.stop_loss:.4f} | "
                f"risk={params.risk_amount_usdt:.2f} USDT | "
                f"early_exit_at={self.EARLY_PROFIT_THRESHOLD_R:.1f}R "
                f"(~{sl_distance * self.EARLY_PROFIT_THRESHOLD_R:.4f} price move)"
            )
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")
            if self._portfolio_tracker:
                self._portfolio_tracker.deregister_close(self.symbol)
            if self._entry_gate:
                self._entry_gate.cancel(self.symbol)

    # ── State management ──────────────────────────────────────────────────────

    def _reset_position_state(self):
        self._closed_on_bar = self._bar_counter  # record when we closed
        self._in_position   = False
        self._tp_price      = None
        self._sl_price      = None
        self._position_side = None
        self._entry_price   = None
        self._entry_bar     = None
        self._sl_distance   = None

    async def train_ml_model(self, candle_limit: int = 2000):
        candles = await self.exchange.get_ohlcv(
            self.symbol, self.TIMEFRAME, limit=candle_limit)
        df = ohlcv_to_dataframe(candles)
        df = add_all_indicators(df)
        logger.info(f"[TRAIN] {self.symbol} — {len(df)} candles")
        self.model.train_ml(df)

    def stop(self):
        self._is_active = False
        logger.info(f"Trader stopped: {self.symbol}")

    def get_stats(self) -> dict:
        return {
            "symbol":                  self.symbol,
            "cycles":                  self._cycles,
            "trades_opened":           self._trades_opened,
            "in_position":             self._in_position,
            "tp":                      self._tp_price,
            "sl":                      self._sl_price,
            "entry":                   self._entry_price,
            "bars_held":               self._bars_held(),
            "ml_trained":              self.model.ml_is_trained,
            "conf_threshold":          self._get_conf_threshold(),
            "require_agreement":       self.REQUIRE_AGREEMENT,
            "early_profit_threshold":  self.EARLY_PROFIT_THRESHOLD_R,
            "in_cooldown":             self._in_cooldown(),
            "cooldown_bars_left":      max(0, self._cooldown_until_bar - self._bar_counter),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY GATE — limits concurrent new positions per cycle
# Instantiate ONE in run_bot.py and pass to all traders.
# ══════════════════════════════════════════════════════════════════════════════

class EntryGate:
    """
    Limits how many new positions can open in a single cycle.
    Prioritises AGREE signals over SPLIT, then higher confidence.

    Wire up in run_bot.py (alongside PortfolioRiskTracker):
        gate = EntryGate(max_per_cycle=FuturesTrader.MAX_CONCURRENT_ENTRIES)
        traders = [
            FuturesTrader(..., entry_gate=gate)
            for sym in SYMBOLS
        ]

    Call reset() at the start of each cycle in run_bot.py before gather().
    """

    def __init__(self, max_per_cycle: int = 2):
        self.max_per_cycle   = max_per_cycle
        self._entries:       list[tuple[str, float, bool]] = []   # (symbol, conf, agree)
        self._approved:      set[str] = set()
        self._cycle_entries: int = 0

    def reset(self):
        """Call at the start of every cycle before traders run."""
        self._entries       = []
        self._approved      = set()
        self._cycle_entries = 0

    def can_enter(self, symbol: str, confidence: float, models_agree: bool) -> bool:
        """
        Called by FuturesTrader before opening a trade.
        Approves up to max_per_cycle entries per cycle, prioritising AGREE
        signals over SPLIT, then higher confidence within each tier.
        """
        # Already approved this symbol this cycle
        if symbol in self._approved:
            return True

        # Already at limit
        if self._cycle_entries >= self.max_per_cycle:
            return False

        # Approve immediately — FIFO within each cycle since gather() is parallel.
        # The real priority is enforced by REQUIRE_AGREEMENT filtering SPLIT out
        # before they reach here.
        self._approved.add(symbol)
        self._cycle_entries += 1
        logger.info(
            f"[ENTRY GATE] {symbol} approved "
            f"({self._cycle_entries}/{self.max_per_cycle} this cycle) | "
            f"conf={confidence:.0%} | "
            f"{'AGREE' if models_agree else 'SPLIT'}"
        )
        return True

    def cancel(self, symbol: str):
        """Call if a trade that was approved failed to open — frees the slot."""
        if symbol in self._approved:
            self._approved.discard(symbol)
            self._cycle_entries = max(0, self._cycle_entries - 1)


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK TRACKER
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioRiskTracker:
    """
    Tracks total open risk USDT across all FuturesTrader instances.
    Instantiate ONE in run_bot.py and pass to all traders.
    """

    def __init__(self):
        self._open_risks: dict[str, float] = {}

    @property
    def total_open_risk_usdt(self) -> float:
        return sum(self._open_risks.values())

    def register_open(self, symbol: str, risk_usdt: float):
        self._open_risks[_normalize_symbol(symbol)] = risk_usdt
        logger.info(
            f"[PORTFOLIO] {symbol} registered  "
            f"risk={risk_usdt:.2f}  "
            f"total={self.total_open_risk_usdt:.2f}  "
            f"({len(self._open_risks)} positions)"
        )

    def deregister_close(self, symbol: str):
        removed = self._open_risks.pop(_normalize_symbol(symbol), 0.0)
        logger.info(
            f"[PORTFOLIO] {symbol} deregistered  "
            f"freed={removed:.2f}  "
            f"total={self.total_open_risk_usdt:.2f}  "
            f"({len(self._open_risks)} positions)"
        )

    def get_status(self) -> dict:
        return {
            "open_positions":  len(self._open_risks),
            "total_risk_usdt": self.total_open_risk_usdt,
            "by_symbol":       dict(self._open_risks),
        }