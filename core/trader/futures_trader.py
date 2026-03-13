"""
core/trader/futures_trader.py
──────────────────────────────
Main trading orchestrator.

Changes in this version vs previous refactor:
  1. Per-symbol confidence thresholds (ETH/SOL=0.38, BTC/BNB=0.42)
     — ETH and SOL models are HOLD-biased; lower threshold recovers valid signals
  2. ATR-based TP/SL using the same atr_mult values as the label engine
     — Fixes the 60%+ TIMEOUT exit rate on BTC/BNB (TP/SL were too tight)
  3. Portfolio-level risk cap (max 3% total open risk across all symbols)
     — BTC/ETH/BNB are highly correlated; simultaneous SLs = 3x loss in one move
  4. Timeout guard: close position if held > MAX_HOLD_BARS bars
  5. Model freshness check at startup (warn if model > 90 days old)
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Optional

from core.exchange.base_exchange import BaseExchange, OrderSide, PositionInfo
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
# SL distance = ATR_SL_MULT × ATR  (1× risk unit)
# TP distance = ATR_TP_MULT × ATR  (2× risk unit → 2:1 RR)
ATR_SL_MULT: dict[str, float] = {
    "BTCUSDT": 1.5,
    "ETHUSDT": 1.8,
    "BNBUSDT": 1.6,
    "SOLUSDT": 2.0,
}
ATR_TP_MULT: dict[str, float] = {sym: v * 2.0 for sym, v in ATR_SL_MULT.items()}

# ── Portfolio risk cap ────────────────────────────────────────────────────────
# Block new entries if total open risk across all symbols >= this % of balance.
MAX_PORTFOLIO_RISK_PCT = 0.03   # 3%

# ── Timeout ───────────────────────────────────────────────────────────────────
MAX_HOLD_BARS = 32   # 32 × 15m = 8 hours — force close if still open


class FuturesTrader:

    TIMEFRAME   = "15m"
    MIN_CANDLES = 250

    def __init__(self, exchange, symbol: str, risk_manager: RiskManager,
                 portfolio_risk_tracker=None, **kwargs):
        self.exchange  = exchange
        self.symbol    = symbol
        self.risk      = risk_manager
        self.model     = HybridModel(symbol=symbol)

        # Shared tracker — injected by the bot orchestrator (main.py).
        # Pass the same PortfolioRiskTracker instance to all traders.
        # If None (single-symbol mode), portfolio cap is disabled.
        self._portfolio_tracker = portfolio_risk_tracker

        self._is_active       = True
        self._cycles          = 0
        self._trades_opened   = 0
        self._in_position     = False
        self._tp_price:       Optional[float]     = None
        self._sl_price:       Optional[float]     = None
        self._position_side:  Optional[OrderSide] = None
        self._entry_price:    Optional[float]     = None
        self._entry_bar:      Optional[int]       = None
        self._bar_counter:    int                 = 0

        self._check_model_freshness()

    # ── Model freshness check ─────────────────────────────────────────────────

    def _check_model_freshness(self):
        """Log a warning if the saved model is more than 90 days old."""
        try:
            from pathlib import Path
            import joblib
            import os
            root       = os.path.dirname(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__))))
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
                    logger.info(
                        f"[MODEL OK] {self.symbol}: {age_days} days old")
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

    def _compute_atr_tpsl(
        self, entry_price: float, side: OrderSide, atr: float
    ) -> tuple[float, float]:
        """
        Return (take_profit, stop_loss) using ATR multipliers that match the
        label engine — this aligns live TP/SL with the signal's 2-hour horizon.

        Example for BTC LONG at $50,000 with ATR=$500:
            SL = 50,000 - 1.5 × 500 = $49,250
            TP = 50,000 + 3.0 × 500 = $51,500
        """
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
        """
        Return True if adding this trade is within the portfolio risk cap.

        BTC/ETH/BNB correlate at ~0.80+. Running all three at 1% individual risk
        is effectively 3% correlated risk — not 1%. The cap blocks new entries
        when total open risk >= 3% of balance so that a simultaneous multi-symbol
        SL event can never exceed a single controlled drawdown.
        """
        if self._portfolio_tracker is None:
            return True

        current   = self._portfolio_tracker.total_open_risk_usdt
        if_added  = current + new_risk_usdt
        cap       = balance_usdt * MAX_PORTFOLIO_RISK_PCT

        if if_added > cap:
            logger.warning(
                f"[PORTFOLIO CAP] {self.symbol}: "
                f"current={current:.2f} + new={new_risk_usdt:.2f} "
                f"= {if_added:.2f} > cap={cap:.2f} "
                f"({MAX_PORTFOLIO_RISK_PCT:.0%} of {balance_usdt:.2f}). "
                f"Skipping entry."
            )
            return False

        return True

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
                await self._monitor_tpsl(current_pos, current_price)
                return

            if self._in_position:
                logger.info(f"[CLOSED] {self.symbol} — ready for next trade")
                self._reset_position_state()

            # ── Get prediction ─────────────────────────────────────────────────
            prediction = self.model.predict(df)
            logger.info(
                f"[SIGNAL] {self.symbol} → {prediction.signal.value} | "
                f"conf={prediction.confidence:.0%} | {prediction.source}"
            )

            if prediction.signal == Signal.HOLD:
                logger.info(f"[HOLD] {self.symbol}")
                return

            # ── Per-symbol confidence gate ─────────────────────────────────────
            conf_threshold = self._get_conf_threshold()
            if prediction.confidence < conf_threshold:
                logger.info(
                    f"[SKIP] {self.symbol} — conf {prediction.confidence:.0%} "
                    f"< {conf_threshold:.0%} ({self.symbol} threshold)"
                )
                return

            side    = OrderSide.LONG if prediction.signal == Signal.LONG else OrderSide.SHORT
            balance = await self.exchange.get_balance()

            if self.risk.is_daily_limit_hit(balance):
                logger.warning(f"[DAILY LIMIT] {self.symbol} — paused")
                return

            # ── Calculate position size ────────────────────────────────────────
            trade_params = self.risk.calculate_trade(
                symbol      = self.symbol,
                side        = side,
                entry_price = current_price,
                balance     = balance,
                open_trades = len(positions),
                current_atr = current_atr,
                confidence  = prediction.confidence,
            )

            if not trade_params.approved:
                logger.warning(
                    f"[REJECTED] {self.symbol} — {trade_params.reject_reason}")
                return

            # ── ATR-based TP/SL override ───────────────────────────────────────
            # Replace the risk manager's TP/SL with ATR-based prices that match
            # the label engine's signal horizon (2h, ATR multipliers per symbol).
            if current_atr and current_atr > 0:
                tp_atr, sl_atr = self._compute_atr_tpsl(
                    current_price, side, current_atr)
                trade_params.take_profit = tp_atr
                trade_params.stop_loss   = sl_atr
                logger.info(
                    f"[ATR TP/SL] {self.symbol}: ATR={current_atr:.4f} | "
                    f"TP={tp_atr:.4f}  SL={sl_atr:.4f}"
                )
            else:
                logger.warning(
                    f"[NO ATR] {self.symbol}: using fixed-% TP/SL fallback")

            # ── Portfolio risk cap ─────────────────────────────────────────────
            if not self._portfolio_risk_ok(
                trade_params.risk_amount_usdt,
                balance.available_balance
            ):
                return

            # Register open risk before execution
            if self._portfolio_tracker is not None:
                self._portfolio_tracker.register_open(
                    self.symbol, trade_params.risk_amount_usdt)

            await self._execute_trade(trade_params)

        except Exception as e:
            logger.error(f"Cycle error {self.symbol}: {e}", exc_info=True)

    # ── Position monitoring ───────────────────────────────────────────────────

    async def _monitor_tpsl(self, pos: PositionInfo, current_price: float):
        pnl_pct = ((current_price - pos.entry_price) / pos.entry_price) * 100
        if pos.side == OrderSide.SHORT:
            pnl_pct = -pnl_pct

        # Timeout guard
        bars_held = 0
        if self._entry_bar is not None:
            bars_held = self._bar_counter - self._entry_bar
            if bars_held >= MAX_HOLD_BARS:
                logger.info(
                    f"[TIMEOUT] {self.symbol} — held {bars_held} bars "
                    f"(max={MAX_HOLD_BARS}), closing at market"
                )
                result = await self.exchange.close_position(self.symbol)
                if result.success:
                    if pos.unrealized_pnl >= 0:
                        self.risk.record_profit(pos.unrealized_pnl)
                    else:
                        self.risk.record_loss(pos.unrealized_pnl)
                    if self._portfolio_tracker:
                        self._portfolio_tracker.deregister_close(self.symbol)
                self._reset_position_state()
                return

        if self._tp_price and self._sl_price:
            tp_hit = (
                (current_price >= self._tp_price)
                if pos.side == OrderSide.LONG
                else (current_price <= self._tp_price)
            )
            sl_hit = (
                (current_price <= self._sl_price)
                if pos.side == OrderSide.LONG
                else (current_price >= self._sl_price)
            )

            logger.info(
                f"[WATCHING] {self.symbol} {pos.side.value.upper()} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"TP={self._tp_price:.4f}  SL={self._sl_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} ({pnl_pct:+.2f}%) | "
                f"bars={bars_held}/{MAX_HOLD_BARS}"
            )

            if tp_hit:
                logger.info(f"[TP HIT] {self.symbol}")
                result = await self.exchange.close_position(self.symbol)
                if result.success:
                    self.risk.record_profit(pos.unrealized_pnl)
                    if self._portfolio_tracker:
                        self._portfolio_tracker.deregister_close(self.symbol)
                self._reset_position_state()
            elif sl_hit:
                logger.info(f"[SL HIT] {self.symbol}")
                result = await self.exchange.close_position(self.symbol)
                if result.success:
                    self.risk.record_loss(pos.unrealized_pnl)
                    if self._portfolio_tracker:
                        self._portfolio_tracker.deregister_close(self.symbol)
                self._reset_position_state()
        else:
            logger.info(
                f"[WATCHING] {self.symbol} | "
                f"entry={pos.entry_price:.4f} now={current_price:.4f} | "
                f"PnL={pos.unrealized_pnl:+.4f} ({pnl_pct:+.2f}%)"
            )

    # ── Trade execution ───────────────────────────────────────────────────────

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
            self._trades_opened  += 1
            self._in_position     = True
            self._tp_price        = params.take_profit
            self._sl_price        = params.stop_loss
            self._position_side   = params.side
            self._entry_price     = result.price
            self._entry_bar       = self._bar_counter
            logger.info(
                f"[OPENED] #{self._trades_opened} {params.symbol} "
                f"{params.side.value.upper()} | "
                f"qty={params.quantity} @ {result.price:.4f} | "
                f"TP={params.take_profit:.4f}  SL={params.stop_loss:.4f} | "
                f"risk={params.risk_amount_usdt:.2f} USDT"
            )
        else:
            logger.error(f"[FAILED] {params.symbol}: {result.message}")
            if self._portfolio_tracker:
                self._portfolio_tracker.deregister_close(self.symbol)

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _reset_position_state(self):
        self._in_position   = False
        self._tp_price      = None
        self._sl_price      = None
        self._position_side = None
        self._entry_price   = None
        self._entry_bar     = None

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
            "symbol":          self.symbol,
            "cycles":          self._cycles,
            "trades_opened":   self._trades_opened,
            "in_position":     self._in_position,
            "tp":              self._tp_price,
            "sl":              self._sl_price,
            "entry":           self._entry_price,
            "bars_held":       (self._bar_counter - self._entry_bar
                                if self._entry_bar is not None else 0),
            "ml_trained":      self.model.ml_is_trained,
            "conf_threshold":  self._get_conf_threshold(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO RISK TRACKER
# Instantiate ONE of these in main.py and pass it to every FuturesTrader.
# ══════════════════════════════════════════════════════════════════════════════

class PortfolioRiskTracker:
    """
    Tracks total open risk USDT across all FuturesTrader instances.

    Wire up in main.py / bot runner:

        tracker = PortfolioRiskTracker()
        traders = [
            FuturesTrader(exchange, sym, risk_mgr, portfolio_risk_tracker=tracker)
            for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT"]
        ]

    The FuturesTrader calls register_open() before opening and
    deregister_close() when a position is closed or fails to open.
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
            f"total_open={self.total_open_risk_usdt:.2f}  "
            f"({len(self._open_risks)} positions)"
        )

    def deregister_close(self, symbol: str):
        removed = self._open_risks.pop(_normalize_symbol(symbol), 0.0)
        logger.info(
            f"[PORTFOLIO] {symbol} deregistered  "
            f"freed={removed:.2f}  "
            f"total_open={self.total_open_risk_usdt:.2f}  "
            f"({len(self._open_risks)} positions)"
        )

    def get_status(self) -> dict:
        return {
            "open_positions":  len(self._open_risks),
            "total_risk_usdt": self.total_open_risk_usdt,
            "by_symbol":       dict(self._open_risks),
        }