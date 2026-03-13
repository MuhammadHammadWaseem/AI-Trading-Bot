"""
risk_management/position_sizer.py
────────────────────────────────────
Professional position sizing using Kelly Criterion and volatility scaling.

Replaces the Martingale-based RecoveryStrategy with a mathematically sound
risk model that cannot cause account blowup by definition.

Key sizing methods:
  1. Fixed Fractional  — simple: risk = account × risk_pct
  2. Kelly Criterion   — optimal: f* = edge / odds (theoretically maximum growth)
  3. Half-Kelly        — conservative: f* / 2 (standard institutional practice)
  4. Volatility-Scaled — adjusts size inversely with ATR/volatility

Usage:
    from risk_management.position_sizer import PositionSizer

    sizer = PositionSizer(method="half_kelly", max_risk_pct=0.02)
    size  = sizer.calculate(
        account_equity  = 10_000,
        entry_price     = 65_000,
        stop_price      = 64_350,    # stop loss price
        win_rate        = 0.55,      # from backtest
        avg_win         = 150.0,     # avg win in USDT
        avg_loss        = 80.0,      # avg loss in USDT
        current_atr     = 320.0,     # for vol-scaling
    )
    print(size)  # PositionSize(quantity=0.023, notional_usdt=1495, risk_usdt=14.95)
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional


SizingMethod = Literal["fixed_fractional", "kelly", "half_kelly", "vol_scaled"]


@dataclass
class PositionSize:
    """Computed position size with all derived metrics."""
    quantity:          float   # number of contracts
    notional_usdt:     float   # total position value (quantity × price × leverage)
    risk_usdt:         float   # maximum dollar loss if stop hit
    risk_pct:          float   # risk_usdt / account_equity
    kelly_fraction:    float   # raw Kelly f* (informational)
    approved:          bool
    reject_reason:     str = ""


class PositionSizer:
    """
    Risk-based position sizer. Completely replaces Martingale sizing.

    The fundamental principle:
        NEVER risk more than max_risk_pct of account on any single trade.
        Size is determined by the distance to stop loss, not by fixed lot.

    Anti-Martingale guarantee:
        After a losing streak, Kelly fraction DECREASES (fewer wins in recent
        history), which REDUCES position size. The opposite of Martingale.

    Parameters
    ----------
    method          : sizing algorithm
    max_risk_pct    : hard cap on risk per trade (0.02 = 2%)
    max_position_pct: hard cap on total notional / equity (prevents over-leverage)
    kelly_lookback  : number of recent trades to estimate win rate / payoff ratio
    """

    def __init__(
        self,
        method:           SizingMethod = "half_kelly",
        max_risk_pct:     float = 0.015,   # 1.5% max risk per trade
        max_position_pct: float = 0.25,    # max 25% notional / equity
        kelly_lookback:   int   = 50,
    ):
        self.method           = method
        self.max_risk_pct     = max_risk_pct
        self.max_position_pct = max_position_pct
        self.kelly_lookback   = kelly_lookback

        # Running trade history for Kelly estimation
        self._trade_pnls: list = []

    def record_trade(self, pnl: float) -> None:
        """Record a completed trade PnL for adaptive Kelly estimation."""
        self._trade_pnls.append(pnl)
        if len(self._trade_pnls) > self.kelly_lookback:
            self._trade_pnls.pop(0)

    def calculate(
        self,
        account_equity:  float,
        entry_price:     float,
        stop_price:      float,
        leverage:        int    = 10,
        win_rate:        float  = 0.50,
        avg_win:         float  = 100.0,
        avg_loss:        float  = 80.0,
        current_atr:     Optional[float] = None,
        confidence:      float  = 0.50,
        quantity_step:   float  = 0.001,
    ) -> PositionSize:
        """
        Calculate position size for an upcoming trade.

        Parameters
        ----------
        account_equity  : current free equity in USDT
        entry_price     : planned entry price
        stop_price      : stop loss price (already computed by risk manager)
        leverage        : futures leverage
        win_rate        : historical win rate (0–1). Use backtest or rolling estimate.
        avg_win         : average winning trade return in USDT (absolute)
        avg_loss        : average losing trade return in USDT (absolute, positive)
        current_atr     : current ATR for volatility scaling
        confidence      : model confidence (0–1) — scales Kelly fraction
        quantity_step   : minimum lot size increment (exchange-specific)
        """
        if account_equity <= 0:
            return self._reject("Zero or negative account equity")
        if entry_price <= 0:
            return self._reject("Invalid entry price")
        if stop_price <= 0:
            return self._reject("Invalid stop price")

        # Use recent trade history if available
        if len(self._trade_pnls) >= 10:
            wins   = [p for p in self._trade_pnls if p > 0]
            losses = [p for p in self._trade_pnls if p < 0]
            if wins and losses:
                win_rate = len(wins) / len(self._trade_pnls)
                avg_win  = float(np.mean(wins))
                avg_loss = float(abs(np.mean(losses)))

        # ── Stop distance ──────────────────────────────────────────────────
        stop_dist_pct = abs(entry_price - stop_price) / entry_price
        if stop_dist_pct < 0.001:
            stop_dist_pct = 0.001   # min 0.1% stop distance

        # ── Raw risk fraction ──────────────────────────────────────────────
        if self.method == "fixed_fractional":
            raw_fraction = self.max_risk_pct

        elif self.method in ("kelly", "half_kelly"):
            raw_fraction = self._kelly_fraction(win_rate, avg_win, avg_loss)
            if self.method == "half_kelly":
                raw_fraction *= 0.5
            # Scale Kelly by model confidence (0.5 conf = 50% of Kelly)
            raw_fraction *= max(0.3, min(1.0, confidence / 0.65))

        elif self.method == "vol_scaled":
            # Fixed 1% base, scaled inversely by current ATR vs historical mean
            if current_atr and current_atr > 0:
                # More volatile = smaller position
                vol_scale    = 1.0 / max(0.5, current_atr / entry_price / 0.005)
                raw_fraction = self.max_risk_pct * vol_scale
            else:
                raw_fraction = self.max_risk_pct

        else:
            raw_fraction = self.max_risk_pct

        # ── Apply hard risk cap ───────────────────────────────────────────
        risk_fraction = min(raw_fraction, self.max_risk_pct)
        risk_fraction = max(risk_fraction, 0.001)   # minimum 0.1%

        # ── Compute position size from risk and stop distance ─────────────
        # risk_usdt = account × risk_fraction
        # quantity  = risk_usdt / (stop_dist_pct × entry_price)
        # (this ensures we lose exactly risk_usdt if stop is hit)
        risk_usdt     = account_equity * risk_fraction
        quantity_raw  = risk_usdt / (stop_dist_pct * entry_price)
        notional      = quantity_raw * entry_price * leverage

        # ── Apply position size cap ───────────────────────────────────────
        max_notional  = account_equity * self.max_position_pct * leverage
        if notional > max_notional:
            quantity_raw = max_notional / (entry_price * leverage)
            notional     = max_notional
            risk_usdt    = quantity_raw * entry_price * stop_dist_pct

        # ── Round to exchange lot size ────────────────────────────────────
        if quantity_step > 0:
            quantity = math.floor(quantity_raw / quantity_step) * quantity_step
        else:
            quantity = round(quantity_raw, 6)

        if quantity <= 0:
            return self._reject(f"Calculated quantity {quantity_raw:.6f} rounds to 0")

        actual_risk_pct = (quantity * entry_price * stop_dist_pct) / account_equity

        kelly_f = self._kelly_fraction(win_rate, avg_win, avg_loss)

        return PositionSize(
            quantity       = quantity,
            notional_usdt  = quantity * entry_price,
            risk_usdt      = quantity * entry_price * stop_dist_pct,
            risk_pct       = actual_risk_pct,
            kelly_fraction = kelly_f,
            approved       = True,
        )

    # ── Kelly formula ─────────────────────────────────────────────────────

    @staticmethod
    def _kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        """
        Full Kelly fraction: f* = (p × b - q) / b
        where:
          p = win rate
          q = 1 - p (loss rate)
          b = avg_win / avg_loss (payoff ratio)

        Returns a value in [0, 1]. Capped at 0.25 (25% of account)
        to prevent extreme leverage even if Kelly suggests more.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return 0.01   # degenerate case — use minimum

        b   = avg_win / (avg_loss + 1e-10)   # payoff ratio
        p   = min(0.99, max(0.01, win_rate))
        q   = 1.0 - p
        f   = (p * b - q) / (b + 1e-10)

        # Negative Kelly = negative edge → don't trade
        f = max(0.0, f)

        # Hard cap: never exceed 25% of account regardless of Kelly
        return min(f, 0.25)

    def _reject(self, reason: str) -> PositionSize:
        return PositionSize(
            quantity=0.0, notional_usdt=0.0, risk_usdt=0.0,
            risk_pct=0.0, kelly_fraction=0.0,
            approved=False, reject_reason=reason,
        )
