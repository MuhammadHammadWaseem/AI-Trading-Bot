"""
core/risk/risk_manager.py
──────────────────────────
Risk management — position sizing, TP/SL, daily loss limits.  REFACTORED.

Key changes from original:
  1. Delegates position sizing to PositionSizer (Kelly criterion)
  2. stop_loss_price is now required input (not computed from fixed %)
     — ATR-based stop price comes from signal or default fallback
  3. Martingale multiplier REMOVED — size is purely risk-based
  4. Added volatility-adjusted TP/SL using ATR
  5. record_trade() propagates to PositionSizer for adaptive Kelly
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from core.exchange.base_exchange import AccountBalance, OrderSide
from risk_management.position_sizer import PositionSizer, PositionSize
from config.logger import get_logger
from config.settings import RiskSettings

logger = get_logger(__name__)


@dataclass
class TradeParameters:
    symbol:             str
    side:               OrderSide
    quantity:           float
    leverage:           int
    entry_price:        float
    take_profit:        float
    stop_loss:          float
    position_size_usdt: float
    risk_amount_usdt:   float
    approved:           bool = True
    reject_reason:      str  = ""


class RiskManager:
    """
    Calculates safe position sizes and validates every trade.

    Position Sizing (replaces Martingale):
        quantity = risk_usdt / (stop_dist_pct × entry_price)
        where risk_usdt = equity × risk_fraction (Kelly or fixed)

    Stop Loss:
        ATR-based: stop = entry ∓ atr_sl_mult × ATR
        Fallback:  stop = entry ∓ entry × sl_pct / 100
    """

    ATR_TP_MULT = 2.0   # take profit = entry ± 2 × ATR
    ATR_SL_MULT = 1.0   # stop loss   = entry ∓ 1 × ATR

    def __init__(self, risk_settings: RiskSettings):
        self.settings       = risk_settings
        self._daily_loss    = 0.0
        self._session_start = 0.0
        self._sizer         = PositionSizer(
            method        = "half_kelly",
            max_risk_pct  = risk_settings.risk_per_trade_pct / 100.0,
        )

    def set_session_balance(self, balance: float) -> None:
        self._session_start = balance

    def record_loss(self, loss_usdt: float) -> None:
        if loss_usdt < 0:
            self._daily_loss += abs(loss_usdt)
            self._sizer.record_trade(loss_usdt)
            logger.warning(f"Daily loss: {self._daily_loss:.2f} USDT")

    def record_profit(self, profit_usdt: float) -> None:
        if profit_usdt > 0:
            self._daily_loss = max(0, self._daily_loss - profit_usdt)
            self._sizer.record_trade(profit_usdt)

    def is_daily_limit_hit(self, balance: AccountBalance) -> bool:
        if self._session_start <= 0:
            return False
        pct = self._daily_loss / self._session_start * 100
        if pct >= self.settings.max_daily_loss_pct:
            logger.error(f"Daily loss limit {pct:.1f}% >= {self.settings.max_daily_loss_pct}%")
            return True
        return False

    def calculate_trade(
        self,
        symbol:        str,
        side:          OrderSide,
        entry_price:   float,
        balance:       AccountBalance,
        open_trades:   int   = 0,
        leverage:      Optional[int]   = None,
        tp_pct:        Optional[float] = None,
        sl_pct:        Optional[float] = None,
        risk_pct:      Optional[float] = None,
        current_atr:   Optional[float] = None,   # NEW: ATR for dynamic TP/SL
        confidence:    float = 0.55,             # NEW: model confidence
    ) -> TradeParameters:

        lev   = leverage or self.settings.leverage
        tp_p  = tp_pct   or self.settings.take_profit_pct
        sl_p  = sl_pct   or self.settings.stop_loss_pct

        # Pre-trade checks
        if open_trades >= self.settings.max_open_trades:
            return self._reject(symbol, side, entry_price,
                                f"Max open trades ({self.settings.max_open_trades}) reached")
        if balance.available_balance <= 0:
            return self._reject(symbol, side, entry_price, "No available balance")
        if self.is_daily_limit_hit(balance):
            return self._reject(symbol, side, entry_price, "Daily loss limit hit")

        # ── ATR-based TP/SL (preferred) ───────────────────────────────────
        if current_atr and current_atr > 0:
            if side == OrderSide.LONG:
                take_profit = round(entry_price + self.ATR_TP_MULT * current_atr, 4)
                stop_loss   = round(entry_price - self.ATR_SL_MULT * current_atr, 4)
            else:
                take_profit = round(entry_price - self.ATR_TP_MULT * current_atr, 4)
                stop_loss   = round(entry_price + self.ATR_SL_MULT * current_atr, 4)
        else:
            # Fallback: fixed %
            if side == OrderSide.LONG:
                take_profit = round(entry_price * (1 + tp_p / 100), 4)
                stop_loss   = round(entry_price * (1 - sl_p / 100), 4)
            else:
                take_profit = round(entry_price * (1 - tp_p / 100), 4)
                stop_loss   = round(entry_price * (1 + sl_p / 100), 4)

        # ── Position sizing (Kelly-based) ─────────────────────────────────
        size: PositionSize = self._sizer.calculate(
            account_equity = balance.available_balance,
            entry_price    = entry_price,
            stop_price     = stop_loss,
            leverage       = lev,
            confidence     = confidence,
        )

        if not size.approved:
            return self._reject(symbol, side, entry_price, size.reject_reason)

        logger.info(
            f"Trade: {symbol} {side.value.upper()} | qty={size.quantity} | "
            f"lev={lev}x | TP={take_profit} SL={stop_loss} | "
            f"risk={size.risk_usdt:.2f} USDT ({size.risk_pct:.1%}) | "
            f"Kelly={size.kelly_fraction:.3f}"
        )

        return TradeParameters(
            symbol             = symbol,
            side               = side,
            quantity           = size.quantity,
            leverage           = lev,
            entry_price        = entry_price,
            take_profit        = take_profit,
            stop_loss          = stop_loss,
            position_size_usdt = size.notional_usdt,
            risk_amount_usdt   = size.risk_usdt,
            approved           = True,
        )

    def _reject(self, symbol, side, price, reason) -> TradeParameters:
        logger.warning(f"Trade rejected: {symbol} {side.value} — {reason}")
        return TradeParameters(
            symbol=symbol, side=side, quantity=0, leverage=1,
            entry_price=price, take_profit=price, stop_loss=price,
            position_size_usdt=0, risk_amount_usdt=0,
            approved=False, reject_reason=reason,
        )
