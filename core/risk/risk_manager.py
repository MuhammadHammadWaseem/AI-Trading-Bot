"""
core/risk/risk_manager.py
──────────────────────────
Risk management — position sizing, TP/SL calculation, daily loss limits.

FIX #4 — ATR-based TP/SL (replaces fixed percentage)
------------------------------------------------------
Problem: TP=2%, SL=1% are fixed percentages that ignore actual market
volatility.  During high-ATR sessions BTC moves 1% in 2 candles, so the
1% SL is hit almost immediately, and the 2% TP is never reached before the
32-bar timeout.  Almost all closes are TIMEOUTs instead of TP/SL hits.

Solution: TP and SL are now expressed as ATR multiples:
    SL = entry ± (ATR * SL_ATR_MULT)   default: 2.0×ATR
    TP = entry ± (ATR * TP_ATR_MULT)   default: 3.0×ATR  (1.5 R:R minimum)

The regime detector can further scale these multipliers via RegimeParams.

FIX #5 — Directional concentration filter
------------------------------------------
Problem: all 4 pairs open SHORT simultaneously — full correlated exposure.

Solution: PortfolioDirectionTracker counts open LONG/SHORT positions.
If adding this trade would put more than MAX_SAME_DIRECTION positions in
the same direction, the trade is rejected.  Default limit: 2.
This means the bot can hold BTC SHORT + ETH SHORT, but a third SHORT on
BNB will be blocked until one of the others closes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
import math

from core.exchange.base_exchange import AccountBalance, OrderSide
from config.logger import get_logger
from config.settings import RiskSettings

logger = get_logger(__name__)


# ── Trade parameters ──────────────────────────────────────────────────────────

@dataclass
class TradeParameters:
    """Fully computed trade parameters ready for execution."""
    symbol:             str
    side:               OrderSide
    quantity:           float
    leverage:           int
    entry_price:        float
    take_profit:        float
    stop_loss:          float
    position_size_usdt: float
    risk_amount_usdt:   float
    approved:           bool  = True
    reject_reason:      str   = ""


# ── Directional portfolio tracker ─────────────────────────────────────────────

class PortfolioDirectionTracker:
    """
    Tracks how many open positions are LONG vs SHORT.

    Call register_open()  when a trade executes.
    Call register_close() when a position closes.
    Call can_open()       before placing a new trade.
    """

    # Maximum same-direction positions at once (FIX #5)
    MAX_SAME_DIRECTION = 2

    def __init__(self):
        self._positions: Dict[str, OrderSide] = {}   # symbol → side

    def register_open(self, symbol: str, side: OrderSide):
        self._positions[symbol] = side
        logger.info(
            f"[PORTFOLIO] {symbol} registered {side.value} | "
            f"LONG={self._count(OrderSide.LONG)}  SHORT={self._count(OrderSide.SHORT)}"
        )

    def register_close(self, symbol: str):
        self._positions.pop(symbol, None)
        logger.info(
            f"[PORTFOLIO] {symbol} deregistered | "
            f"LONG={self._count(OrderSide.LONG)}  SHORT={self._count(OrderSide.SHORT)}"
        )

    def can_open(self, side: OrderSide) -> tuple[bool, str]:
        """Return (allowed, reason_if_not)."""
        current = self._count(side)
        if current >= self.MAX_SAME_DIRECTION:
            direction = "LONG" if side == OrderSide.LONG else "SHORT"
            return False, (
                f"Already {current} {direction} positions open "
                f"(max {self.MAX_SAME_DIRECTION})"
            )
        return True, ""

    def _count(self, side: OrderSide) -> int:
        return sum(1 for s in self._positions.values() if s == side)

    @property
    def open_count(self) -> int:
        return len(self._positions)


# ── Main risk manager ─────────────────────────────────────────────────────────

class RiskManager:
    """
    Calculates safe position sizes and validates trade parameters.

    TP/SL mode:
        If atr is provided → ATR-based levels (preferred).
        Otherwise          → fixed percentage fallback.

    Key ATR multipliers (can be scaled by regime):
        SL_ATR_MULT  = 2.0   (stop 2 ATR away from entry)
        TP_ATR_MULT  = 3.0   (target 3 ATR away — 1.5 R:R)
    """

    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 3.0

    def __init__(self, risk_settings: RiskSettings):
        self.settings              = risk_settings
        self._daily_loss_usdt      = 0.0
        self._session_start_balance = 0.0
        self.portfolio             = PortfolioDirectionTracker()

    def set_session_balance(self, balance: float):
        self._session_start_balance = balance

    def record_loss(self, loss_usdt: float):
        if loss_usdt < 0:
            self._daily_loss_usdt += abs(loss_usdt)
            logger.warning(f"Daily loss: {self._daily_loss_usdt:.2f} USDT")

    def record_profit(self, profit_usdt: float):
        if profit_usdt > 0:
            self._daily_loss_usdt = max(0, self._daily_loss_usdt - profit_usdt)

    def is_daily_limit_hit(self, balance: AccountBalance) -> bool:
        if self._session_start_balance <= 0:
            return False
        daily_loss_pct = (self._daily_loss_usdt / self._session_start_balance) * 100
        limit          = self.settings.max_daily_loss_pct
        if daily_loss_pct >= limit:
            logger.error(
                f"Daily loss limit hit: {daily_loss_pct:.1f}% >= {limit}% — "
                f"stopping all trades"
            )
            return True
        return False

    def calculate_trade(
        self,
        symbol:       str,
        side:         OrderSide,
        entry_price:  float,
        balance:      AccountBalance,
        open_trades:  int              = 0,
        # ATR for dynamic TP/SL (FIX #4)
        atr:          Optional[float]  = None,
        # Regime multipliers (from RegimeParams)
        sl_atr_mult:  Optional[float]  = None,
        tp_atr_mult:  Optional[float]  = None,
        size_scale:   float            = 1.0,
        # Legacy fixed-pct overrides (fallback only)
        leverage:     Optional[int]    = None,
        tp_pct:       Optional[float]  = None,
        sl_pct:       Optional[float]  = None,
        risk_pct:     Optional[float]  = None,
    ) -> TradeParameters:
        """
        Calculate all parameters for a trade.
        Returns TradeParameters with approved=False if any check fails.
        """
        lev   = leverage or self.settings.leverage
        r_pct = risk_pct or self.settings.risk_per_trade_pct

        # ── Pre-trade checks ───────────────────────────────────────────────
        if open_trades >= self.settings.max_open_trades:
            return self._reject(
                symbol, side, entry_price,
                f"Max open trades ({self.settings.max_open_trades}) reached",
            )

        if balance.available_balance <= 0:
            return self._reject(symbol, side, entry_price, "No available balance")

        if self.is_daily_limit_hit(balance):
            return self._reject(symbol, side, entry_price, "Daily loss limit hit")

        # FIX #5 — directional concentration check
        ok, reason = self.portfolio.can_open(side)
        if not ok:
            return self._reject(symbol, side, entry_price, reason)

        # ── Position sizing ────────────────────────────────────────────────
        risk_amount   = balance.available_balance * (r_pct / 100) * size_scale
        position_usdt = risk_amount * lev
        quantity      = self._round_quantity(position_usdt / entry_price)

        if quantity <= 0:
            return self._reject(symbol, side, entry_price, "Calculated quantity is 0")

        # ── TP/SL: ATR-based preferred, fixed-pct fallback ────────────────
        sl_m = sl_atr_mult or self.SL_ATR_MULT
        tp_m = tp_atr_mult or self.TP_ATR_MULT

        if atr and atr > 0:
            sl_dist = atr * sl_m
            tp_dist = atr * tp_m
            if side == OrderSide.LONG:
                take_profit = round(entry_price + tp_dist, 4)
                stop_loss   = round(entry_price - sl_dist, 4)
            else:
                take_profit = round(entry_price - tp_dist, 4)
                stop_loss   = round(entry_price + sl_dist, 4)
        else:
            # Legacy percentage fallback
            tp = tp_pct or self.settings.take_profit_pct
            sl = sl_pct or self.settings.stop_loss_pct
            if side == OrderSide.LONG:
                take_profit = round(entry_price * (1 + tp / 100), 4)
                stop_loss   = round(entry_price * (1 - sl / 100), 4)
            else:
                take_profit = round(entry_price * (1 - tp / 100), 4)
                stop_loss   = round(entry_price * (1 + sl / 100), 4)

        logger.info(
            f"Trade: {symbol} {side.value.upper()} | qty={quantity} | lev={lev}x | "
            f"TP={take_profit} SL={stop_loss} | risk={risk_amount:.2f} USDT"
        )

        return TradeParameters(
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=lev,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            position_size_usdt=position_usdt,
            risk_amount_usdt=risk_amount,
            approved=True,
        )

    def _round_quantity(self, qty: float, precision: int = 3) -> float:
        factor = 10 ** precision
        return math.floor(qty * factor) / factor

    def _reject(self, symbol: str, side: OrderSide, price: float, reason: str) -> TradeParameters:
        logger.warning(f"Trade rejected: {symbol} {side.value} — {reason}")
        return TradeParameters(
            symbol=symbol, side=side, quantity=0,
            leverage=1, entry_price=price,
            take_profit=price, stop_loss=price,
            position_size_usdt=0, risk_amount_usdt=0,
            approved=False, reject_reason=reason,
        )
