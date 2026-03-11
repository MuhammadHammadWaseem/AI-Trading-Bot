"""
core/risk/risk_manager.py
──────────────────────────
Risk management — position sizing, TP/SL calculation, daily loss limits.
Every trade MUST pass through the risk manager before execution.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

from core.exchange.base_exchange import AccountBalance, OrderSide
from config.logger import get_logger
from config.settings import RiskSettings

logger = get_logger(__name__)


@dataclass
class TradeParameters:
    """Fully computed trade parameters ready for execution."""
    symbol:       str
    side:         OrderSide
    quantity:     float        # Contract quantity
    leverage:     int
    entry_price:  float
    take_profit:  float        # Absolute price
    stop_loss:    float        # Absolute price
    position_size_usdt: float  # Dollar exposure
    risk_amount_usdt:   float  # Max loss in USDT
    approved:     bool = True
    reject_reason: str = ""


class RiskManager:
    """
    Calculates safe position sizes and validates trade parameters.
    
    Key formulas:
        position_size = balance * risk_per_trade_pct / 100
        quantity      = (position_size * leverage) / entry_price
        take_profit   = entry ± (entry * tp_pct / 100)
        stop_loss     = entry ∓ (entry * sl_pct / 100)
    """

    def __init__(self, risk_settings: RiskSettings):
        self.settings = risk_settings
        self._daily_loss_usdt = 0.0
        self._session_start_balance = 0.0

    def set_session_balance(self, balance: float):
        """Call at bot startup to track daily loss."""
        self._session_start_balance = balance

    def record_loss(self, loss_usdt: float):
        """Record a realized loss for daily tracking."""
        if loss_usdt < 0:
            self._daily_loss_usdt += abs(loss_usdt)
            logger.warning(f"📉 Daily loss so far: {self._daily_loss_usdt:.2f} USDT")

    def record_profit(self, profit_usdt: float):
        """Record a realized profit."""
        if profit_usdt > 0:
            self._daily_loss_usdt = max(0, self._daily_loss_usdt - profit_usdt)

    def is_daily_limit_hit(self, balance: AccountBalance) -> bool:
        """Check if daily loss limit has been reached."""
        if self._session_start_balance <= 0:
            return False

        daily_loss_pct = (self._daily_loss_usdt / self._session_start_balance) * 100
        limit = self.settings.max_daily_loss_pct

        if daily_loss_pct >= limit:
            logger.error(
                f"🚫 Daily loss limit hit: {daily_loss_pct:.1f}% >= {limit}% — "
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
        open_trades:  int = 0,
        # Optional overrides (for user-customized settings)
        leverage:     Optional[int]   = None,
        tp_pct:       Optional[float] = None,
        sl_pct:       Optional[float] = None,
        risk_pct:     Optional[float] = None,
    ) -> TradeParameters:
        """
        Calculate all parameters for a trade.
        Returns TradeParameters with approved=False if any check fails.
        """
        lev    = leverage or self.settings.leverage
        tp     = tp_pct   or self.settings.take_profit_pct
        sl     = sl_pct   or self.settings.stop_loss_pct
        r_pct  = risk_pct or self.settings.risk_per_trade_pct

        # ── Pre-trade checks ───────────────────────────────────────────────
        if open_trades >= self.settings.max_open_trades:
            return self._reject(
                symbol, side, entry_price,
                f"Max open trades ({self.settings.max_open_trades}) reached"
            )

        if balance.available_balance <= 0:
            return self._reject(symbol, side, entry_price, "No available balance")

        if self.is_daily_limit_hit(balance):
            return self._reject(symbol, side, entry_price, "Daily loss limit hit")

        # ── Position sizing ────────────────────────────────────────────────
        risk_amount   = balance.available_balance * (r_pct / 100)
        position_usdt = risk_amount * lev
        quantity      = self._round_quantity(position_usdt / entry_price)

        if quantity <= 0:
            return self._reject(symbol, side, entry_price, "Calculated quantity is 0")

        # ── TP/SL prices ───────────────────────────────────────────────────
        if side == OrderSide.LONG:
            take_profit = round(entry_price * (1 + tp / 100), 4)
            stop_loss   = round(entry_price * (1 - sl / 100), 4)
        else:  # SHORT
            take_profit = round(entry_price * (1 - tp / 100), 4)
            stop_loss   = round(entry_price * (1 + sl / 100), 4)

        logger.info(
            f"📊 Trade calculated: {symbol} {side.value.upper()} | "
            f"qty={quantity} | lev={lev}x | "
            f"TP={take_profit} (+{tp}%) | SL={stop_loss} (-{sl}%) | "
            f"risk={risk_amount:.2f} USDT"
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
        """Round quantity to exchange precision."""
        factor = 10 ** precision
        return math.floor(qty * factor) / factor

    def _reject(
        self, symbol: str, side: OrderSide, price: float, reason: str
    ) -> TradeParameters:
        logger.warning(f"❌ Trade rejected: {symbol} {side.value} — {reason}")
        return TradeParameters(
            symbol=symbol, side=side, quantity=0,
            leverage=1, entry_price=price,
            take_profit=price, stop_loss=price,
            position_size_usdt=0, risk_amount_usdt=0,
            approved=False, reject_reason=reason,
        )
