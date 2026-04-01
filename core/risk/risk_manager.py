"""
core/risk/risk_manager.py
──────────────────────────
Risk management — position sizing, ATR-based TP/SL, daily loss limits,
directional concentration filter, and directional loss-streak guard.

Directional limit:
    MAX_SAME_DIRECTION = 1  — at most 1 LONG and 1 SHORT open at once.

Loss-streak guard:
    3 consecutive losses in a direction → 8-bar cooldown on that direction.

Risk per trade: 0.5% (reduced from 1.0%)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
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
    approved:           bool = True
    reject_reason:      str  = ""


# ── Directional portfolio tracker ─────────────────────────────────────────────

class PortfolioDirectionTracker:
    """
    Enforces directional diversification across all open positions.
    MAX_SAME_DIRECTION = 1: never 2 SHORTs or 2 LONGs simultaneously.
    Tracks per-direction loss streaks and blocks after 3 consecutive losses.
    """

    MAX_SAME_DIRECTION   = 1
    STREAK_BLOCK_COUNT   = 3
    STREAK_COOLDOWN_BARS = 8

    def __init__(self):
        self._positions: Dict[str, OrderSide] = {}
        self._outcomes: Dict[str, Deque[bool]] = {
            "long":  deque(maxlen=self.STREAK_BLOCK_COUNT),
            "short": deque(maxlen=self.STREAK_BLOCK_COUNT),
        }
        self._direction_cooldown: Dict[str, int] = {"long": 0, "short": 0}

    def tick(self):
        """Call once per cycle to decrement directional cooldowns."""
        for d in self._direction_cooldown:
            if self._direction_cooldown[d] > 0:
                self._direction_cooldown[d] -= 1
                if self._direction_cooldown[d] == 0:
                    logger.info(f"[DIRECTION] {d.upper()} cooldown lifted — entries allowed again")

    def register_open(self, symbol: str, side: OrderSide):
        self._positions[symbol] = side
        logger.info(
            f"[PORTFOLIO] {symbol} registered {side.value} | "
            f"LONG={self._count(OrderSide.LONG)}  SHORT={self._count(OrderSide.SHORT)}"
        )

    def register_close(self, symbol: str, profit: bool):
        side = self._positions.pop(symbol, None)
        if side is None:
            return

        direction = side.value
        self._outcomes[direction].append(profit)

        outcomes = list(self._outcomes[direction])
        if (
            len(outcomes) >= self.STREAK_BLOCK_COUNT
            and not any(outcomes[-self.STREAK_BLOCK_COUNT:])
        ):
            logger.warning(
                f"[DIRECTION BLOCK] {direction.upper()} — "
                f"{self.STREAK_BLOCK_COUNT} consecutive losses. "
                f"Blocking {direction} entries for {self.STREAK_COOLDOWN_BARS} bars."
            )
            self._direction_cooldown[direction] = self.STREAK_COOLDOWN_BARS

        logger.info(
            f"[PORTFOLIO] {symbol} deregistered ({'WIN' if profit else 'LOSS'}) | "
            f"LONG={self._count(OrderSide.LONG)}  SHORT={self._count(OrderSide.SHORT)}"
        )

    def can_open(self, side: OrderSide) -> Tuple[bool, str]:
        direction = side.value

        cooldown = self._direction_cooldown[direction]
        if cooldown > 0:
            return False, (
                f"{direction.upper()} entries blocked after loss streak "
                f"({cooldown} bars remaining)"
            )

        current = self._count(side)
        if current >= self.MAX_SAME_DIRECTION:
            return False, (
                f"Already {current} {direction.upper()} position(s) open "
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

    TP/SL: ATR-based when atr is provided (preferred), fixed-pct fallback.
        SL = entry ± (ATR × sl_atr_mult)    default 2.0×ATR
        TP = entry ± (ATR × tp_atr_mult)    default 3.0×ATR  → 1.5 R:R

    Risk per trade: 0.5% of available balance (changed from 1.0%)
    """

    SL_ATR_MULT = 2.0
    TP_ATR_MULT = 3.0

    def __init__(self, risk_settings: RiskSettings):
        self.settings               = risk_settings
        self._daily_loss_usdt       = 0.0
        self._session_start_balance = 0.0
        self.portfolio              = PortfolioDirectionTracker()

    def set_session_balance(self, balance: float):
        self._session_start_balance = balance

    def tick(self):
        """Call once per main loop cycle to advance cooldown timers."""
        self.portfolio.tick()

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

        # Check percentage-based limit
        pct = (self._daily_loss_usdt / self._session_start_balance) * 100
        if pct >= self.settings.max_daily_loss_pct:
            logger.error(
                f"Daily loss limit hit: {pct:.1f}% >= {self.settings.max_daily_loss_pct}% "
                f"— stopping all trades"
            )
            return True

        # Check absolute USDT limit (from Laravel config, 0 = disabled)
        if self.settings.max_daily_loss_usdt > 0:
            if self._daily_loss_usdt >= self.settings.max_daily_loss_usdt:
                logger.error(
                    f"Daily USDT loss limit hit: {self._daily_loss_usdt:.2f} >= "
                    f"{self.settings.max_daily_loss_usdt:.2f} USDT — stopping all trades"
                )
                return True

        return False

    def calculate_trade(
        self,
        symbol:      str,
        side:        OrderSide,
        entry_price: float,
        balance:     AccountBalance,
        open_trades: int            = 0,
        atr:         Optional[float] = None,
        sl_atr_mult: Optional[float] = None,
        tp_atr_mult: Optional[float] = None,
        size_scale:  float           = 1.0,
        leverage:    Optional[int]   = None,
        tp_pct:      Optional[float] = None,
        sl_pct:      Optional[float] = None,
        risk_pct:    Optional[float] = None,
    ) -> TradeParameters:

        lev   = leverage or self.settings.leverage
        r_pct = risk_pct or self.settings.risk_per_trade_pct  # default now 0.5%

        # ── Pre-trade checks ──────────────────────────────────────────────
        if open_trades >= self.settings.max_open_trades:
            return self._reject(symbol, side, entry_price,
                                f"Max open trades ({self.settings.max_open_trades}) reached")

        if balance.available_balance <= 0:
            return self._reject(symbol, side, entry_price, "No available balance")

        if self.is_daily_limit_hit(balance):
            return self._reject(symbol, side, entry_price, "Daily loss limit hit")

        ok, reason = self.portfolio.can_open(side)
        if not ok:
            return self._reject(symbol, side, entry_price, reason)

        # ── Position sizing ───────────────────────────────────────────────
        risk_amount   = balance.available_balance * (r_pct / 100) * size_scale
        position_usdt = risk_amount * lev
        quantity      = self._round_quantity(position_usdt / entry_price)

        if quantity <= 0:
            return self._reject(symbol, side, entry_price, "Calculated quantity is 0")

        # ── Minimum notional enforcement (Binance USDM requires >= $100) ─────
        # After floor-rounding, the actual notional (qty × price) can drop below
        # the exchange minimum even when position_usdt is above it.
        # e.g. BTC @ $68k: position_usdt=$124 → qty_raw=0.00181 → floored=0.001
        #      actual_notional = 0.001 × $68k = $68 → exchange rejects with -4164
        # Fix: bump qty up to the next step if notional is below the minimum.
        # Only proceed if the bump is ≤ 2× the intended position (avoid silent risk creep).
        MIN_BINANCE_NOTIONAL = 100.0
        actual_notional = quantity * entry_price
        if actual_notional < MIN_BINANCE_NOTIONAL:
            # Step size is 3 decimal places for most symbols
            step = 10 ** -3
            min_qty = math.ceil(MIN_BINANCE_NOTIONAL / entry_price / step) * step
            min_qty = round(min_qty, 3)
            bumped_notional = min_qty * entry_price
            oversize_ratio  = bumped_notional / position_usdt if position_usdt > 0 else 999
            if oversize_ratio > 2.0:
                return self._reject(
                    symbol, side, entry_price,
                    f"Notional too small: ${actual_notional:.2f} < ${MIN_BINANCE_NOTIONAL:.0f} minimum. "
                    f"Bumping to min would require ${bumped_notional:.2f} ({oversize_ratio:.1f}× intended). "
                    f"Increase balance or risk% to trade this symbol."
                )
            logger.info(
                f"[MIN_NOTIONAL] {symbol} — qty bumped {quantity:.3f} → {min_qty:.3f} "
                f"(notional: ${actual_notional:.2f} → ${bumped_notional:.2f}) to meet $100 minimum"
            )
            quantity = min_qty

        # ── ATR-based TP/SL ───────────────────────────────────────────────
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
            tp = tp_pct or self.settings.take_profit_pct
            sl = sl_pct or self.settings.stop_loss_pct
            if side == OrderSide.LONG:
                take_profit = round(entry_price * (1 + tp / 100), 4)
                stop_loss   = round(entry_price * (1 - sl / 100), 4)
            else:
                take_profit = round(entry_price * (1 - tp / 100), 4)
                stop_loss   = round(entry_price * (1 + sl / 100), 4)

        # ── Guard 1: TP direction sanity ─────────────────────────────────
        # For LONG: TP must be above entry. For SHORT: TP must be below entry.
        # If ATR is extreme or sign is wrong, reject the trade.
        if side == OrderSide.LONG and take_profit <= entry_price:
            return self._reject(symbol, side, entry_price,
                                f"TP direction wrong: LONG TP={take_profit} <= entry={entry_price}")
        if side == OrderSide.SHORT and take_profit >= entry_price:
            return self._reject(symbol, side, entry_price,
                                f"TP direction wrong: SHORT TP={take_profit} >= entry={entry_price}")

        # ── Guard 2: Minimum SL distance (prevents noise-level stops) ────
        # SL must be at least 0.10% of entry price away from entry.
        # A tighter SL will be hit by normal bid/ask spread or slippage.
        MIN_SL_PCT = 0.10  # 0.10% minimum
        sl_dist    = abs(entry_price - stop_loss)
        min_sl_dist = entry_price * (MIN_SL_PCT / 100)
        if sl_dist < min_sl_dist:
            return self._reject(symbol, side, entry_price,
                                f"SL too tight: sl_dist={sl_dist:.4f} < min={min_sl_dist:.4f} "
                                f"({MIN_SL_PCT}% of entry). Risk of immediate noise stop-out.")

        # ── Guard 3: Minimum R:R ratio ───────────────────────────────────
        # TP distance must be at least 1.0× the SL distance.
        # A lower ratio guarantees losses even with >50% win rate.
        MIN_RR = 1.0
        tp_dist = abs(entry_price - take_profit)
        actual_rr = tp_dist / sl_dist if sl_dist > 0 else 0.0
        if actual_rr < MIN_RR:
            return self._reject(symbol, side, entry_price,
                                f"R:R too low: tp_dist={tp_dist:.4f} / sl_dist={sl_dist:.4f} "
                                f"= {actual_rr:.3f} < {MIN_RR}. Trade would be structurally unprofitable.")

        logger.info(
            f"Trade: {symbol} {side.value.upper()} | qty={quantity} | lev={lev}x | "
            f"TP={take_profit} SL={stop_loss} | R:R={actual_rr:.2f} | risk={risk_amount:.2f} USDT"
        )

        return TradeParameters(
            symbol=symbol, side=side, quantity=quantity,
            leverage=lev, entry_price=entry_price,
            take_profit=take_profit, stop_loss=stop_loss,
            position_size_usdt=position_usdt, risk_amount_usdt=risk_amount,
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