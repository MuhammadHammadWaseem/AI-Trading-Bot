"""
core/exchange/paper_exchange.py
────────────────────────────────
Paper trading engine. Implements the EXACT same BaseExchange interface as
BinanceExchange so FuturesTrader never needs to know which mode it is in.

Data flow:
  ┌─────────────────────────────────────────────────────────┐
  │  PaperExchange                                          │
  │   ├── Market data  → delegates to real BinanceExchange  │
  │   └── Orders       → fills virtually at current price   │
  └─────────────────────────────────────────────────────────┘

The real BinanceExchange instance is used ONLY for:
  get_ohlcv(), get_current_price(), connect()

All order methods (open_long, open_short, close_position, etc.) are
simulated internally with a virtual USDT balance.
"""

import asyncio
import math
import time
import uuid
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from core.exchange.base_exchange import (
    BaseExchange, OrderSide, OrderStatus,
    PositionInfo, OrderResult, AccountBalance,
)
from core.exchange.binance_exchange import BinanceExchange
from config.settings import ExchangeCredentials

logger = logging.getLogger(__name__)


@dataclass
class _VirtualPosition:
    symbol:      str
    side:        OrderSide
    entry_price: float
    quantity:    float
    leverage:    int
    take_profit: Optional[float]
    stop_loss:   Optional[float]
    order_id:    str
    opened_at:   float = field(default_factory=time.monotonic)


class PaperExchange(BaseExchange):
    """
    Paper trading exchange.

    Drop-in replacement for BinanceExchange:
      exchange = PaperExchange(credentials, initial_balance=10_000)
      await exchange.connect()   # connects to real Binance for data only
      ...
    All trade methods simulate fills. Balance tracked in memory.
    """

    # Realistic Binance USDM taker fee (0.04%)
    TAKER_FEE_RATE = 0.0004

    def __init__(
        self,
        credentials: ExchangeCredentials,
        initial_balance: float = 10_000.0,
    ) -> None:
        # Underlying exchange client: public market data only.
        # Paper mode uses virtual orders and should not require real API keys.
        self._real = BinanceExchange(credentials, public_only=True)

        # Virtual portfolio state
        self._initial_balance:  float = initial_balance
        self._cash:             float = initial_balance   # free USDT
        self._positions:        Dict[str, _VirtualPosition] = {}
        self._realized_pnl:     float = 0.0
        self._total_fees_paid:  float = 0.0
        self._closed_trades:    List[dict] = []
        self._order_seq:        int = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def connect(self) -> bool:
        ok = await self._real.connect()
        if ok:
            logger.info(
                f"[PAPER] Exchange connected — virtual balance: "
                f"${self._initial_balance:,.2f} USDT"
            )
        return ok

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        try:
            await self._real._exchange.close()
        except Exception:
            pass

    def get_exchange_name(self) -> str:
        return "binance_paper"

    # ── Market data (pass-through to real exchange) ────────────────────────

    async def get_current_price(self, symbol: str) -> float:
        return await self._real.get_current_price(symbol)

    async def get_ohlcv(
        self, symbol: str, timeframe: str = "5m", limit: int = 200
    ) -> List[dict]:
        return await self._real.get_ohlcv(symbol, timeframe, limit)

    # ── Balance ───────────────────────────────────────────────────────────

    async def get_balance(self) -> AccountBalance:
        unrealized = 0.0
        used_margin = 0.0

        for sym, pos in self._positions.items():
            try:
                price = await self._real.get_current_price(sym)
            except Exception:
                price = pos.entry_price

            if pos.side == OrderSide.LONG:
                unrealized += (price - pos.entry_price) * pos.quantity
            else:
                unrealized += (pos.entry_price - price) * pos.quantity

            used_margin += (pos.entry_price * pos.quantity) / pos.leverage

        return AccountBalance(
            total_balance     = self._cash + used_margin + unrealized,
            available_balance = self._cash,
            used_margin       = used_margin,
            unrealized_pnl    = unrealized,
        )

    # ── Leverage (no-op) ──────────────────────────────────────────────────

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        return True  # no-op in paper mode

    # ── Order methods ─────────────────────────────────────────────────────

    async def open_long(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 1,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> OrderResult:
        return await self._fill_open(
            symbol, OrderSide.LONG, quantity, leverage, take_profit, stop_loss
        )

    async def open_short(
        self,
        symbol: str,
        quantity: float,
        leverage: int = 1,
        take_profit: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> OrderResult:
        return await self._fill_open(
            symbol, OrderSide.SHORT, quantity, leverage, take_profit, stop_loss
        )

    async def _fill_open(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        leverage: int,
        take_profit: Optional[float],
        stop_loss: Optional[float],
    ) -> OrderResult:
        price    = await self._real.get_current_price(symbol)
        notional = price * quantity
        margin   = notional / leverage
        fee      = notional * self.TAKER_FEE_RATE

        # Reject if insufficient free cash
        if self._cash < margin + fee:
            msg = (
                f"Insufficient paper balance: need ${margin + fee:.2f} "
                f"(margin=${margin:.2f} + fee=${fee:.4f}), "
                f"available=${self._cash:.2f}"
            )
            logger.warning(f"[PAPER] {msg}")
            return OrderResult(
                success=False, order_id="", symbol=symbol, side=side,
                quantity=quantity, price=price,
                status=OrderStatus.CANCELLED, message=msg,
            )

        # Deduct margin + fee from free cash
        self._cash           -= margin + fee
        self._total_fees_paid += fee

        self._order_seq += 1
        order_id = f"PAPER-{self._order_seq:06d}"

        self._positions[symbol] = _VirtualPosition(
            symbol=symbol, side=side, entry_price=price,
            quantity=quantity, leverage=leverage,
            take_profit=take_profit, stop_loss=stop_loss,
            order_id=order_id,
        )

        logger.info(
            f"[PAPER] OPEN {side.value.upper()} {symbol} "
            f"qty={quantity} @ {price:.4f} | "
            f"margin=${margin:.2f} fee=${fee:.4f} | "
            f"cash_remaining=${self._cash:.2f}"
        )

        return OrderResult(
            success=True, order_id=order_id, symbol=symbol,
            side=side, quantity=quantity, price=price,
            status=OrderStatus.OPEN,
        )

    async def close_position(
        self,
        symbol: str,
        order_id: Optional[str] = None,
    ) -> OrderResult:
        pos = self._positions.get(symbol)
        if not pos:
            return OrderResult(
                success=False, order_id="", symbol=symbol,
                side=OrderSide.LONG, quantity=0, price=0,
                status=OrderStatus.CLOSED, message="No open position",
            )
        return await self._fill_close(symbol, pos, pos.quantity)

    async def close_position_partial(
        self, symbol: str, quantity: float
    ) -> OrderResult:
        pos = self._positions.get(symbol)
        if not pos:
            return OrderResult(
                success=False, order_id="", symbol=symbol,
                side=OrderSide.LONG, quantity=0, price=0,
                status=OrderStatus.CANCELLED, message="No open position",
            )
        qty = min(quantity, pos.quantity)
        return await self._fill_close(symbol, pos, qty, partial=True)

    async def _fill_close(
        self,
        symbol: str,
        pos: _VirtualPosition,
        quantity: float,
        partial: bool = False,
    ) -> OrderResult:
        price    = await self._real.get_current_price(symbol)
        notional = price * quantity
        fee      = notional * self.TAKER_FEE_RATE
        margin   = (pos.entry_price * quantity) / pos.leverage

        if pos.side == OrderSide.LONG:
            raw_pnl = (price - pos.entry_price) * quantity
        else:
            raw_pnl = (pos.entry_price - price) * quantity

        net_pnl = raw_pnl - fee

        # Return margin + raw_pnl minus fee
        self._cash            += margin + net_pnl
        self._total_fees_paid += fee
        self._realized_pnl    += net_pnl

        self._closed_trades.append({
            "symbol":      symbol,
            "side":        pos.side.value,
            "entry":       pos.entry_price,
            "exit":        price,
            "quantity":    quantity,
            "raw_pnl":     round(raw_pnl, 6),
            "fee":         round(fee, 6),
            "net_pnl":     round(net_pnl, 6),
            "closed_at":   time.time(),
            "partial":     partial,
        })

        if partial:
            # Reduce quantity; position stays open
            pos.quantity = round(pos.quantity - quantity, 6)
            if pos.quantity <= 0:
                del self._positions[symbol]
        else:
            del self._positions[symbol]

        logger.info(
            f"[PAPER] {'PARTIAL ' if partial else ''}CLOSE {symbol} "
            f"qty={quantity} @ {price:.4f} | "
            f"pnl=${raw_pnl:+.4f} fee=${fee:.4f} net=${net_pnl:+.4f} | "
            f"cash=${self._cash:.2f}"
        )

        return OrderResult(
            success=True, order_id=pos.order_id, symbol=symbol,
            side=pos.side, quantity=quantity, price=price,
            status=OrderStatus.CLOSED,
        )

    async def get_open_positions(self) -> List[PositionInfo]:
        result = []
        for sym, pos in list(self._positions.items()):
            try:
                price = await self._real.get_current_price(sym)
            except Exception:
                price = pos.entry_price

            if pos.side == OrderSide.LONG:
                upnl = (price - pos.entry_price) * pos.quantity
            else:
                upnl = (pos.entry_price - price) * pos.quantity

            result.append(PositionInfo(
                symbol        = sym,
                side          = pos.side,
                entry_price   = pos.entry_price,
                current_price = price,
                quantity      = pos.quantity,
                leverage      = pos.leverage,
                unrealized_pnl = upnl,
                take_profit   = pos.take_profit,
                stop_loss     = pos.stop_loss,
                order_id      = pos.order_id,
            ))
        return result

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        return True  # no real orders to cancel

    # ── Paper-specific stats ───────────────────────────────────────────────

    @property
    def paper_stats(self) -> dict:
        """Summary of virtual portfolio performance."""
        return {
            "mode":            "paper",
            "initial_balance": round(self._initial_balance, 2),
            "current_cash":    round(self._cash, 2),
            "realized_pnl":    round(self._realized_pnl, 4),
            "total_fees":      round(self._total_fees_paid, 6),
            "open_positions":  len(self._positions),
            "closed_trades":   len(self._closed_trades),
            "return_pct":      round(
                (self._realized_pnl / self._initial_balance) * 100, 2
            ) if self._initial_balance else 0,
        }
