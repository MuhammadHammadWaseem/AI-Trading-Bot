"""
core/exchange/base_exchange.py
───────────────────────────────
Abstract base class for all exchanges.
Every exchange (Binance, OKX, Bybit, etc.) must implement this interface.
This ensures the rest of the bot never talks directly to any exchange —
it only talks to this contract.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class OrderSide(str, Enum):
    LONG  = "long"
    SHORT = "short"


class OrderStatus(str, Enum):
    OPEN      = "open"
    CLOSED    = "closed"
    CANCELLED = "cancelled"
    PARTIAL   = "partial"


@dataclass
class PositionInfo:
    """Represents an open futures position."""
    symbol:          str
    side:            OrderSide
    entry_price:     float
    current_price:   float
    quantity:        float
    leverage:        int
    unrealized_pnl:  float
    take_profit:     Optional[float] = None
    stop_loss:       Optional[float] = None
    order_id:        Optional[str]   = None


@dataclass
class OrderResult:
    """Result of placing an order."""
    success:    bool
    order_id:   str
    symbol:     str
    side:       OrderSide
    quantity:   float
    price:      float
    status:     OrderStatus
    message:    str = ""


@dataclass
class AccountBalance:
    """Futures wallet balance info."""
    total_balance:     float   # Total USDT
    available_balance: float   # Available for trading
    used_margin:       float   # Currently in positions
    unrealized_pnl:    float   # Floating P&L
    currency:          str = "USDT"


class BaseExchange(ABC):
    """
    Abstract exchange interface.
    All exchange-specific logic stays INSIDE the implementation classes.
    The bot only uses these methods.
    """

    @abstractmethod
    async def connect(self) -> bool:
        """Initialize connection and validate credentials."""
        ...

    @abstractmethod
    async def get_balance(self) -> AccountBalance:
        """Get current futures wallet balance."""
        ...

    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get latest market price for a symbol."""
        ...

    @abstractmethod
    async def get_ohlcv(
        self,
        symbol:    str,
        timeframe: str = "15m",
        limit:     int = 200,
    ) -> List[Dict]:
        """
        Fetch OHLCV (candlestick) data.
        Returns list of dicts: [timestamp, open, high, low, close, volume]
        """
        ...

    @abstractmethod
    async def open_long(
        self,
        symbol:       str,
        quantity:     float,
        leverage:     int,
        take_profit:  Optional[float] = None,
        stop_loss:    Optional[float] = None,
    ) -> OrderResult:
        """Open a LONG (buy) futures position."""
        ...

    @abstractmethod
    async def open_short(
        self,
        symbol:       str,
        quantity:     float,
        leverage:     int,
        take_profit:  Optional[float] = None,
        stop_loss:    Optional[float] = None,
    ) -> OrderResult:
        """Open a SHORT (sell) futures position."""
        ...

    @abstractmethod
    async def close_position(
        self,
        symbol:   str,
        order_id: Optional[str] = None,
    ) -> OrderResult:
        """Close an existing position."""
        ...

    @abstractmethod
    async def get_open_positions(self) -> List[PositionInfo]:
        """Get all currently open positions."""
        ...

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        ...

    @abstractmethod
    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a pending order."""
        ...

    @abstractmethod
    def get_exchange_name(self) -> str:
        """Return exchange name string."""
        ...
