"""
core/exchange/binance_exchange.py
----------------------------------
Binance USDM Futures - Testnet & Live support via ccxt.
"""

import ccxt.async_support as ccxt
from typing import Dict, List, Optional

from core.exchange.base_exchange import (
    BaseExchange, OrderSide, OrderStatus,
    PositionInfo, OrderResult, AccountBalance
)
from config.logger import get_logger
from config.settings import ExchangeCredentials

logger = get_logger(__name__)


class BinanceExchange(BaseExchange):

    TESTNET_BASE = "https://testnet.binancefuture.com"

    def __init__(self, credentials: ExchangeCredentials):
        self.credentials = credentials
        self._exchange: Optional[ccxt.binanceusdm] = None

    async def connect(self) -> bool:
        try:
            config = {
                "apiKey": self.credentials.api_key,
                "secret": self.credentials.secret,
                "options": {
                    "defaultType": "future",
                    "fetchCurrencies": False,
                    "adjustForTimeDifference": True,
                },
                "enableRateLimit": True,
            }

            self._exchange = ccxt.binanceusdm(config)

            # Override only the fapi URLs (futures endpoints) with testnet base
            if self.credentials.testnet:
                b = self.TESTNET_BASE
                self._exchange.urls["api"]["fapiPublic"]    = f"{b}/fapi/v1"
                self._exchange.urls["api"]["fapiPublicV2"]  = f"{b}/fapi/v2"
                self._exchange.urls["api"]["fapiPrivate"]   = f"{b}/fapi/v1"
                self._exchange.urls["api"]["fapiPrivateV2"] = f"{b}/fapi/v2"
                self._exchange.urls["api"]["fapiData"]      = f"{b}/futures/data"

            # Verify connection
            await self._exchange.fetch_balance({"type": "future"})
            mode = "TESTNET" if self.credentials.testnet else "LIVE"
            logger.info(f"[OK] Binance {mode} connected")
            return True

        except ccxt.AuthenticationError as e:
            logger.error(f"[FAIL] Auth error - check API keys: {e}")
            return False
        except Exception as e:
            logger.error(f"[FAIL] Connection error: {e}")
            return False

    async def get_balance(self) -> AccountBalance:
        try:
            balance = await self._exchange.fetch_balance({"type": "future"})
            usdt = balance.get("USDT", {})
            info = balance.get("info", {})
            unrealized = float(info.get("totalUnrealizedProfit", 0)) if isinstance(info, dict) else 0.0
            return AccountBalance(
                total_balance=float(usdt.get("total", 0)),
                available_balance=float(usdt.get("free", 0)),
                used_margin=float(usdt.get("used", 0)),
                unrealized_pnl=unrealized,
            )
        except Exception as e:
            logger.error(f"get_balance error: {e}")
            raise

    async def get_current_price(self, symbol: str) -> float:
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
            return float(ticker["last"])
        except Exception as e:
            logger.error(f"get_current_price({symbol}) error: {e}")
            raise

    async def get_ohlcv(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> List[Dict]:
        try:
            raw = await self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return [
                {"timestamp": c[0], "open": float(c[1]), "high": float(c[2]),
                 "low": float(c[3]), "close": float(c[4]), "volume": float(c[5])}
                for c in raw
            ]
        except Exception as e:
            logger.error(f"get_ohlcv({symbol}) error: {e}")
            raise

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        try:
            await self._exchange.set_leverage(leverage, symbol)
            return True
        except Exception as e:
            logger.warning(f"set_leverage({symbol}, {leverage}): {e}")
            return False

    async def open_long(self, symbol, quantity, leverage,
                        take_profit=None, stop_loss=None) -> OrderResult:
        return await self._open_position(
            symbol, OrderSide.LONG, quantity, leverage, take_profit, stop_loss)

    async def open_short(self, symbol, quantity, leverage,
                         take_profit=None, stop_loss=None) -> OrderResult:
        return await self._open_position(
            symbol, OrderSide.SHORT, quantity, leverage, take_profit, stop_loss)

    async def _open_position(self, symbol, side, quantity, leverage,
                              take_profit, stop_loss) -> OrderResult:
        try:
            await self.set_leverage(symbol, leverage)
            ccxt_side = "buy" if side == OrderSide.LONG else "sell"
            order = await self._exchange.create_market_order(
                symbol, ccxt_side, quantity,
                params={"positionSide": "BOTH"}
            )
            price    = float(order.get("price") or order.get("average") or 0)
            order_id = str(order["id"])
            logger.info(
                f"[TRADE] {side.value.upper()} {symbol} "
                f"qty={quantity} price={price} lev={leverage}x"
            )
            if take_profit:
                await self._place_take_profit(symbol, side, quantity, take_profit)
            if stop_loss:
                await self._place_stop_loss(symbol, side, quantity, stop_loss)
            return OrderResult(
                success=True, order_id=order_id, symbol=symbol,
                side=side, quantity=quantity, price=price,
                status=OrderStatus.OPEN
            )
        except ccxt.InsufficientFunds as e:
            return OrderResult(
                success=False, order_id="", symbol=symbol, side=side,
                quantity=quantity, price=0, status=OrderStatus.CANCELLED,
                message=f"Insufficient funds: {e}"
            )
        except Exception as e:
            logger.error(f"_open_position error {symbol}: {e}")
            return OrderResult(
                success=False, order_id="", symbol=symbol, side=side,
                quantity=quantity, price=0, status=OrderStatus.CANCELLED,
                message=str(e)
            )

    async def _place_take_profit(self, symbol, side, quantity, tp_price):
        try:
            close_side = "sell" if side == OrderSide.LONG else "buy"
            await self._exchange.create_order(
                symbol, "TAKE_PROFIT_MARKET", close_side, quantity,
                params={"stopPrice": tp_price, "closePosition": True,
                        "positionSide": "BOTH"}
            )
        except Exception as e:
            logger.warning(f"TP placement {symbol}: {e}")

    async def _place_stop_loss(self, symbol, side, quantity, sl_price):
        try:
            close_side = "sell" if side == OrderSide.LONG else "buy"
            await self._exchange.create_order(
                symbol, "STOP_MARKET", close_side, quantity,
                params={"stopPrice": sl_price, "closePosition": True,
                        "positionSide": "BOTH"}
            )
        except Exception as e:
            logger.warning(f"SL placement {symbol}: {e}")

    async def close_position(self, symbol, order_id=None) -> OrderResult:
        try:
            positions = await self.get_open_positions()
            pos = next((p for p in positions if p.symbol == symbol), None)
            if not pos:
                return OrderResult(
                    success=False, order_id="", symbol=symbol,
                    side=OrderSide.LONG, quantity=0, price=0,
                    status=OrderStatus.CLOSED, message="No open position"
                )
            close_side = "sell" if pos.side == OrderSide.LONG else "buy"
            order = await self._exchange.create_market_order(
                symbol, close_side, pos.quantity,
                params={"reduceOnly": True, "positionSide": "BOTH"}
            )
            price = float(order.get("price") or order.get("average") or 0)
            logger.info(f"[CLOSE] {symbol} PnL={pos.unrealized_pnl:.4f} USDT")
            return OrderResult(
                success=True, order_id=str(order["id"]), symbol=symbol,
                side=pos.side, quantity=pos.quantity, price=price,
                status=OrderStatus.CLOSED
            )
        except Exception as e:
            logger.error(f"close_position {symbol}: {e}")
            return OrderResult(
                success=False, order_id="", symbol=symbol,
                side=OrderSide.LONG, quantity=0, price=0,
                status=OrderStatus.CANCELLED, message=str(e)
            )

    async def get_open_positions(self) -> List[PositionInfo]:
        try:
            positions = await self._exchange.fetch_positions()
            result = []
            for p in positions:
                if float(p.get("contracts", 0)) == 0:
                    continue
                side = OrderSide.LONG if p["side"] == "long" else OrderSide.SHORT
                result.append(PositionInfo(
                    symbol=p["symbol"], side=side,
                    entry_price=float(p.get("entryPrice", 0)),
                    current_price=float(p.get("markPrice", 0)),
                    quantity=float(p.get("contracts", 0)),
                    leverage=int(p.get("leverage", 1)),
                    unrealized_pnl=float(p.get("unrealizedPnl", 0)),
                ))
            return result
        except Exception as e:
            logger.error(f"get_open_positions error: {e}")
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            await self._exchange.cancel_order(order_id, symbol)
            return True
        except Exception as e:
            logger.warning(f"cancel_order {order_id}: {e}")
            return False

    def get_exchange_name(self) -> str:
        return f"binance_{'testnet' if self.credentials.testnet else 'live'}"

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *args):
        if self._exchange:
            await self._exchange.close()