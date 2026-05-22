"""
core/exchange/exchange_factory.py

Factory for exchange clients.

Modes:
  "paper" -> PaperExchange: public market data, virtual orders, no API keys required
  "live"  -> BinanceExchange: real market data, real orders, real Binance keys required
"""

import asyncio
import logging
from typing import Literal

from config.settings import settings, ExchangeCredentials
from core.exchange.base_exchange import BaseExchange
from core.exchange.binance_exchange import BinanceExchange
from core.exchange.paper_exchange import PaperExchange

logger = logging.getLogger(__name__)

TradingMode = Literal["paper", "live"]

_TESTNET_REJECTION_MSG = (
    "Please connect real Binance API keys. "
    "Testnet keys are not supported - create API keys at binance.com > "
    "Profile > API Management, then enable Futures trading."
)


def _looks_like_testnet_key(api_key: str) -> bool:
    """
    Real Binance API keys are normally long. Empty/short keys are acceptable
    only for paper mode, where orders are virtual and market data is public.
    """
    return len((api_key or "").strip()) < 40


async def validate_binance_keys(api_key: str, api_secret: str) -> dict:
    """
    Validate that keys are real Binance Futures keys, not testnet keys.
    Used for live-mode credential checks.
    """
    import ccxt.async_support as ccxt_async

    result = {"valid": False, "is_testnet": False, "balance": 0.0, "message": ""}

    if not api_key or not api_secret:
        result["message"] = "API key and secret are required."
        return result

    if _looks_like_testnet_key(api_key) or _looks_like_testnet_key(api_secret):
        result["is_testnet"] = True
        result["message"] = _TESTNET_REJECTION_MSG
        return result

    exchange = ccxt_async.binanceusdm({
        "apiKey": api_key,
        "secret": api_secret,
        "options": {"defaultType": "future", "fetchCurrencies": False},
        "enableRateLimit": True,
    })

    try:
        balance_data = await asyncio.wait_for(
            exchange.fetch_balance({"type": "future"}),
            timeout=12,
        )
        usdt = balance_data.get("USDT", {})
        result.update({
            "valid": True,
            "balance": float(usdt.get("free", 0)),
            "message": "Connected",
        })
    except ccxt_async.AuthenticationError:
        result["message"] = (
            "Authentication failed. Please verify your API key and secret. "
            "Ensure Futures trading is enabled on your Binance account."
        )
    except ccxt_async.NetworkError as exc:
        if "testnet" in str(exc).lower():
            result["is_testnet"] = True
            result["message"] = _TESTNET_REJECTION_MSG
        else:
            result["message"] = f"Network error while connecting to Binance: {exc}"
    except asyncio.TimeoutError:
        result["message"] = "Connection to Binance timed out. Try again in a moment."
    except Exception as exc:
        result["message"] = f"Unexpected error: {exc}"
    finally:
        try:
            await exchange.close()
        except Exception:
            pass

    return result


def create_exchange(
    exchange_name: str = "binance",
    credentials: ExchangeCredentials = None,
) -> BaseExchange:
    """Create a live exchange using environment credentials. Used by CLI mode."""
    if credentials is None:
        credentials = settings.get_exchange_credentials(exchange_name)
    return BinanceExchange(credentials)


async def create_exchange_from_config(
    exchange_name: str,
    api_key: str,
    api_secret: str,
    testnet: bool = False,
    trading_mode: TradingMode = "paper",
    paper_balance: float = 10_000.0,
) -> BaseExchange:
    """
    Create the correct exchange for a managed bot run.

    Paper mode intentionally permits missing/testnet keys because it never sends
    private exchange orders. Live mode remains strict and requires real keys.
    """
    if exchange_name != "binance":
        raise ValueError(f"Unsupported exchange '{exchange_name}'. Only 'binance' is implemented.")

    if trading_mode not in ("paper", "live"):
        raise ValueError(f"Unknown trading_mode='{trading_mode}'. Use 'paper' or 'live'.")

    if trading_mode == "paper":
        credentials = ExchangeCredentials(
            api_key=api_key or "",
            secret=api_secret or "",
            testnet=False,
        )
        logger.info(
            f"[FACTORY] Creating PAPER exchange "
            f"(public market data, virtual orders, balance=${paper_balance:,.0f})"
        )
        return PaperExchange(credentials, initial_balance=paper_balance)

    if _looks_like_testnet_key(api_key) or _looks_like_testnet_key(api_secret):
        raise ValueError(_TESTNET_REJECTION_MSG)

    credentials = ExchangeCredentials(
        api_key=api_key,
        secret=api_secret,
        testnet=False,
    )

    logger.warning("[FACTORY] Creating LIVE exchange - REAL FUNDS WILL BE USED")
    return BinanceExchange(credentials)
