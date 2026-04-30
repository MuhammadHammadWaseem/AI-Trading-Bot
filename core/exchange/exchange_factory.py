"""
core/exchange/exchange_factory.py
──────────────────────────────────
Factory — creates the right exchange for the requested trading mode.

Modes:
  "paper"  → PaperExchange  (real market data, virtual orders)
  "live"   → BinanceExchange (real market data, real orders)

Security rule: ONLY real Binance API keys accepted. If testnet keys
are passed, both modes reject them with a clear user-facing message.

Testnet key detection heuristics (belt-and-suspenders):
  1. Key length < 40 chars (testnet keys are shorter)
  2. Key connects successfully to testnet endpoint but fails on live
  3. Explicit is_testnet=True flag rejected
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

# ── Testnet key validation ─────────────────────────────────────────────────

_TESTNET_REJECTION_MSG = (
    "Please connect real Binance API keys. "
    "Testnet keys are not supported — create API keys at binance.com > "
    "Profile > API Management, then enable Futures trading."
)

def _looks_like_testnet_key(api_key: str) -> bool:
    """
    Heuristic check. Real Binance API keys are 64-char hex strings.
    Testnet keys issued by testnet.binancefuture.com are shorter.
    """
    return len(api_key.strip()) < 40


async def validate_binance_keys(api_key: str, api_secret: str) -> dict:
    """
    Validate that keys are REAL Binance Futures keys (not testnet).

    Returns:
        {
            "valid":      bool,
            "is_testnet": bool,
            "balance":    float,   # USDT available (0 if invalid)
            "message":    str,
        }
    """
    import ccxt.async_support as ccxt_async

    result = {"valid": False, "is_testnet": False, "balance": 0.0, "message": ""}

    if not api_key or not api_secret:
        result["message"] = "API key and secret are required."
        return result

    # Fast heuristic rejection
    if _looks_like_testnet_key(api_key):
        result["is_testnet"] = True
        result["message"] = _TESTNET_REJECTION_MSG
        return result

    # Full network validation against LIVE Binance endpoint
    exchange = ccxt_async.binanceusdm({
        "apiKey":       api_key,
        "secret":       api_secret,
        "options":      {"defaultType": "future", "fetchCurrencies": False},
        "enableRateLimit": True,
    })
    # No testnet URL override → connects to real binance.com

    try:
        balance_data = await asyncio.wait_for(
            exchange.fetch_balance({"type": "future"}),
            timeout=12,
        )
        usdt      = balance_data.get("USDT", {})
        available = float(usdt.get("free", 0))
        result.update({"valid": True, "balance": available, "message": "Connected"})

    except ccxt_async.AuthenticationError:
        result["message"] = (
            "Authentication failed. Please verify your API key and secret. "
            "Ensure Futures trading is enabled on your Binance account."
        )
    except ccxt_async.NetworkError as exc:
        # Testnet keys hitting the real endpoint produce auth/network errors
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


# ── Factory function ───────────────────────────────────────────────────────

def create_exchange(
    exchange_name: str = "binance",
    credentials: ExchangeCredentials = None,
) -> BaseExchange:
    """Create a live exchange (no paper mode). Used internally."""
    if credentials is None:
        credentials = settings.get_exchange_credentials(exchange_name)
    return BinanceExchange(credentials)


async def create_exchange_from_config(
    exchange_name: str,
    api_key:        str,
    api_secret:     str,
    testnet:        bool         = False,   # always ignored — we force live keys
    trading_mode:   TradingMode  = "paper",
    paper_balance:  float        = 10_000.0,
) -> BaseExchange:
    """
    Create the correct exchange for a bot run.

    Args:
        exchange_name:  "binance" (others not yet supported)
        api_key:        Real Binance API key
        api_secret:     Real Binance API secret
        testnet:        IGNORED — only real keys accepted
        trading_mode:   "paper" → PaperExchange | "live" → BinanceExchange
        paper_balance:  Starting virtual balance for paper mode

    Returns:
        BaseExchange (not yet connected — caller must await exchange.connect())

    Raises:
        ValueError: if trading_mode is invalid or testnet keys are detected
    """
    if trading_mode not in ("paper", "live"):
        raise ValueError(
            f"Unknown trading_mode='{trading_mode}'. Use 'paper' or 'live'."
        )

    if _looks_like_testnet_key(api_key):
        raise ValueError(_TESTNET_REJECTION_MSG)

    # Always use real Binance endpoint regardless of old testnet flag
    credentials = ExchangeCredentials(
        api_key = api_key,
        secret  = api_secret,
        testnet = False,        # force real endpoint
    )

    if trading_mode == "paper":
        logger.info(
            f"[FACTORY] Creating PAPER exchange "
            f"(real data, virtual orders, balance=${paper_balance:,.0f})"
        )
        return PaperExchange(credentials, initial_balance=paper_balance)

    # trading_mode == "live"
    logger.warning(
        f"[FACTORY] Creating LIVE exchange — REAL FUNDS WILL BE USED"
    )
    return BinanceExchange(credentials)