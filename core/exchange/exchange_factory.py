"""
core/exchange/exchange_factory.py
──────────────────────────────────
Factory pattern — create exchange instances by name.
Adding a new exchange = add one entry here. Nothing else changes.
"""

from config.settings import settings, ExchangeCredentials
from core.exchange.base_exchange import BaseExchange
from core.exchange.binance_exchange import BinanceExchange


EXCHANGE_MAP = {
    "binance": BinanceExchange,
    # "okx":    OKXExchange,     ← add here when ready
    # "bybit":  BybitExchange,   ← add here when ready
}


def create_exchange(
    exchange_name: str = "binance",
    credentials: ExchangeCredentials = None,
) -> BaseExchange:
    """
    Create and return an exchange instance.

    Args:
        exchange_name: "binance" | "okx" | "bybit"
        credentials:   Optional override; uses settings default if None

    Returns:
        BaseExchange instance (not yet connected — call await .connect())
    """
    exchange_name = exchange_name.lower()

    if exchange_name not in EXCHANGE_MAP:
        raise ValueError(
            f"Exchange '{exchange_name}' not supported. "
            f"Available: {list(EXCHANGE_MAP.keys())}"
        )

    if credentials is None:
        credentials = settings.get_exchange_credentials(exchange_name)

    exchange_class = EXCHANGE_MAP[exchange_name]
    return exchange_class(credentials)


async def create_exchange_from_config(
    exchange_name: str,
    api_key:       str,
    api_secret:    str,
    testnet:       bool = True,
) -> BaseExchange:
    """
    Create an exchange using credentials supplied at runtime (from Laravel config).
    This is used by run_bot_managed.py where credentials come from the JSON config
    written by Laravel rather than from .env settings.
    """
    from config.settings import ExchangeCredentials
    creds = ExchangeCredentials(
        api_key = api_key,
        secret  = api_secret,
        testnet = testnet,
    )
    return create_exchange(exchange_name=exchange_name, credentials=creds)